"""Single-frame 2D I-JEPA baseline.

I-JEPA (Assran et al. 2023) was designed for single images — this is the 2D
counterpart of our faithful 1D port (`src/models/jepa_vit.py`), for the
single-frame physics experiment. (The spatio-temporal JEPA is the helenqu
3D-conv model in `external/`, which downsamples time with Conv3d — a different
regime; this one is the original image-I-JEPA recipe.)

Recipe (unchanged from the original / our 1D port):
  - ViT encoder over 2D patches, fixed 2D sin-cos pos-embed.
  - Target encoder = EMA copy; sees the FULL image, target features are taken
    at the target patches and LayerNorm'd.
  - Context encoder sees only context patches; a narrow predictor fills mask
    tokens at the target positions and predicts their features.
  - smooth-L1 loss, per-iteration EMA ramp. No pixel reconstruction.

Downstream probe representation: target-branch patch tokens, mean-pooled
(`encode`).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from benchmarks.mae.mae import get_2d_sincos_pos_embed


class PatchEmbed2D(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_chans=4, embed_dim=256):
        super().__init__()
        assert img_size % patch_size == 0
        self.grid = img_size // patch_size
        self.num_patches = self.grid ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):                       # (B, C, H, W) -> (B, P, D)
        return self.proj(x).flatten(2).transpose(1, 2)


def apply_masks(x, idx):
    """x: (B, P, D), idx: (B, K) -> (B, K, D)."""
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))


def sample_masks(batch, num_patches, n_ctx, n_tgt, device):
    """Disjoint random context + target patch indices per item (simplified — kept for ref)."""
    perm = torch.rand(batch, num_patches, device=device).argsort(dim=1)
    return perm[:, :n_ctx], perm[:, n_ctx:n_ctx + n_tgt]


def _block(gh, gw, smin, smax, armin, armax, device):
    """A contiguous rectangular block of patches: area=scale*GH*GW, given aspect ratio."""
    s = (smin + (smax - smin) * torch.rand(1, device=device)).item()
    ar = (armin + (armax - armin) * torch.rand(1, device=device)).item()
    area = s * gh * gw
    h = max(1, min(int(round((area * ar) ** 0.5)), gh))
    w = max(1, min(int(round((area / ar) ** 0.5)), gw))
    top = torch.randint(0, gh - h + 1, (1,), device=device).item()
    left = torch.randint(0, gw - w + 1, (1,), device=device).item()
    m = torch.zeros(gh, gw, dtype=torch.bool, device=device)
    m[top:top + h, left:left + w] = True
    return m


def sample_block_masks(batch, gh, gw, n_targets=4, device="cpu"):
    """FAITHFUL I-JEPA multi-block masking (Assran et al. 2023): 4 target blocks
    (scale 0.15-0.2, aspect 0.75-1.5), context = one large block (scale 0.85-1.0)
    MINUS the target region. Shared across the batch (as in the I-JEPA collator)."""
    tgt = torch.zeros(gh, gw, dtype=torch.bool, device=device)
    for _ in range(n_targets):
        tgt |= _block(gh, gw, 0.15, 0.2, 0.75, 1.5, device)
    ctx = _block(gh, gw, 0.85, 1.0, 1.0, 1.0, device) & ~tgt
    ci = ctx.flatten().nonzero(as_tuple=True)[0]
    ti = tgt.flatten().nonzero(as_tuple=True)[0]
    if ci.numel() == 0:
        ci = (~tgt).flatten().nonzero(as_tuple=True)[0]
    return ci[None].expand(batch, -1), ti[None].expand(batch, -1)


class ViT2D(nn.Module):
    """2D ViT encoder with optional patch keep-mask."""
    def __init__(self, img_size=128, patch_size=16, in_chans=4,
                  embed_dim=256, depth=6, num_heads=8, mlp_ratio=4.):
        super().__init__()
        self.patch_embed = PatchEmbed2D(img_size, patch_size, in_chans, embed_dim)
        P = self.patch_embed.num_patches
        pe = get_2d_sincos_pos_embed(embed_dim, self.patch_embed.grid, cls_token=False)
        self.pos_embed = nn.Parameter(torch.from_numpy(pe).float().unsqueeze(0), requires_grad=False)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        self.num_patches = P

    def forward(self, x, keep_idx=None):
        x = self.patch_embed(x) + self.pos_embed
        if keep_idx is not None:
            x = apply_masks(x, keep_idx)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class Predictor2D(nn.Module):
    """Predicts target-patch features from context tokens."""
    def __init__(self, num_patches, embed_dim=256, pred_dim=128, depth=4, num_heads=4, mlp_ratio=4.):
        super().__init__()
        self.embed = nn.Linear(embed_dim, pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=.02)
        pe = get_2d_sincos_pos_embed(pred_dim, int(num_patches ** .5), cls_token=False)
        self.pos = nn.Parameter(torch.from_numpy(pe).float().unsqueeze(0), requires_grad=False)
        self.blocks = nn.ModuleList([Block(pred_dim, num_heads, mlp_ratio, qkv_bias=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(pred_dim)
        self.proj = nn.Linear(pred_dim, embed_dim)

    def forward(self, ctx_tokens, ctx_idx, tgt_idx):
        B = ctx_tokens.size(0)
        x = self.embed(ctx_tokens) + apply_masks(self.pos.expand(B, -1, -1), ctx_idx)
        tgt = self.mask_token.expand(B, tgt_idx.size(1), -1) + apply_masks(self.pos.expand(B, -1, -1), tgt_idx)
        x = torch.cat([x, tgt], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)[:, ctx_idx.size(1):]
        return self.proj(x)


class IJEPA2D(nn.Module):
    """Bundles context encoder, EMA target encoder, predictor for I-JEPA."""
    def __init__(self, img_size=128, patch_size=16, in_chans=4,
                  embed_dim=256, depth=6, num_heads=8,
                  pred_dim=128, pred_depth=4, pred_heads=4):
        super().__init__()
        self.encoder = ViT2D(img_size, patch_size, in_chans, embed_dim, depth, num_heads)
        self.target = ViT2D(img_size, patch_size, in_chans, embed_dim, depth, num_heads)
        self.target.load_state_dict(self.encoder.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.predictor = Predictor2D(self.encoder.num_patches, embed_dim, pred_dim, pred_depth, pred_heads)
        self.num_patches = self.encoder.num_patches

    @torch.no_grad()
    def update_target(self, tau):
        for pe, pt in zip(self.encoder.parameters(), self.target.parameters()):
            pt.mul_(tau).add_(pe.data, alpha=1 - tau)

    def forward(self, imgs, ctx_idx, tgt_idx):
        """Returns (pred, target_feats) for smooth_l1_loss."""
        import torch.nn.functional as F
        with torch.no_grad():
            h = self.target(imgs)                       # full image, all patches
            h = apply_masks(h, tgt_idx)
            h = F.layer_norm(h, (h.size(-1),))
        z = self.encoder(imgs, keep_idx=ctx_idx)
        pred = self.predictor(z, ctx_idx, tgt_idx)
        return pred, h

    @torch.no_grad()
    def encode(self, imgs):
        """Frozen probe representation: target-branch patch tokens, mean-pooled."""
        return self.target(imgs).mean(dim=1)            # (B, embed_dim)


def ijepa2d_physics(img_size=128, in_chans=4, patch_size=16, **kw):
    """~7M params (encoder) parity, single-frame 2D I-JEPA."""
    return IJEPA2D(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                    embed_dim=256, depth=6, num_heads=8,
                    pred_dim=128, pred_depth=4, pred_heads=4, **kw)
