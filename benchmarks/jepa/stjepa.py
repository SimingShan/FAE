"""Spatio-temporal I-JEPA baseline (video JEPA).

Same I-JEPA objective as ijepa2d.py, but tokenizes a clip with a Conv3d **tubelet**
patch-embed (like VideoMAE) + 3D sin-cos pos-embed, so context/target masking and
the predictor operate over the space-TIME patch grid. Matched to VideoMAE's
tokenization and to ijepa2d's recipe → a fair multi-frame JEPA in the same harness.

Downstream probe representation: target-branch patch tokens, mean-pooled (`encode`).
Input: (B, C, T, H, W).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block

from benchmarks.mae.videomae import get_3d_sincos_pos_embed
from benchmarks.jepa.ijepa2d import apply_masks, sample_masks, _block


def sample_tube_block_masks(batch, t_grid, s_grid, n_targets=4, device="cpu"):
    """FAITHFUL V-JEPA-style spatio-temporal masking: sample I-JEPA spatial blocks
    (4 targets + 1 context-minus-targets) on the s_grid x s_grid plane, then extend
    each as a TUBE across all t_grid temporal positions. Shared across the batch."""
    tgt2d = torch.zeros(s_grid, s_grid, dtype=torch.bool, device=device)
    for _ in range(n_targets):
        tgt2d |= _block(s_grid, s_grid, 0.15, 0.2, 0.75, 1.5, device)
    ctx2d = _block(s_grid, s_grid, 0.85, 1.0, 1.0, 1.0, device) & ~tgt2d
    if not ctx2d.any():
        ctx2d = ~tgt2d
    S = s_grid * s_grid
    sp_ctx = ctx2d.flatten().nonzero(as_tuple=True)[0]
    sp_tgt = tgt2d.flatten().nonzero(as_tuple=True)[0]
    ci = torch.cat([sp_ctx + t * S for t in range(t_grid)])
    ti = torch.cat([sp_tgt + t * S for t in range(t_grid)])
    return ci[None].expand(batch, -1), ti[None].expand(batch, -1)


class PatchEmbed3D(nn.Module):
    def __init__(self, img_size, patch_size, num_frames, tubelet, in_chans, embed_dim):
        super().__init__()
        assert img_size % patch_size == 0 and num_frames % tubelet == 0
        self.t_grid = num_frames // tubelet
        self.s_grid = img_size // patch_size
        self.num_patches = self.t_grid * self.s_grid ** 2
        self.proj = nn.Conv3d(in_chans, embed_dim,
                              kernel_size=(tubelet, patch_size, patch_size),
                              stride=(tubelet, patch_size, patch_size))

    def forward(self, x):                          # (B, C, T, H, W) -> (B, P, D)
        return self.proj(x).flatten(2).transpose(1, 2)


class ViT3D(nn.Module):
    def __init__(self, img_size, patch_size, num_frames, tubelet, in_chans,
                 embed_dim=256, depth=6, num_heads=8, mlp_ratio=4.):
        super().__init__()
        self.patch_embed = PatchEmbed3D(img_size, patch_size, num_frames, tubelet, in_chans, embed_dim)
        pe = get_3d_sincos_pos_embed(embed_dim, self.patch_embed.t_grid, self.patch_embed.s_grid)
        self.pos_embed = nn.Parameter(torch.from_numpy(pe).float().unsqueeze(0), requires_grad=False)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        self.num_patches = self.patch_embed.num_patches

    def forward(self, x, keep_idx=None):
        x = self.patch_embed(x) + self.pos_embed
        if keep_idx is not None:
            x = apply_masks(x, keep_idx)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class Predictor3D(nn.Module):
    def __init__(self, t_grid, s_grid, embed_dim=256, pred_dim=128, depth=4, num_heads=4, mlp_ratio=4.):
        super().__init__()
        self.embed = nn.Linear(embed_dim, pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=.02)
        pe = get_3d_sincos_pos_embed(pred_dim, t_grid, s_grid)
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


class STJEPA(nn.Module):
    def __init__(self, img_size, num_frames, patch_size=16, tubelet=2, in_chans=4,
                 embed_dim=256, depth=6, num_heads=8, pred_dim=128, pred_depth=4, pred_heads=4):
        super().__init__()
        mk = lambda: ViT3D(img_size, patch_size, num_frames, tubelet, in_chans, embed_dim, depth, num_heads)
        self.encoder = mk()
        self.target = mk()
        self.target.load_state_dict(self.encoder.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.predictor = Predictor3D(self.encoder.patch_embed.t_grid, self.encoder.patch_embed.s_grid,
                                     embed_dim, pred_dim, pred_depth, pred_heads)
        self.num_patches = self.encoder.num_patches

    @torch.no_grad()
    def update_target(self, tau):
        for pe, pt in zip(self.encoder.parameters(), self.target.parameters()):
            pt.mul_(tau).add_(pe.data, alpha=1 - tau)

    def forward(self, clips, ctx_idx, tgt_idx):
        with torch.no_grad():
            h = self.target(clips)
            h = apply_masks(h, tgt_idx)
            h = F.layer_norm(h, (h.size(-1),))
        z = self.encoder(clips, keep_idx=ctx_idx)
        return self.predictor(z, ctx_idx, tgt_idx), h

    @torch.no_grad()
    def encode(self, clips):
        return self.target(clips).mean(dim=1)


def stjepa_physics(img_size=224, num_frames=4, in_chans=4, patch_size=16, tubelet=2, **kw):
    return STJEPA(img_size=img_size, num_frames=num_frames, patch_size=patch_size, tubelet=tubelet,
                  in_chans=in_chans, embed_dim=256, depth=6, num_heads=8,
                  pred_dim=128, pred_depth=4, pred_heads=4, **kw)
