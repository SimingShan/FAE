"""1D I-JEPA ViT — baseline encoder (faithful port of Meta's I-JEPA to 1D).

Model components used by ``scripts/train_jepa_vit.py`` and the diagnostics:

  - ``VisionTransformer1D``           patchify (patch 16, X=1024 -> 64 patches),
                                      fixed sin-cos pos-embed, pre-LN blocks.
  - ``VisionTransformerPredictor1D``  narrow predictor that fills mask tokens
                                      at target positions from context tokens.
  - ``apply_masks`` / ``sample_masks``  patch-index gather / disjoint sampling.

Training (in the script): target encoder = EMA copy, sees the FULL field;
context encoder sees only context patches; smooth-L1 between predictor output
and layer-normed target features. No reconstruction, no VICReg.

The representation used downstream is the TARGET branch, mean-pooled over
patch tokens: ``model(u, masks=None).mean(dim=1)``.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


def get_1d_sincos_pos_embed(embed_dim, num_pos):
    """Fixed 1D sin-cos positional embedding, (num_pos, embed_dim)."""
    assert embed_dim % 2 == 0
    pos = np.arange(num_pos, dtype=np.float32)
    omega = np.arange(embed_dim // 2, dtype=np.float32) / (embed_dim / 2.0)
    omega = 1.0 / (10000 ** omega)                              # (D/2,)
    out = np.einsum("p,d->pd", pos, omega)                      # (P, D/2)
    return np.concatenate([np.sin(out), np.cos(out)], axis=-1)  # (P, D)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)\
                          .permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                                  # (B, H, N, d)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1); attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden); self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim); self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim); self.attn = Attention(dim, num_heads, qkv_bias, drop, drop)
        self.norm2 = nn.LayerNorm(dim); self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed1D(nn.Module):
    def __init__(self, img_size=1024, patch_size=16, in_chans=1, embed_dim=384):
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = img_size // patch_size
        self.patch_size = patch_size
        self.proj = nn.Conv1d(in_chans, embed_dim,
                                kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, X) or (B, 1, X) -> (B, num_patches, embed_dim)
        if x.dim() == 2: x = x.unsqueeze(1)
        x = self.proj(x).transpose(1, 2)  # (B, P, D)
        return x


def apply_masks(x, masks):
    """x: (B, N, D), masks: (B, K) int indices -> (B, K, D)."""
    return torch.gather(x, dim=1, index=masks.unsqueeze(-1).expand(-1, -1, x.size(-1)))


def sample_masks(batch_size, num_patches, n_ctx, n_tgt, device):
    """Disjoint random context + target patch indices per batch item."""
    rand = torch.rand(batch_size, num_patches, device=device)
    perm = rand.argsort(dim=1)
    ctx = perm[:, :n_ctx]
    tgt = perm[:, n_ctx : n_ctx + n_tgt]
    return ctx, tgt


class VisionTransformer1D(nn.Module):
    """1D ViT encoder with optional patch masking (keep only context patches)."""
    def __init__(self, img_size=1024, patch_size=16, embed_dim=384,
                  depth=8, num_heads=6, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbed1D(img_size, patch_size, 1, embed_dim)
        num_patches = self.patch_embed.num_patches
        pe = get_1d_sincos_pos_embed(embed_dim, num_patches)
        self.pos_embed = nn.Parameter(torch.from_numpy(pe).float().unsqueeze(0),
                                         requires_grad=False)              # (1, P, D)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio)
                                       for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias); nn.init.ones_(m.weight)

    def forward(self, x, masks=None):
        """x: (B, X) field. masks: None or (B, K) int -> keep only those patches."""
        x = self.patch_embed(x)                  # (B, P, D)
        x = x + self.pos_embed                   # add pos before mask
        if masks is not None:
            x = apply_masks(x, masks)            # (B, K, D)
        for blk in self.blocks: x = blk(x)
        return self.norm(x)


class VisionTransformerPredictor1D(nn.Module):
    """Predicts target-position embeddings from context tokens."""
    def __init__(self, num_patches=64, embed_dim=384,
                  predictor_embed_dim=192, depth=6, num_heads=6, mlp_ratio=4.0):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        pe = get_1d_sincos_pos_embed(predictor_embed_dim, num_patches)
        self.predictor_pos_embed = nn.Parameter(torch.from_numpy(pe).float().unsqueeze(0),
                                                    requires_grad=False)
        self.blocks = nn.ModuleList([Block(predictor_embed_dim, num_heads, mlp_ratio)
                                       for _ in range(depth)])
        self.norm = nn.LayerNorm(predictor_embed_dim)
        self.proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias); nn.init.ones_(m.weight)

    def forward(self, ctx_tokens, masks_x, masks_pred):
        """
        ctx_tokens: (B, N_ctx, D_enc)
        masks_x:    (B, N_ctx) indices of context patches
        masks_pred: (B, N_tgt) indices of target patches
        returns:    (B, N_tgt, D_enc) predicted target features
        """
        B = ctx_tokens.size(0)
        x = self.predictor_embed(ctx_tokens)                        # (B, N_ctx, D_pred)
        pe = self.predictor_pos_embed.expand(B, -1, -1)             # (B, P, D_pred)
        x = x + apply_masks(pe, masks_x)
        tgt_pe = apply_masks(pe, masks_pred)                        # (B, N_tgt, D_pred)
        mask_toks = self.mask_token.expand(B, masks_pred.size(1), -1) + tgt_pe
        x = torch.cat([x, mask_toks], dim=1)                        # (B, N_ctx+N_tgt, D_pred)
        for blk in self.blocks: x = blk(x)
        x = self.norm(x)
        x = x[:, masks_x.size(1):]                                   # target tokens
        return self.proj(x)                                          # (B, N_tgt, D_enc)
