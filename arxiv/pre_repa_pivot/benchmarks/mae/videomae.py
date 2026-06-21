"""Video MAE — the temporal pixel baseline, pretrained FROM SCRATCH in our pipeline.

Extends our 2D MAE (mae.py) to video: Conv3d **tubelet** patch-embed over
(tubelet_size x patch x patch), **tube masking** (the same spatial patches masked
across all time, VideoMAE's signature), reconstruct the masked tubes' pixels.
Frozen-probe representation = mean of encoder tokens with no masking — identical
readout to AE/MAE/I-JEPA, so it sits in the same fair linear-probe comparison as
FAE-temporal. ~ViT-Tiny params.
"""
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from benchmarks.mae.mae import (get_2d_sincos_pos_embed,
                                 get_1d_sincos_pos_embed_from_grid)


def get_3d_sincos_pos_embed(embed_dim, t_size, grid_size):
    """Factorized 3D sin-cos: [temporal(D/2) | spatial(D/2)], t-major order."""
    assert embed_dim % 2 == 0
    spatial = get_2d_sincos_pos_embed(embed_dim // 2, grid_size)            # (S, D/2)
    temporal = get_1d_sincos_pos_embed_from_grid(embed_dim // 2,
                                                 np.arange(t_size, dtype=np.float32))  # (T, D/2)
    S = grid_size * grid_size
    pos = np.zeros((t_size, S, embed_dim), dtype=np.float32)
    pos[:, :, :embed_dim // 2] = temporal[:, None, :]
    pos[:, :, embed_dim // 2:] = spatial[None, :, :]
    return pos.reshape(t_size * S, embed_dim)


class VideoMAE(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_frames=16, tubelet_size=2,
                 in_chans=4, embed_dim=256, depth=6, num_heads=8,
                 decoder_embed_dim=192, decoder_depth=3, decoder_num_heads=6,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.t_grid = num_frames // tubelet_size
        self.s_grid = img_size // patch_size
        self.num_spatial = self.s_grid ** 2
        num_patches = self.t_grid * self.num_spatial
        self.num_patches = num_patches

        self.patch_embed = nn.Conv3d(in_chans, embed_dim,
                                     kernel_size=(tubelet_size, patch_size, patch_size),
                                     stride=(tubelet_size, patch_size, patch_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, tubelet_size * patch_size ** 2 * in_chans, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        self.pos_embed.data[0, 1:].copy_(torch.from_numpy(
            get_3d_sincos_pos_embed(self.pos_embed.shape[-1], self.t_grid, self.s_grid)))
        self.decoder_pos_embed.data[0, 1:].copy_(torch.from_numpy(
            get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.t_grid, self.s_grid)))
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        """(B,C,T,H,W) -> (B, num_patches, tubelet*p*p*C), token order t,h,w."""
        B, C, T, H, W = x.shape
        tp, p = self.tubelet_size, self.patch_size
        ti, hi, wi = T // tp, H // p, W // p
        x = x.reshape(B, C, ti, tp, hi, p, wi, p)
        x = torch.einsum('abcdefgh->acegdfhb', x)              # (B,ti,hi,wi,tp,p,p,C)
        return x.reshape(B, ti * hi * wi, tp * p * p * C)

    def tube_mask(self, x, mask_ratio):
        """x:(B,L,D), L=t_grid*S -> keep the same spatial subset across all time."""
        B, L, D = x.shape
        ti, S = self.t_grid, self.num_spatial
        x = x.view(B, ti, S, D)
        len_keep = int(S * (1 - mask_ratio))
        noise = torch.rand(B, S, device=x.device)
        ids_shuffle = noise.argsort(1)
        ids_restore = ids_shuffle.argsort(1)
        ids_keep = ids_shuffle[:, :len_keep]
        idx = ids_keep[:, None, :, None].expand(B, ti, len_keep, D)
        x_keep = torch.gather(x, 2, idx).reshape(B, ti * len_keep, D)
        mask = torch.ones(B, S, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)              # (B,S) spatial mask
        return x_keep, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)     # (B, L, embed)
        x = x + self.pos_embed[:, 1:, :]
        if mask_ratio > 0:
            x, mask, ids_restore = self.tube_mask(x, mask_ratio)
        else:
            mask, ids_restore = torch.zeros(x.size(0), self.num_spatial, device=x.device), None
        cls = self.cls_token + self.pos_embed[:, :1, :]
        x = torch.cat([cls.expand(x.size(0), -1, -1), x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x), mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        B, ti, S = x.size(0), self.t_grid, self.num_spatial
        cls, tok = x[:, :1, :], x[:, 1:, :].view(B, ti, -1, x.size(-1))   # (B,ti,len_keep,dD)
        len_keep = tok.size(2)
        mask_tokens = self.mask_token.expand(B, ti, S - len_keep, -1)
        tok = torch.cat([tok, mask_tokens], dim=2)                        # (B,ti,S,dD)
        idx = ids_restore[:, None, :, None].expand(B, ti, S, tok.size(-1))
        tok = torch.gather(tok, 2, idx).reshape(B, ti * S, -1)
        x = torch.cat([cls, tok], dim=1) + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        return self.decoder_pred(self.decoder_norm(x))[:, 1:, :]

    def forward_loss(self, x, pred, mask):
        target = self.patchify(x)
        loss = ((pred - target) ** 2).mean(dim=-1)                        # (B,L)
        full = mask[:, None, :].expand(-1, self.t_grid, -1).reshape(mask.size(0), -1)  # (B,L)
        denom = full.sum()
        return loss.mean() if denom == 0 else (loss * full).sum() / denom

    def forward(self, x, mask_ratio=0.9):
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        return self.forward_loss(x, pred, mask), pred, mask

    @torch.no_grad()
    def encode(self, x):
        """Frozen probe rep: mean of encoder tokens, no masking. (B, embed_dim)."""
        latent, _, _ = self.forward_encoder(x, mask_ratio=0.0)
        return latent[:, 1:, :].mean(dim=1)


def videomae_physics(img_size=224, num_frames=16, in_chans=4, patch_size=16,
                     tubelet_size=2, **kw):
    """~7M params (parity), tube-masked video MAE for 4-channel physics clips."""
    return VideoMAE(img_size=img_size, num_frames=num_frames, in_chans=in_chans,
                    patch_size=patch_size, tubelet_size=tubelet_size,
                    embed_dim=256, depth=6, num_heads=8,
                    decoder_embed_dim=192, decoder_depth=3, decoder_num_heads=6, **kw)
