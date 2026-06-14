"""MAE (and AE) baseline — clean, faithful port of Kaiming He's MAE.

Source: facebookresearch/mae `models_mae.py` + `util/pos_embed.py`
(cloned to external/mae). The masking, encoder, decoder, and patch-MSE loss
are unchanged from the original. Only three things are adapted:

  1. `in_chans` is generalized (the original hard-codes 3) so it runs on our
     4-channel physics fields (pressure, tracer/density, v_x, v_y).
  2. timm 1.0.25's `Block` no longer takes `qk_scale`, so that kwarg is dropped.
  3. AE mode: `mask_ratio=0` performs no masking and the loss is taken over ALL
     patches (the original divides by mask.sum(), which is 0 when nothing is
     masked). This gives a clean ViT autoencoder baseline sharing the MAE
     backbone — so AE-vs-MAE isolates the masking signal.

Downstream representation for the frozen probe: `encode(imgs)` = mean of the
encoder's patch tokens with no masking (standard MAE linear-probe readout).
"""
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block


# ----------------------------------------------------------------------
# 2D sin-cos positional embedding (verbatim from mae/util/pos_embed.py,
# with the deprecated np.float -> float fix)
# ----------------------------------------------------------------------
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0).reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with a ViT backbone (Kaiming He et al., 2021)."""
    def __init__(self, img_size=128, patch_size=16, in_chans=4,
                  embed_dim=384, depth=6, num_heads=6,
                  decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=8,
                  mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                  norm_pix_loss=False):
        super().__init__()
        self.in_chans = in_chans
        # --- encoder ---
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --- decoder ---
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
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

    def patchify(self, imgs):
        """(N, C, H, W) -> (N, L, p*p*C)."""
        p = self.patch_embed.patch_size[0]
        c = self.in_chans
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        return x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * c))

    def unpatchify(self, x):
        """(N, L, p*p*C) -> (N, C, H, W)."""
        p = self.patch_embed.patch_size[0]
        c = self.in_chans
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x), mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        return self.decoder_pred(x)[:, 1:, :]

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
        loss = ((pred - target) ** 2).mean(dim=-1)            # [N, L]
        denom = mask.sum()
        if denom == 0:                                         # AE mode: all patches
            return loss.mean()
        return (loss * mask).sum() / denom                     # MAE: masked patches

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    @torch.no_grad()
    def encode(self, imgs):
        """Frozen representation for the downstream probe: mean of encoder
        patch tokens (no masking), excluding the cls token."""
        latent, _, _ = self.forward_encoder(imgs, mask_ratio=0.0)
        return latent[:, 1:, :].mean(dim=1)                    # (N, embed_dim)


# ----------------------------------------------------------------------
# Factories. mae_physics = MAE baseline (mask 0.75); ae_physics = AE (mask 0).
# ~7M params at the defaults below (parity with FAE/JEPA) — see smoke test.
# ----------------------------------------------------------------------
def mae_physics(img_size=128, in_chans=4, patch_size=16, **kw):
    # ~7M params (parity with FAE/JEPA) at 128^2, 4 channels.
    return MaskedAutoencoderViT(
        img_size=img_size, patch_size=patch_size, in_chans=in_chans,
        embed_dim=256, depth=6, num_heads=8,
        decoder_embed_dim=192, decoder_depth=3, decoder_num_heads=6, **kw)


def ae_physics(img_size=128, in_chans=4, patch_size=16, **kw):
    """Same backbone as mae_physics; use with mask_ratio=0.0 -> ViT autoencoder."""
    return mae_physics(img_size=img_size, in_chans=in_chans, patch_size=patch_size, **kw)
