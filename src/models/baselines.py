"""Baseline encoders for the G1 1D family.

All target ~7M params for fair comparison with V3.

- MLPSparseAE  — variable-N set-pooled MLP (sparse encoder, comparable to V3)
- CNN1DAE      — 1D conv encoder-decoder (dense)
- MAE1DAE      — 1D Masked AutoEncoder ViT (dense, 75% mask during training)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. MLPSparseAE — variable-N sparse encoder via DeepSet/set-pool
# =====================================================================
class MLPSparseEncoder(nn.Module):
    """Per-sensor MLP → mean-pool → final MLP. Variable N."""
    def __init__(self, in_dim: int = 2, emb_dim: int = 1024, latent_dim: int = 320,
                  depth: int = 4):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, emb_dim)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(emb_dim),
                nn.Linear(emb_dim, emb_dim),
                nn.GELU(),
                nn.Linear(emb_dim, emb_dim),
            ))
        self.out_proj = nn.Linear(emb_dim, latent_dim)

    def forward(self, u, coords):
        """u: (B, N, 1)  coords: (B, N, D).  Returns (B, latent_dim)."""
        if coords.dim() == 2:
            coords = coords.unsqueeze(0).expand(u.size(0), -1, -1)
        x = torch.cat([u, coords], dim=-1)                      # (B, N, 1+D)
        x = self.in_proj(x)                                      # (B, N, emb)
        for blk in self.blocks:
            x = x + blk(x)                                       # residual
        pooled = x.mean(dim=1)                                   # (B, emb)
        return self.out_proj(pooled)                             # (B, latent_dim)


class MLPSparseDecoder(nn.Module):
    """Query-coord-conditioned MLP decoder. Per-query latent broadcast."""
    def __init__(self, latent_dim: int = 320, coord_dim: int = 1,
                  emb_dim: int = 1024, depth: int = 4):
        super().__init__()
        self.in_proj = nn.Linear(latent_dim + coord_dim, emb_dim)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(emb_dim),
                nn.Linear(emb_dim, emb_dim),
                nn.GELU(),
                nn.Linear(emb_dim, emb_dim),
            ))
        self.head = nn.Linear(emb_dim, 1)

    def forward(self, latent, query_coords):
        """latent: (B, latent_dim)  query_coords: (B, N_q, D)  → (B, N_q, 1)"""
        if query_coords.dim() == 2:
            query_coords = query_coords.unsqueeze(0).expand(latent.size(0), -1, -1)
        B, N_q, D = query_coords.shape
        l = latent.unsqueeze(1).expand(B, N_q, -1)
        x = torch.cat([l, query_coords], dim=-1)
        x = self.in_proj(x)
        for blk in self.blocks:
            x = x + blk(x)
        return self.head(x)


class MLPSparseAE(nn.Module):
    """Sparse MLP encoder-decoder (variable-N, set-pooled). ~7M params."""
    def __init__(self, coord_dim: int = 1, latent_dim: int = 320,
                  enc_emb: int = 640, dec_emb: int = 640,
                  enc_depth: int = 4, dec_depth: int = 4):
        super().__init__()
        self.encoder = MLPSparseEncoder(in_dim=1 + coord_dim, emb_dim=enc_emb,
                                          latent_dim=latent_dim, depth=enc_depth)
        self.decoder = MLPSparseDecoder(latent_dim=latent_dim, coord_dim=coord_dim,
                                          emb_dim=dec_emb, depth=dec_depth)
        self.latent_dim = latent_dim

    def forward(self, u, in_coords, query_coords):
        z = self.encoder(u, in_coords)
        pred = self.decoder(z, query_coords)
        return pred, z


# =====================================================================
# 2. CNN1DAE — 1D conv enc-dec (dense)
# =====================================================================
class CNN1DAE(nn.Module):
    """1D conv enc-dec for dense full-field reconstruction.

    Expects (B, 1, X). Latent is the bottleneck activation, mean-pooled across
    spatial dim for probe evaluation.
    """
    def __init__(self, channels=(128, 256, 512, 1024), kernel: int = 5,
                  latent_dim: int = 320):
        super().__init__()
        self.channels = channels
        pad = kernel // 2
        # Encoder: 4 conv-stride2 layers
        in_ch = 1
        enc_layers = []
        for c in channels:
            enc_layers.append(nn.Conv1d(in_ch, c, kernel, stride=2, padding=pad))
            enc_layers.append(nn.GELU())
            enc_layers.append(nn.BatchNorm1d(c))
            in_ch = c
        self.encoder_conv = nn.Sequential(*enc_layers)
        # After 4 stride-2 layers: 1024 → 64.  Bottleneck dim = channels[-1] * 64
        # Mean-pool to get latent.
        self.latent_proj = nn.Linear(channels[-1], latent_dim)
        # Decoder: 4 transposed conv layers
        dec_layers = []
        rev = list(reversed(channels)) + [1]
        for i in range(len(channels)):
            dec_layers.append(nn.ConvTranspose1d(rev[i], rev[i+1], kernel,
                                                    stride=2, padding=pad,
                                                    output_padding=1))
            if i < len(channels) - 1:
                dec_layers.append(nn.GELU())
                dec_layers.append(nn.BatchNorm1d(rev[i+1]))
        self.decoder_conv = nn.Sequential(*dec_layers)
        # decoder takes the unpooled latent feature map at the bottleneck spatial dim
        # We need to "unpool" back to the bottleneck spatial dim before decoding.
        self.unpool_dim = channels[-1]
        self.latent_dim = latent_dim
        # Project pooled latent → spatial latent for decoding
        self.expand_proj = nn.Linear(latent_dim, channels[-1])

    def encode(self, u):
        """u: (B, X) or (B, 1, X)  → bottleneck features (B, C, X//16)."""
        if u.dim() == 2:
            u = u.unsqueeze(1)
        return self.encoder_conv(u)                              # (B, C_last, X//16)

    def forward(self, u):
        """u: (B, 1, X) → recon (B, 1, X) and pooled latent (B, latent_dim)."""
        feats = self.encode(u)                                   # (B, C, X//16)
        pooled = feats.mean(dim=-1)                              # (B, C)
        z = self.latent_proj(pooled)                             # (B, latent_dim)
        # Decode: project back, broadcast spatially
        spatial_dim = feats.shape[-1]
        ld = self.expand_proj(z)                                 # (B, C)
        feats_rec = ld.unsqueeze(-1).expand(-1, -1, spatial_dim)
        recon = self.decoder_conv(feats_rec)
        return recon, z


# =====================================================================
# 3. MAE1DAE — 1D Masked Autoencoder ViT (dense)
# =====================================================================
class MAE1DAE(nn.Module):
    """1D ViT MAE: tokenize 1024-pt field into 64 patches of 16, mask 75%,
    reconstruct. Encoder embeds all patches; decoder reconstructs masked ones.

    For probe evaluation, use the encoder output mean-pooled across patches
    (no masking).
    """
    def __init__(self, x_len: int = 1024, patch_size: int = 16,
                  emb_dim: int = 384, depth: int = 3, num_heads: int = 4,
                  decoder_dim: int = 256, decoder_depth: int = 2,
                  mlp_ratio: int = 4, mask_ratio: float = 0.75,
                  latent_dim: int = 320):
        super().__init__()
        assert x_len % patch_size == 0
        self.x_len = x_len
        self.patch_size = patch_size
        self.n_patches = x_len // patch_size
        self.emb_dim = emb_dim
        self.mask_ratio = mask_ratio
        self.latent_dim = latent_dim

        # Encoder patch embedding
        self.patch_embed = nn.Linear(patch_size, emb_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, emb_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Encoder transformer
        self.enc_blocks = nn.ModuleList()
        for _ in range(depth):
            self.enc_blocks.append(nn.TransformerEncoderLayer(
                d_model=emb_dim, nhead=num_heads,
                dim_feedforward=emb_dim * mlp_ratio,
                batch_first=True, activation="gelu", norm_first=True))
        self.enc_norm = nn.LayerNorm(emb_dim)
        # Project encoder out to probe-friendly latent (also used for downstream)
        self.latent_proj = nn.Linear(emb_dim, latent_dim)

        # Decoder
        self.dec_proj = nn.Linear(emb_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.dec_pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, decoder_dim))
        nn.init.trunc_normal_(self.dec_pos_embed, std=0.02)
        self.dec_blocks = nn.ModuleList()
        for _ in range(decoder_depth):
            self.dec_blocks.append(nn.TransformerEncoderLayer(
                d_model=decoder_dim, nhead=4,
                dim_feedforward=decoder_dim * mlp_ratio,
                batch_first=True, activation="gelu", norm_first=True))
        self.dec_head = nn.Linear(decoder_dim, patch_size)

    def patchify(self, x):
        """(B, 1, X) → (B, n_patches, patch_size)."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, _, X = x.shape
        return x.view(B, self.n_patches, self.patch_size)

    def unpatchify(self, x):
        """(B, n_patches, patch_size) → (B, 1, X)."""
        return x.reshape(x.size(0), 1, self.x_len)

    def encode(self, x, mask_ratio: float = None):
        """Encode with random masking. Returns:
          enc_out: (B, n_kept, emb_dim) — encoder features for kept patches
          mask:    (B, n_patches) — 0 = kept, 1 = masked
          ids_keep: (B, n_kept)
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        B = x.size(0)
        patches = self.patchify(x)                              # (B, P, ps)
        emb = self.patch_embed(patches) + self.pos_embed         # (B, P, D)
        if mask_ratio > 0:
            n_keep = int(self.n_patches * (1 - mask_ratio))
            noise = torch.rand(B, self.n_patches, device=x.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_keep = ids_shuffle[:, :n_keep]
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            mask = torch.ones(B, self.n_patches, device=x.device)
            mask[:, :n_keep] = 0
            mask = torch.gather(mask, 1, ids_restore)
            emb_kept = torch.gather(emb, 1, ids_keep.unsqueeze(-1).expand(-1, -1, self.emb_dim))
        else:
            emb_kept = emb
            n_keep = self.n_patches
            ids_keep = torch.arange(self.n_patches, device=x.device)[None].expand(B, -1)
            ids_restore = ids_keep.clone()
            mask = torch.zeros(B, self.n_patches, device=x.device)
        # Prepend cls token
        cls = self.cls_token.expand(B, -1, -1)
        emb_kept = torch.cat([cls, emb_kept], dim=1)
        for blk in self.enc_blocks:
            emb_kept = blk(emb_kept)
        emb_kept = self.enc_norm(emb_kept)
        return emb_kept, mask, ids_restore

    def decode(self, enc_out, ids_restore):
        """Decode masked patches. enc_out includes cls token at index 0."""
        B = enc_out.size(0)
        x = self.dec_proj(enc_out)                              # (B, 1 + n_kept, dec_dim)
        # Separate cls
        cls = x[:, :1]
        x_kept = x[:, 1:]                                       # (B, n_kept, dec_dim)
        n_kept = x_kept.size(1)
        # Pad with mask tokens
        mask_tokens = self.mask_token.expand(B, self.n_patches - n_kept, -1)
        x_full = torch.cat([x_kept, mask_tokens], dim=1)        # (B, P, dec_dim)
        # Unshuffle
        x_full = torch.gather(x_full, 1,
                                ids_restore.unsqueeze(-1).expand(-1, -1, x_full.size(-1)))
        x_full = x_full + self.dec_pos_embed
        x_full = torch.cat([cls, x_full], dim=1)
        for blk in self.dec_blocks:
            x_full = blk(x_full)
        x_full = x_full[:, 1:]                                   # drop cls
        return self.dec_head(x_full)                             # (B, P, patch_size)

    def forward(self, x):
        """Training forward: encode with masking, decode, return (recon, mask)."""
        enc_out, mask, ids_restore = self.encode(x, self.mask_ratio)
        recon_patches = self.decode(enc_out, ids_restore)
        recon = self.unpatchify(recon_patches)
        return recon, mask

    def encode_full(self, x):
        """Encode without masking — for probe / classification evaluation."""
        enc_out, _, _ = self.encode(x, mask_ratio=0.0)
        # Mean-pool over patches (excluding cls)
        feat = enc_out[:, 1:].mean(dim=1)
        return self.latent_proj(feat)


# =====================================================================
# Smoke test (no GPU required)
# =====================================================================
if __name__ == "__main__":
    B, N, X = 4, 256, 1024

    print("=== MLPSparseAE ===")
    m = MLPSparseAE(coord_dim=1, latent_dim=320, enc_emb=1024, dec_emb=1024)
    n_par = sum(p.numel() for p in m.parameters())
    print(f"params: {n_par/1e6:.2f}M")
    u = torch.randn(B, N, 1)
    c = torch.rand(B, N, 1)
    cq = torch.rand(B, X, 1)
    pred, z = m(u, c, cq)
    print(f"  pred {tuple(pred.shape)}  latent {tuple(z.shape)}")

    print("\n=== CNN1DAE ===")
    m = CNN1DAE()
    n_par = sum(p.numel() for p in m.parameters())
    print(f"params: {n_par/1e6:.2f}M")
    x = torch.randn(B, 1, X)
    pred, z = m(x)
    print(f"  pred {tuple(pred.shape)}  latent {tuple(z.shape)}")

    print("\n=== MAE1DAE ===")
    m = MAE1DAE()
    n_par = sum(p.numel() for p in m.parameters())
    print(f"params: {n_par/1e6:.2f}M")
    x = torch.randn(B, 1, X)
    recon, mask = m(x)
    z = m.encode_full(x)
    print(f"  recon {tuple(recon.shape)}  mask {tuple(mask.shape)}  latent {tuple(z.shape)}")
