"""ARCHIVED FAE decoder variant — the CViT (query-self-attention) readout.

Removed from src/models/fae.py on the REPA pivot. Kept for the record.

WHY ARCHIVED: this decoder adds SELF-ATTENTION between query points (CViTBlock does
cross-attn -> MLP -> self-attn(queries) -> MLP). That couples the queries, so the decoded
value at a coordinate depends on which *other* coordinates are queried in the same batch —
which breaks the FAE's defining property: a resolution-free, pointwise neural-operator decode
(f(x) = D(z, x), evaluable independently at any x, consistent across query sets/resolutions).
The SenseiverDecoder (cross-attention only, each query independent) is the principled default
and the only one ever used in the final pipeline.

Dependencies (from fae.py): nn, CrossAttention, SelfAttention, Residual, fourier_features.
Not standalone-runnable; this file is documentation of the dead variant.
"""
import torch.nn as nn


class WiderMLP(nn.Module):
    """LayerNorm -> Linear(D, mult*D) -> GELU -> Linear(mult*D, D). Standard FFN. (Used only by CViTBlock.)"""
    def __init__(self, dim: int, mult: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * mult)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * mult, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(self.norm(x))))


class CViTBlock(nn.Module):
    """Cross-attn -> MLP -> self-attn(queries) -> MLP; all residual, pre-LN.
    The self-attention between queries lets neighboring query points coordinate (smoothness)
    — and is exactly what breaks resolution-free pointwise decoding."""
    def __init__(self, dim_q, dim_kv, num_heads, mlp_mult=2, dropout=0.0):
        super().__init__()
        self.cross     = Residual(CrossAttention(dim_q, dim_kv, num_heads, dropout), dropout)   # noqa: F821
        self.cross_mlp = Residual(WiderMLP(dim_q, mlp_mult), dropout)
        self.self_     = Residual(SelfAttention(dim_q, num_heads, dropout), dropout)            # noqa: F821
        self.self_mlp  = Residual(WiderMLP(dim_q, mlp_mult), dropout)

    def forward(self, q, kv):
        q = self.cross(q, kv); q = self.cross_mlp(q)
        q = self.self_(q);     q = self.self_mlp(q)
        return q


class CViTDecoder(nn.Module):
    """Stack of CViTBlocks — a deeper, Perceiver-IO / CViT-style readout (query-coupled)."""
    def __init__(self, emb_dim_in=320, dec_dim=320, n_freq=32, max_freq=32,
                  num_heads=4, num_blocks=2, mlp_mult=2, dropout=0.0,
                  latent_size=1, coord_dim=2, out_chans=1):
        import torch
        super().__init__()
        self.n_freq = n_freq; self.max_freq = max_freq; self.coord_dim = coord_dim
        self.output_buffer = nn.Parameter(torch.empty(latent_size, dec_dim))
        with torch.no_grad():
            self.output_buffer.normal_(0.0, 0.02).clamp_(-2.0, 2.0)
        coord_feat_dim = 2 * coord_dim * n_freq
        self.query_proj = nn.Linear(coord_feat_dim + dec_dim, dec_dim)
        self.blocks = nn.ModuleList([CViTBlock(dec_dim, emb_dim_in, num_heads, mlp_mult, dropout)
                                     for _ in range(num_blocks)])
        self.head = nn.Linear(dec_dim, out_chans)

    def forward(self, latents, query_coords):
        if query_coords.dim() == 2:
            query_coords = query_coords.unsqueeze(0).expand(latents.size(0), -1, -1)
        B, N_q = query_coords.shape[:2]
        cf = fourier_features(query_coords, self.n_freq, self.max_freq)                         # noqa: F821
        ob = self.output_buffer.unsqueeze(0).expand(B, N_q, -1)
        q = self.query_proj(torch.cat([cf, ob], dim=-1))                                        # noqa: F821
        for blk in self.blocks:
            q = blk(q, latents)
        return self.head(q)
