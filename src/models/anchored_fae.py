"""Anchored-FAE — the 'lead' redesign toward a discretization-consistent latent FIELD.

The vanilla FAE uses M *anonymous* learned latents (`self.latents`), so the representation has no
spatial geometry. Here each latent is SEEDED by a fixed anchor coordinate c_j, so latent j carries a
persistent physical meaning z_field(c_j) — the latent set becomes a coarse field on the domain.

Trained with reconstruction (the only objective that worked) + DISCRETIZATION-CONSISTENCY: two sensor
views of the same field must agree per-anchor (token-by-token at the shared anchor coords, not pooled).

Falsifiable claim (the NMI hook): the latent field CONVERGES as the observation set is refined —
‖z_n − z_dense‖ decreases monotonically in the sensor count n. `convergence_curve` measures it.
"""
import torch, torch.nn as nn
from src.models.fae import EncoderLayer, FAEDecoder, fourier_features
from src.data.well2d import make_coords_2d


class AnchoredFAEEncoder(nn.Module):
    def __init__(self, emb_dim=320, num_iter=4, depth_per_iter=4, num_cross_heads=4, num_self_heads=8,
                 n_freq=32, max_freq=32, val_dim=32, n_anchor_side=12, coord_dim=2, in_chans=3,
                 fourier_geometric=False):
        super().__init__()
        self.num_iter, self.n_freq, self.max_freq, self.fgeo = num_iter, n_freq, max_freq, fourier_geometric
        self.coord_dim, self.in_chans, self.emb_dim = coord_dim, in_chans, emb_dim
        self.n_anchor_side = n_anchor_side
        cfd = 2 * coord_dim * n_freq
        self.coord_proj = nn.Linear(cfd, emb_dim - val_dim)
        self.val_proj = nn.Linear(in_chans, val_dim)
        # spatial anchors: M latents on a fixed grid, each SEEDED by fourier(anchor_coord)
        self.register_buffer("anchor_coords", make_coords_2d(n_anchor_side))      # (M, 2) in [0,1]
        self.anchor_proj = nn.Linear(cfd, emb_dim)
        self.latent_residual = nn.Parameter(torch.zeros(1, n_anchor_side ** 2, emb_dim))  # learned refinement
        self.layer_1 = EncoderLayer(emb_dim, depth_per_iter, num_cross_heads, num_self_heads)
        self.layer_n = EncoderLayer(emb_dim, depth_per_iter, num_cross_heads, num_self_heads) if num_iter > 1 else None

    def latent_seed(self, B):
        af = fourier_features(self.anchor_coords, self.n_freq, self.max_freq, self.fgeo)  # (M, cfd)
        return (self.anchor_proj(af).unsqueeze(0) + self.latent_residual).expand(B, -1, -1)

    def forward(self, u, coords):                                # u:(B,N,C) coords:(N,D)/(B,N,D) -> (B,M,emb)
        if coords.dim() == 2:
            coords = coords.unsqueeze(0).expand(u.size(0), -1, -1)
        cf = fourier_features(coords, self.n_freq, self.max_freq, self.fgeo)
        tokens = torch.cat([self.coord_proj(cf), self.val_proj(u)], dim=-1)
        q = self.latent_seed(u.size(0))
        q = self.layer_1(q, tokens)
        for _ in range(self.num_iter - 1):
            q = self.layer_n(q, tokens)
        return q                                                 # the latent FIELD at the anchors


class AnchoredFAE(nn.Module):
    def __init__(self, emb_dim=320, n_anchor_side=12, in_chans=3, coord_dim=2, num_iter=4,
                 fourier_geometric=False):
        super().__init__()
        self.encoder = AnchoredFAEEncoder(emb_dim=emb_dim, n_anchor_side=n_anchor_side, in_chans=in_chans,
                                          coord_dim=coord_dim, num_iter=num_iter, fourier_geometric=fourier_geometric)
        self.decoder = FAEDecoder(emb_dim_in=emb_dim, dec_dim=emb_dim, coord_dim=coord_dim,
                                  out_chans=in_chans, fourier_geometric=fourier_geometric)
        self.emb_dim = emb_dim

    def encode(self, u, coords):
        return self.encoder(u, coords)

    def decode(self, z, coords):
        return self.decoder(z, coords)
