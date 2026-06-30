"""Unified coordinate-query decoder for the DOWNSTREAM pipeline (forecast / reconstruct).

Reads ANY frozen-encoder token set — FAE latents (B, n_latents, D) or MAE patch tokens (B, n_patches, D),
both D-wide — and renders the field at arbitrary query coordinates. ONE architecture for every encoder, so
the comparison isolates the LATENT, not the decoder. Trained FRESH on reconstruction (decoupled from the
operator, L-DeepONet-style). `dec_depth`/`dec_dim` make it deliberately LESS bottlenecked than the single
cross-layer SSL decoder (`FAEDecoder`) — but queries stay INDEPENDENT (no query-query attention), so it
remains a resolution-free neural-operator decode. Reuses fae.py CrossLayer + fourier_features verbatim.
"""
import torch
import torch.nn as nn
from src.models.fae import CrossLayer, fourier_features


class LatentDecoder(nn.Module):
    def __init__(self, token_dim=320, dec_dim=384, dec_depth=4, num_heads=6,
                 n_freq=32, max_freq=32, coord_dim=2, out_chans=1, dropout=0.0, fourier_geometric=False):
        super().__init__()
        self.n_freq, self.max_freq, self.coord_dim, self.fgeo = n_freq, max_freq, coord_dim, fourier_geometric
        self.output_buffer = nn.Parameter(torch.empty(1, dec_dim).normal_(0.0, 0.02).clamp_(-2.0, 2.0))
        self.query_proj = nn.Linear(2 * coord_dim * n_freq + dec_dim, dec_dim)
        self.cross_layers = nn.ModuleList(                       # stack of cross-layers; queries never attend to each other
            [CrossLayer(dec_dim, token_dim, num_heads, dropout) for _ in range(dec_depth)])
        self.norm = nn.LayerNorm(dec_dim)
        self.head = nn.Linear(dec_dim, out_chans)

    def forward(self, tokens, query_coords):
        """tokens: (B, N_tok, token_dim) from any frozen encoder; query_coords: (B, N_q, coord_dim) or (N_q, coord_dim)
        -> (B, N_q, out_chans)."""
        if query_coords.dim() == 2:
            query_coords = query_coords.unsqueeze(0).expand(tokens.size(0), -1, -1)
        B, N_q = query_coords.shape[:2]
        cf = fourier_features(query_coords, self.n_freq, self.max_freq, self.fgeo)
        q = self.query_proj(torch.cat([cf, self.output_buffer.unsqueeze(0).expand(B, N_q, -1)], dim=-1))
        for cl in self.cross_layers:
            q = cl(q, tokens)                                   # each query independently cross-attends the token set
        return self.head(self.norm(q))
