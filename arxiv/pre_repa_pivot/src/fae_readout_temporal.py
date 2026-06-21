"""ARCHIVED FAE components — the learned readout and the temporal-aggregate path.

Removed from src/models/fae.py on the REPA pivot (never reachable by current defaults). Kept for record.

- QueryReadout: an end-to-end-trained K-query attentive readout (representation = flatten of K query
  tokens), an alternative to mean-pooling the latent tokens. Gated by FAE(readout_queries>0); always 0
  in practice, so represent() mean-pools. The attentive-probe eval that motivated it is also archived.
- TemporalAggregator: the time_mode="aggregate" path — per-frame encode then temporal self-attention +
  query pool. Never used; temporal is handled the 'coord' way (time as a 3rd input coordinate, coord_dim=3).

Dependencies (from fae.py): nn, torch, CrossLayer, SelfLayer. Documentation of dead variants, not runnable.
"""
import torch
import torch.nn as nn


class QueryReadout(nn.Module):
    """K learnable query tokens cross-attend the M encoder tokens -> (B, K, dim). Representation =
    flatten(forward(tokens)). Idea: a readout trained under SSL pressure can expose capacity that the
    mean-pool discards. In practice readout_queries stayed 0 (mean-pool), so this was never used."""
    def __init__(self, dim, num_queries=8, num_heads=4, depth=1, dropout=0.0):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(1, num_queries, dim))
        with torch.no_grad():
            self.queries.normal_(0.0, 0.02).clamp_(-2.0, 2.0)
        self.layers = nn.ModuleList([CrossLayer(dim, dim, num_heads, dropout) for _ in range(depth)])  # noqa: F821
        self.num_queries = num_queries

    def forward(self, tokens):                  # (B, M, dim) -> (B, K, dim)
        q = self.queries.expand(tokens.size(0), -1, -1)
        for layer in self.layers:
            q = layer(q, tokens)
        return q


class TemporalAggregator(nn.Module):
    """time_mode='aggregate': per-frame latents (B, T, M, dim) -> temporal self-attention over T ->
    learned-query cross-attention pool -> (B, M, dim). The default 'coord' mode handles time as a third
    input coordinate (x, y, t) jointly in the encoder, so this was never used."""
    def __init__(self, dim, num_heads=8, depth=1, dropout=0.0):
        super().__init__()
        self.frames = nn.ModuleList([SelfLayer(dim, num_heads, dropout) for _ in range(depth)])  # noqa: F821
        self.query = nn.Parameter(torch.empty(1, 1, dim))
        with torch.no_grad():
            self.query.normal_(0.0, 0.02)
        self.pool = CrossLayer(dim, dim, num_heads, dropout)  # noqa: F821

    def forward(self, z):                                   # z: (B, T, M, dim)
        B, T, M, D = z.shape
        z = z.permute(0, 2, 1, 3).reshape(B * M, T, D)
        for layer in self.frames:
            z = layer(z)
        q = self.query.expand(B * M, -1, -1)
        return self.pool(q, z).reshape(B, M, D)
