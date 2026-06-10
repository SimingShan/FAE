"""FAE-NP — probabilistic Function AutoEncoder (functional Neural Process).

The probabilistic counterpart of the deterministic FAE (+VICReg):

  - ``FAEEncoder``              unchanged (cross-attn into latent tokens)
  - mean-pool L tokens -> single vector, so the KL is over ONE Gaussian
  - ``GaussianLatentHead``      (mu, logvar) of dim d_latent
  - ``LatentToContext``         z -> n_context decoder tokens (pure projection)
  - ``HeteroscedasticCViTDecoder``  per-query (mu_y, logvar_y)
  - loss: NP ELBO with KL(q(z|C,T) || q(z|C)) + per-slot free bits
    (see scripts/train_fae_np.py)

Why a single global z (not per-token): a per-token Gaussian feeding a
cross-attention decoder lets the decoder average noise across L=128 tokens
(effective noise / sqrt(L)), so the encoder can "win" with wide-uninformative
posteriors. One global z removes the averaging trick — noise must be explicit.

Test-time representation: ``encode_distribution(u_C, x_C)[0]`` (the mean),
shape (B, d_latent).

Class names were modernized from the original V4 implementation
(``V4`` -> ``FAENP``); module attribute names are unchanged, so existing
checkpoints load with ``strict=True``.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

from .fae import FAEEncoder, CViTBlock, fourier_features


# ----------------------------------------------------------------------
# Global Gaussian latent head: pool L tokens -> single Gaussian
# ----------------------------------------------------------------------
class GaussianLatentHead(nn.Module):
    """Tokens (B, L, d_model) -> mean-pool -> (mu, logvar) of dim d_latent.

    Init: logvar bias = -2 (std ~ 0.37, moderately tight initial posteriors).
    Clamp: (-6, 1) so var stays in [0.0025, 2.7] — the tight upper bound keeps
    the encoder from going wide-uninformative.
    """
    def __init__(self, d_model: int, d_latent: int):
        super().__init__()
        self.to_mu = nn.Linear(d_model, d_latent)
        self.to_logvar = nn.Linear(d_model, d_latent)
        nn.init.zeros_(self.to_logvar.weight)
        nn.init.constant_(self.to_logvar.bias, -2.0)
        self.d_latent = d_latent

    def forward(self, tokens):                             # (B, L, d_model)
        pooled = tokens.mean(dim=1)                        # (B, d_model)
        mu = self.to_mu(pooled)
        logvar = self.to_logvar(pooled).clamp(-6.0, 1.0)
        return mu, logvar                                  # each (B, d_latent)


# ----------------------------------------------------------------------
# z -> L context tokens (same KV interface the deterministic decoder has)
# ----------------------------------------------------------------------
class LatentToContext(nn.Module):
    """Project global z to n_context decoder tokens.

    Deliberately a pure projection — a learnable token bank with FiLM
    conditioning lets the decoder reconstruct without using z (FiLM init 0
    => context = bank => tiny gradient to z => encoder collapse).
    """
    def __init__(self, d_latent: int, n_context: int, dec_dim: int):
        super().__init__()
        self.proj = nn.Linear(d_latent, n_context * dec_dim)
        self.n_context = n_context
        self.dec_dim = dec_dim

    def forward(self, z):                                  # z: (B, d_latent)
        return self.proj(z).view(z.size(0), self.n_context, self.dec_dim)


# ----------------------------------------------------------------------
# Heteroscedastic decoder: (mu_y, logvar_y) per query coordinate
# ----------------------------------------------------------------------
class HeteroscedasticCViTDecoder(nn.Module):
    """Same body as the deterministic CViTDecoder; two output heads (mu, logvar)."""
    def __init__(self, emb_dim_in=320, dec_dim=320, n_freq=32, max_freq=32,
                  num_heads=4, num_blocks=2, mlp_mult=2, dropout=0.0,
                  latent_size=1, coord_dim=1):
        super().__init__()
        self.n_freq = n_freq
        self.max_freq = max_freq
        self.coord_dim = coord_dim

        self.output_buffer = nn.Parameter(torch.empty(latent_size, dec_dim))
        with torch.no_grad():
            self.output_buffer.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

        coord_feat_dim = 2 * coord_dim * n_freq
        self.query_proj = nn.Linear(coord_feat_dim + dec_dim, dec_dim)
        self.blocks = nn.ModuleList([
            CViTBlock(dec_dim, emb_dim_in, num_heads, mlp_mult, dropout)
            for _ in range(num_blocks)
        ])
        self.head_mu = nn.Linear(dec_dim, 1)
        self.head_logvar = nn.Linear(dec_dim, 1)
        nn.init.zeros_(self.head_logvar.weight)
        nn.init.constant_(self.head_logvar.bias, -2.0)

    def forward(self, latents, query_coords):
        if query_coords.dim() == 2:
            query_coords = query_coords.unsqueeze(0).expand(latents.size(0), -1, -1)
        B, N_q = query_coords.shape[:2]
        cf = fourier_features(query_coords, self.n_freq, self.max_freq)
        ob = self.output_buffer.unsqueeze(0).expand(B, N_q, -1)
        q = self.query_proj(torch.cat([cf, ob], dim=-1))
        for blk in self.blocks:
            q = blk(q, latents)
        mu_y = self.head_mu(q).squeeze(-1)
        logvar_y = self.head_logvar(q).squeeze(-1).clamp(-6.0, 2.0)
        return mu_y, logvar_y


# ----------------------------------------------------------------------
# FAE-NP: the full NP autoencoder
# ----------------------------------------------------------------------
class FAENP(nn.Module):
    def __init__(self, emb_dim=320, num_iter=4, depth_per_iter=4,
                  num_latents=128, num_cross_heads=4, num_self_heads=8,
                  n_freq=32, max_freq=32, coord_dim=1,
                  d_latent=256,
                  decoder_num_blocks=2, decoder_mlp_mult=2,
                  n_context_tokens=64, dec_dim=320):
        super().__init__()
        self.encoder = FAEEncoder(
            emb_dim=emb_dim, num_iter=num_iter, depth_per_iter=depth_per_iter,
            num_cross_heads=num_cross_heads, num_self_heads=num_self_heads,
            n_freq=n_freq, max_freq=max_freq, num_latents=num_latents,
            coord_dim=coord_dim)
        self.latent_head = GaussianLatentHead(emb_dim, d_latent)
        self.z_to_context = LatentToContext(d_latent, n_context_tokens, dec_dim)
        self.decoder = HeteroscedasticCViTDecoder(
            emb_dim_in=dec_dim, dec_dim=dec_dim,
            n_freq=n_freq, max_freq=max_freq,
            num_blocks=decoder_num_blocks, mlp_mult=decoder_mlp_mult,
            coord_dim=coord_dim)
        self.d_latent = d_latent
        self.emb_dim = emb_dim
        self.dec_dim = dec_dim
        self.n_context_tokens = n_context_tokens
        self.coord_dim = coord_dim

    def encode_distribution(self, u, in_coords):
        """u: (B, N_in, 1), in_coords: (N_in, D) or (B, N_in, D).
        Returns (mu, logvar), each (B, d_latent)."""
        tokens = self.encoder(u, in_coords)
        return self.latent_head(tokens)

    @staticmethod
    def reparam(mu, logvar):
        return mu + (0.5 * logvar).exp() * torch.randn_like(mu)

    def decode(self, z, query_coords):
        ctx = self.z_to_context(z)                          # (B, n_context, dec_dim)
        return self.decoder(ctx, query_coords)


# ----------------------------------------------------------------------
# Loss helpers
# ----------------------------------------------------------------------
def gaussian_kl(mu_q, logvar_q, mu_p, logvar_p):
    """Per-element KL(N(mu_q, var_q) || N(mu_p, var_p))."""
    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    return 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p).pow(2)) / var_p - 1.0)


def het_gaussian_nll(y, mu, logvar):
    """-log p(y | mu, var) for a diagonal Gaussian, incl. the 0.5*log(2*pi) constant."""
    return 0.5 * (logvar + (y - mu).pow(2) / logvar.exp() + math.log(2 * math.pi))


# Backward-compat aliases: original class names from the V4 implementation.
HeteroscedasticDecoderCViT = HeteroscedasticCViTDecoder
V4 = FAENP
