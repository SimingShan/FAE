"""FAE — Function AutoEncoder.

A sparse, function-space autoencoder for PDE fields. The input is an arbitrary
set of coordinate-value pairs ``{(x_i, u_i)}`` (any sensor count N); the
encoder summarizes them into a fixed set of M learned latent tokens via
Perceiver-style iterative cross-attention; the decoder reads values back out
at arbitrary query coordinates via coordinate-conditioned cross-attention.

Components
----------
- ``FAEEncoder``      tokens = cross-attn(latents -> sensor tokens) + self-attn.
                      Senseiver-style weight sharing: a distinct first layer,
                      then one shared layer applied (num_iter - 1) times.
- ``FAEDecoder``  single cross-attention readout; each query decoded
                        independently (resolution-free neural-operator decode).
- ``FAE``               encoder + decoder;
                        forward: (u, in_coords, query_coords) -> (pred, tokens).

The pooled representation used everywhere downstream is ``tokens.mean(dim=1)``.

Training recipes live in ``scripts/train_fae.py``:
- ``fae_recon``   multi-count sparse reconstruction only.
- ``fae_vicreg``  two-view (independent sensor subsets) reconstruction +
                  VICReg similarity/variance/covariance on pooled latents
                  through an 8192^3 projector. This is the deterministic
                  core method.

Module attribute names are stable, so existing checkpoints load with ``strict=True``.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------
class Residual(nn.Module):
    """y = x + Dropout(module(x)) — Senseiver-style residual wrapper."""
    def __init__(self, module: nn.Module, dropout: float = 0.0):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, *args, **kwargs):
        return self.dropout(self.module(*args, **kwargs)) + args[0]


class MLP(nn.Module):
    """LayerNorm -> Linear(D, D) -> GELU -> Linear(D, D). No expansion."""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(self.norm(x))))


class CrossAttention(nn.Module):
    """Pre-LN multi-head cross-attention: q attends to kv."""
    def __init__(self, dim_q, dim_kv, num_heads, dropout=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim_q)
        self.norm_kv = nn.LayerNorm(dim_kv)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim_q, num_heads=num_heads,
            kdim=dim_kv, vdim=dim_kv,
            dropout=dropout, batch_first=True)

    def forward(self, q, kv):
        out, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv))
        return out


class SelfAttention(nn.Module):
    """Pre-LN multi-head self-attention."""
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True)

    def forward(self, x):
        x_n = self.norm(x)
        out, _ = self.attn(x_n, x_n, x_n)
        return out


class CrossLayer(nn.Module):
    """Residual cross-attention followed by a residual MLP."""
    def __init__(self, dim_q, dim_kv, num_heads, dropout=0.0):
        super().__init__()
        self.cross = Residual(CrossAttention(dim_q, dim_kv, num_heads, dropout), dropout)
        self.mlp   = Residual(MLP(dim_q), dropout)

    def forward(self, q, kv):
        q = self.cross(q, kv)
        q = self.mlp(q)
        return q


class SelfLayer(nn.Module):
    """Residual self-attention followed by a residual MLP."""
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.attn = Residual(SelfAttention(dim, num_heads, dropout), dropout)
        self.mlp  = Residual(MLP(dim), dropout)

    def forward(self, x):
        x = self.attn(x)
        x = self.mlp(x)
        return x


# ----------------------------------------------------------------------
# Coordinate features
# ----------------------------------------------------------------------
def fourier_features(coords, n_freq, max_freq=32):
    """Linear-spaced (Nyquist-capped) sin/cos features.

    coords: (..., D) in [0, 1] -> (..., 2 * D * n_freq).
    """
    device = coords.device
    freqs = torch.linspace(1.0, float(max_freq) / 2.0, n_freq,
                            device=device, dtype=torch.float32)
    args = coords.unsqueeze(-1) * freqs * math.pi
    sins = torch.sin(args).flatten(-2)
    coss = torch.cos(args).flatten(-2)
    return torch.cat([sins, coss], dim=-1)


# ----------------------------------------------------------------------
# Encoder
# ----------------------------------------------------------------------
class EncoderLayer(nn.Module):
    """One Perceiver iteration: cross-attend inputs, then self-attend latents."""
    def __init__(self, dim, depth_per_iter, num_cross_heads, num_self_heads, dropout=0.0):
        super().__init__()
        self.cross = CrossLayer(dim, dim, num_cross_heads, dropout)
        self.self_layers = nn.ModuleList([
            SelfLayer(dim, num_self_heads, dropout)
            for _ in range(depth_per_iter)
        ])

    def forward(self, q, kv):
        q = self.cross(q, kv)
        for sl in self.self_layers:
            q = sl(q)
        return q


class FAEEncoder(nn.Module):
    """Perceiver-style set encoder over coordinate-value tokens.

    A sensor token is concat[coord_proj(fourier(x_i)), val_proj(u_i)].
    M learned latents iteratively cross-attend the sensor tokens;
    ``layer_1`` is distinct, ``layer_n`` is shared across the remaining
    (num_iter - 1) iterations (Senseiver weight-sharing pattern).

    Cost is O(M * N) in the sensor count N — linear, unlike full self-attention.
    """
    def __init__(self, emb_dim=320, num_iter=4, depth_per_iter=4,
                  num_cross_heads=4, num_self_heads=8,
                  n_freq=32, max_freq=32, val_dim=32,
                  num_latents=128, dropout=0.0, coord_dim=2, in_chans=1):
        super().__init__()
        self.num_iter = num_iter
        self.n_freq = n_freq
        self.max_freq = max_freq
        self.coord_dim = coord_dim
        self.in_chans = in_chans

        coord_feat_dim = 2 * coord_dim * n_freq
        self.coord_proj = nn.Linear(coord_feat_dim, emb_dim - val_dim)
        self.val_proj = nn.Linear(in_chans, val_dim)

        self.latents = nn.Parameter(torch.empty(1, num_latents, emb_dim))
        with torch.no_grad():
            self.latents.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

        self.layer_1 = EncoderLayer(emb_dim, depth_per_iter,
                                       num_cross_heads, num_self_heads, dropout)
        if num_iter > 1:
            self.layer_n = EncoderLayer(emb_dim, depth_per_iter,
                                           num_cross_heads, num_self_heads, dropout)
        else:
            self.layer_n = None

    def forward(self, u, coords):
        """u: (B, N, 1), coords: (N, D) or (B, N, D) -> tokens (B, M, emb_dim)."""
        if coords.dim() == 2:
            coords = coords.unsqueeze(0).expand(u.size(0), -1, -1)
        cf = fourier_features(coords, self.n_freq, self.max_freq)
        c = self.coord_proj(cf)
        v = self.val_proj(u)
        tokens = torch.cat([c, v], dim=-1)

        q = self.latents.expand(u.size(0), -1, -1)
        q = self.layer_1(q, tokens)
        for _ in range(self.num_iter - 1):
            q = self.layer_n(q, tokens)
        return q


# ----------------------------------------------------------------------
# Decoders
# ----------------------------------------------------------------------
class FAEDecoder(nn.Module):
    """Single cross-attention readout at arbitrary query coordinates.

    Query = proj(concat[fourier(x_q), learned output buffer]); one CrossLayer
    against the latent tokens; linear head (no terminal LayerNorm).
    """
    def __init__(self, emb_dim_in=320, dec_dim=320, n_freq=32, max_freq=32,
                  num_heads=4, dropout=0.0, latent_size=1, coord_dim=2, out_chans=1):
        super().__init__()
        self.n_freq = n_freq
        self.max_freq = max_freq
        self.coord_dim = coord_dim

        self.output_buffer = nn.Parameter(torch.empty(latent_size, dec_dim))
        with torch.no_grad():
            self.output_buffer.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

        coord_feat_dim = 2 * coord_dim * n_freq
        self.query_proj = nn.Linear(coord_feat_dim + dec_dim, dec_dim)
        self.cross_layer = CrossLayer(dec_dim, emb_dim_in, num_heads, dropout)
        self.head = nn.Linear(dec_dim, out_chans)

    def forward(self, latents, query_coords, return_feats=False):
        if query_coords.dim() == 2:
            query_coords = query_coords.unsqueeze(0).expand(latents.size(0), -1, -1)
        B, N_q = query_coords.shape[:2]

        cf = fourier_features(query_coords, self.n_freq, self.max_freq)
        ob = self.output_buffer.unsqueeze(0).expand(B, N_q, -1)
        q = torch.cat([cf, ob], dim=-1)
        q = self.query_proj(q)
        q = self.cross_layer(q, latents)
        return q if return_feats else self.head(q)   # q = per-query feature (REPA alignment target)


# ----------------------------------------------------------------------
# Learned multi-query readout (optional)
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Latent predictor (delta-t conditioned token flow; used by train_fae.py predict/twoview)
# ----------------------------------------------------------------------
class TokenPredictor(nn.Module):
    """Delta-t-conditioned set->set predictor over the M latent tokens — a learned latent FLOW
    L_t -> L_{t+d} (an approximate evolution operator). Conditioning on the continuous gap d makes
    the model functional in time too, and starves trivial collapse (one fixed gap is too easy)."""
    def __init__(self, dim, depth=2, heads=8, dropout=0.0, dt_freq=16):
        super().__init__()
        self.dt_freq = dt_freq
        self.dt_mlp = nn.Sequential(nn.Linear(2 * dt_freq, dim), nn.GELU(), nn.Linear(dim, dim))
        self.layers = nn.ModuleList([SelfLayer(dim, heads, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, dim)

    def _dt_embed(self, dt):                                   # dt: (B,) in [0, 1]
        f = torch.arange(1, self.dt_freq + 1, device=dt.device, dtype=dt.dtype)
        a = dt[:, None] * f[None, :] * math.pi
        return self.dt_mlp(torch.cat([torch.sin(a), torch.cos(a)], dim=-1))    # (B, dim)

    def forward(self, x, dt):                                  # x (B,M,D), dt (B,)
        x = x + self._dt_embed(dt).unsqueeze(1)                # condition every token on delta
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))


# ----------------------------------------------------------------------
# Full autoencoder
# ----------------------------------------------------------------------
class FAE(nn.Module):
    """Function AutoEncoder: FAEEncoder + FAEDecoder.

    forward: (u, in_coords, query_coords) -> (pred, tokens)
      u            (B, N, 1)      sensor values
      in_coords    (N, D) or (B, N, D)
      query_coords (N_q, D) or (B, N_q, D)
      pred         (B, N_q, 1)
      tokens       (B, M, emb_dim)   pooled representation = tokens.mean(dim=1)
    """
    def __init__(self, emb_dim=320, num_iter=4, depth_per_iter=4,
                  num_cross_heads=4, num_self_heads=8,
                  n_freq=32, max_freq=32, val_dim=32,
                  dec_n_freq=32, dec_max_freq=32, dec_num_heads=4,
                  num_latents=128, dropout=0.0, coord_dim=2,
                  in_chans=1):
        super().__init__()
        self.encoder = FAEEncoder(
            emb_dim=emb_dim, num_iter=num_iter, depth_per_iter=depth_per_iter,
            num_cross_heads=num_cross_heads, num_self_heads=num_self_heads,
            n_freq=n_freq, max_freq=max_freq, val_dim=val_dim,
            num_latents=num_latents, dropout=dropout,
            coord_dim=coord_dim,
            in_chans=in_chans)
        self.decoder = FAEDecoder(
            emb_dim_in=emb_dim, dec_dim=emb_dim,
            n_freq=dec_n_freq, max_freq=dec_max_freq,
            num_heads=dec_num_heads, dropout=dropout,
            latent_size=1, coord_dim=coord_dim, out_chans=in_chans)
        self.emb_dim = emb_dim
        self.num_latents = num_latents
        self.coord_dim = coord_dim

    def represent(self, tokens):
        """Pooled representation used downstream: mean over the latent tokens."""
        return tokens.mean(dim=1)                        # (B, emb_dim)

    def encode_tokens(self, u, in_coords):
        """Sensor values (B, N, C) + coords (N, coord_dim) -> latent tokens (B, M, emb_dim)."""
        return self.encoder(u, in_coords)

    def forward(self, u, in_coords, query_coords):
        tokens = self.encode_tokens(u, in_coords)
        pred = self.decoder(tokens, query_coords)
        return pred, tokens


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, N, N_q = 2, 256, 4096

    m = FAE(num_latents=128, coord_dim=1).to(device)
    n_par = sum(p.numel() for p in m.parameters())
    print(f"FAE (M=128, num_iter=4, depth_per_iter=4, coord_dim=1): {n_par/1e6:.3f}M params")

    coords_in = torch.rand(B, N, 1, device=device)
    coords_q = torch.rand(B, N_q, 1, device=device)
    u = torch.randn(B, N, 1, device=device)

    pred, toks = m(u, coords_in, coords_q)
    print(f"  forward: pred={tuple(pred.shape)}  tokens={tuple(toks.shape)}")

    for n_test in [64, 128, 512, 2048]:
        c_in = torch.rand(B, n_test, 1, device=device)
        u_test = torch.randn(B, n_test, 1, device=device)
        p_test, _ = m(u_test, c_in, coords_q)
        assert p_test.shape == (B, N_q, 1), f"failed at N={n_test}"
    print("  variable sensor count N in [64, 128, 512, 2048] OK")
