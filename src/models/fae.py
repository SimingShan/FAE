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
- ``SenseiverDecoder``  single cross-attention readout (default).
- ``CViTDecoder``       deeper readout: N blocks of
                        [cross-attn -> MLP -> self-attn(queries) -> MLP].
- ``FAE``               encoder + decoder;
                        forward: (u, in_coords, query_coords) -> (pred, tokens).

The pooled representation used everywhere downstream is ``tokens.mean(dim=1)``.

Training recipes live in ``scripts/train_fae.py``:
- ``fae_recon``   multi-count sparse reconstruction only.
- ``fae_vicreg``  two-view (independent sensor subsets) reconstruction +
                  VICReg similarity/variance/covariance on pooled latents
                  through an 8192^3 projector. This is the deterministic
                  core method.

Note: class names were modernized from the original V3 implementation
(``PerceiverSparseAEV3`` -> ``FAE`` etc.); module attribute names are
unchanged, so existing checkpoints load with ``strict=True``.
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


class SenseiverMLP(nn.Module):
    """LayerNorm -> Linear(D, D) -> GELU -> Linear(D, D). No expansion."""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(self.norm(x))))


class WiderMLP(nn.Module):
    """LayerNorm -> Linear(D, mult*D) -> GELU -> Linear(mult*D, D). Standard FFN."""
    def __init__(self, dim: int, mult: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * mult)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * mult, dim)

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
        self.mlp   = Residual(SenseiverMLP(dim_q), dropout)

    def forward(self, q, kv):
        q = self.cross(q, kv)
        q = self.mlp(q)
        return q


class SelfLayer(nn.Module):
    """Residual self-attention followed by a residual MLP."""
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.attn = Residual(SelfAttention(dim, num_heads, dropout), dropout)
        self.mlp  = Residual(SenseiverMLP(dim), dropout)

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


# Backward-compat aliases (older scripts/notebooks).
_fourier_features_linear = fourier_features
_fourier_features_2d_linear = fourier_features


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
class SenseiverDecoder(nn.Module):
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

    def forward(self, latents, query_coords):
        if query_coords.dim() == 2:
            query_coords = query_coords.unsqueeze(0).expand(latents.size(0), -1, -1)
        B, N_q = query_coords.shape[:2]

        cf = fourier_features(query_coords, self.n_freq, self.max_freq)
        ob = self.output_buffer.unsqueeze(0).expand(B, N_q, -1)
        q = torch.cat([cf, ob], dim=-1)
        q = self.query_proj(q)
        q = self.cross_layer(q, latents)
        return self.head(q)


class CViTBlock(nn.Module):
    """Cross-attn -> MLP -> self-attn(queries) -> MLP; all residual, pre-LN.

    The self-attention between queries lets neighboring query points coordinate
    (smoothness); the MLPs use standard 2x expansion.
    """
    def __init__(self, dim_q, dim_kv, num_heads, mlp_mult=2, dropout=0.0):
        super().__init__()
        self.cross    = Residual(CrossAttention(dim_q, dim_kv, num_heads, dropout), dropout)
        self.cross_mlp = Residual(WiderMLP(dim_q, mlp_mult), dropout)
        self.self_    = Residual(SelfAttention(dim_q, num_heads, dropout), dropout)
        self.self_mlp = Residual(WiderMLP(dim_q, mlp_mult), dropout)

    def forward(self, q, kv):
        q = self.cross(q, kv)
        q = self.cross_mlp(q)
        q = self.self_(q)
        q = self.self_mlp(q)
        return q


class CViTDecoder(nn.Module):
    """Stack of CViTBlocks — a deeper, Perceiver-IO / CViT-style readout."""
    def __init__(self, emb_dim_in=320, dec_dim=320, n_freq=32, max_freq=32,
                  num_heads=4, num_blocks=2, mlp_mult=2, dropout=0.0,
                  latent_size=1, coord_dim=2, out_chans=1):
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
        self.head = nn.Linear(dec_dim, out_chans)

    def forward(self, latents, query_coords):
        if query_coords.dim() == 2:
            query_coords = query_coords.unsqueeze(0).expand(latents.size(0), -1, -1)
        B, N_q = query_coords.shape[:2]
        cf = fourier_features(query_coords, self.n_freq, self.max_freq)
        ob = self.output_buffer.unsqueeze(0).expand(B, N_q, -1)
        q = self.query_proj(torch.cat([cf, ob], dim=-1))
        for blk in self.blocks:
            q = blk(q, latents)
        return self.head(q)


# ----------------------------------------------------------------------
# Learned multi-query readout (optional)
# ----------------------------------------------------------------------
class QueryReadout(nn.Module):
    """K learnable query tokens cross-attend the M encoder tokens -> (B, K, dim).

    The mean-pool readout discards capacity the encoder retains (the latent
    tokens carry intrinsic dim ~22, the mean ~10) and that capacity is NOT
    linearly accessible post-hoc. A readout trained end-to-end UNDER the SSL
    pressure can reorganize the token set into K*dim linearly-usable
    coordinates. Representation = flatten(forward(tokens)).
    """
    def __init__(self, dim, num_queries=8, num_heads=4, depth=1, dropout=0.0):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(1, num_queries, dim))
        with torch.no_grad():
            self.queries.normal_(0.0, 0.02).clamp_(-2.0, 2.0)
        self.layers = nn.ModuleList([
            CrossLayer(dim, dim, num_heads, dropout) for _ in range(depth)])
        self.num_queries = num_queries

    def forward(self, tokens):                  # (B, M, dim) -> (B, K, dim)
        q = self.queries.expand(tokens.size(0), -1, -1)
        for layer in self.layers:
            q = layer(q, tokens)
        return q


# ----------------------------------------------------------------------
# Full autoencoder
# ----------------------------------------------------------------------
class FAE(nn.Module):
    """Function AutoEncoder: FAEEncoder + configurable decoder.

    decoder_kind:
      - "senseiver" (default): single cross-attention readout.
      - "cvit": decoder_num_blocks x CViTBlock readout.

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
                  decoder_kind="senseiver", decoder_num_blocks=2,
                  decoder_mlp_mult=2, readout_queries=0, in_chans=1):
        super().__init__()
        self.encoder = FAEEncoder(
            emb_dim=emb_dim, num_iter=num_iter, depth_per_iter=depth_per_iter,
            num_cross_heads=num_cross_heads, num_self_heads=num_self_heads,
            n_freq=n_freq, max_freq=max_freq, val_dim=val_dim,
            num_latents=num_latents, dropout=dropout, coord_dim=coord_dim,
            in_chans=in_chans)
        if decoder_kind == "senseiver":
            self.decoder = SenseiverDecoder(
                emb_dim_in=emb_dim, dec_dim=emb_dim,
                n_freq=dec_n_freq, max_freq=dec_max_freq,
                num_heads=dec_num_heads, dropout=dropout,
                latent_size=1, coord_dim=coord_dim, out_chans=in_chans)
        elif decoder_kind == "cvit":
            self.decoder = CViTDecoder(
                emb_dim_in=emb_dim, dec_dim=emb_dim,
                n_freq=dec_n_freq, max_freq=dec_max_freq,
                num_heads=dec_num_heads, num_blocks=decoder_num_blocks,
                mlp_mult=decoder_mlp_mult, dropout=dropout,
                latent_size=1, coord_dim=coord_dim, out_chans=in_chans)
        else:
            raise ValueError(f"unknown decoder_kind={decoder_kind!r}")
        self.decoder_kind = decoder_kind
        self.emb_dim = emb_dim
        self.num_latents = num_latents
        self.coord_dim = coord_dim
        # Optional learned readout (0 = mean-pool, the default representation).
        self.readout = (QueryReadout(emb_dim, readout_queries)
                          if readout_queries > 0 else None)
        self.readout_queries = readout_queries

    def represent(self, tokens):
        """Pooled representation used downstream: mean-pool, or flattened
        learned readout when one is configured."""
        if self.readout is not None:
            return self.readout(tokens).flatten(1)      # (B, K*emb_dim)
        return tokens.mean(dim=1)                        # (B, emb_dim)

    def forward(self, u, in_coords, query_coords):
        tokens = self.encoder(u, in_coords)
        pred = self.decoder(tokens, query_coords)
        return pred, tokens


# Backward-compat aliases: original class names from the V3 implementation.
PerceiverEncoderV3 = FAEEncoder
DecoderV3 = SenseiverDecoder
DecoderCViT = CViTDecoder
PerceiverSparseAEV3 = FAE


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
