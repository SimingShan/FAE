"""Functional JEPA — temporal latent-prediction SSL on a sparse coordinate-set encoder.

The cleanest dynamics + invariance objective, no reconstruction:
  encode a sparse view of the field at time t          -> La  (online, grad)
  encode a DIFFERENT sparse view at t+h (stop-grad/EMA) -> Lb  (target, no grad)
  a token-set predictor maps La -> L̂b ; match L̂b to Lb.
Predicting the *future* latent forces dynamics; the differing sparsity forces
observation-invariance. The predictor (attention over the M latent tokens) is
**discardable after training** — only the encoder is kept.

Anti-collapse is load-bearing here (no recon anchor):
  raw    SimSiam — predictor + stop-grad, cosine match.
  ema    BYOL    — EMA target encoder + predictor, cosine match.
  vicreg variance/covariance on the predicted tokens (handled in the trainer).
"""
import copy
import math
import torch
import torch.nn as nn

from src.models.fae import FAEEncoder, SelfLayer


class TokenPredictor(nn.Module):
    """Δt-conditioned set->set predictor over the M latent tokens — a learned
    latent FLOW L_t -> L_{t+Δ}. Discardable from the representation eval, but it
    IS an approximate evolution operator, so it's kept on disk. Conditioning on
    Δ (a continuous gap) is what makes the model functional in *time* too, and
    what starves the trivial collapse (one fixed gap is too easy)."""
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
        x = x + self._dt_embed(dt).unsqueeze(1)                # condition every token on Δ
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))


class MLPPredictor(nn.Module):
    """Δt-conditioned PER-TOKEN MLP predictor — NO cross-token attention. A
    deliberately weaker latent flow: it can't mix tokens, so it can't 'solve' the
    prediction by itself, forcing the ENCODER to carry the dynamics. The control
    for 'does the attention predictor over-train and do the encoder's job?'."""
    def __init__(self, dim, depth=2, dt_freq=16, mult=4):
        super().__init__()
        self.dt_freq = dt_freq
        self.dt_mlp = nn.Sequential(nn.Linear(2 * dt_freq, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * mult), nn.GELU(),
                          nn.Linear(dim * mult, dim)) for _ in range(depth)])
        self.head = nn.Linear(dim, dim)

    def _dt_embed(self, dt):
        f = torch.arange(1, self.dt_freq + 1, device=dt.device, dtype=dt.dtype)
        a = dt[:, None] * f[None, :] * math.pi
        return self.dt_mlp(torch.cat([torch.sin(a), torch.cos(a)], dim=-1))

    def forward(self, x, dt):
        x = x + self._dt_embed(dt).unsqueeze(1)
        for blk in self.blocks:
            x = x + blk(x)                                     # residual, per-token only
        return self.head(x)


class FunctionalJEPA(nn.Module):
    def __init__(self, emb_dim=320, num_iter=4, depth_per_iter=4, num_latents=128,
                 num_cross_heads=4, num_self_heads=8, n_freq=32, max_freq=32,
                 coord_dim=2, in_chans=4, pred_depth=2, pred_type="attn",
                 use_ema=False, ema_decay=0.996):
        super().__init__()
        self.encoder = FAEEncoder(
            emb_dim=emb_dim, num_iter=num_iter, depth_per_iter=depth_per_iter,
            num_cross_heads=num_cross_heads, num_self_heads=num_self_heads,
            n_freq=n_freq, max_freq=max_freq, num_latents=num_latents,
            coord_dim=coord_dim, in_chans=in_chans)
        self.predictor = (TokenPredictor(emb_dim, depth=pred_depth, heads=num_self_heads)
                          if pred_type == "attn" else MLPPredictor(emb_dim, depth=pred_depth))
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.emb_dim = emb_dim
        if use_ema:
            self.target = copy.deepcopy(self.encoder)
            for p in self.target.parameters():
                p.requires_grad_(False)

    def encode(self, u, coords):                 # online (B, M, D)
        return self.encoder(u, coords)

    @torch.no_grad()
    def encode_target(self, u, coords):          # target: EMA copy, or sg-shared encoder
        enc = self.target if self.use_ema else self.encoder
        return enc(u, coords)

    def predict(self, tokens, dt):               # (B, M, D), Δ in [0,1] -> predicted target
        return self.predictor(tokens, dt)

    def represent(self, tokens):                 # pooled rep for probe/PR
        return tokens.mean(dim=1)

    @torch.no_grad()
    def update_ema(self):
        if not self.use_ema:
            return
        for pe, pt in zip(self.encoder.parameters(), self.target.parameters()):
            pt.mul_(self.ema_decay).add_(pe.data, alpha=1.0 - self.ema_decay)
