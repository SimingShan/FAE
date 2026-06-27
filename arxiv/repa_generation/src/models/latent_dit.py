"""Token-sequence flow-matching DiT for RAE stage-2. Operates directly on latent tokens (B,N,D) —
works for FAE's 128-token SET *and* MAE/JEPA grid tokens (no patchify, no positional grid assumption;
the encoder's tokens already carry position for the ViTs, and FAE's set is order-free). AdaLN-Zero
time conditioning, velocity output (linear-path flow matching, same recipe as generate.py).
"""
import math
import torch, torch.nn as nn


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimeEmbed(nn.Module):
    def __init__(self, dim, freq=256):
        super().__init__(); self.freq = freq
        self.mlp = nn.Sequential(nn.Linear(freq, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        h = self.freq // 2
        f = torch.exp(-math.log(10000) * torch.arange(h, device=t.device, dtype=t.dtype) / h)
        a = t[:, None] * f[None]
        return self.mlp(torch.cat([torch.cos(a), torch.sin(a)], -1))


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, mlp=4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp)), nn.GELU(), nn.Linear(int(dim * mlp), dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight); nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, c):
        sa, ba, ga, sm, bm, gm = self.ada(c).chunk(6, -1)
        h = modulate(self.n1(x), sa, ba); a, _ = self.attn(h, h, h)
        x = x + ga.unsqueeze(1) * a
        x = x + gm.unsqueeze(1) * self.mlp(modulate(self.n2(x), sm, bm))
        return x


class LatentDiT(nn.Module):
    """(B,N,D) latent + t -> velocity (B,N,D)."""
    def __init__(self, dim, depth=8, heads=8):
        super().__init__()
        self.tin = nn.Linear(dim, dim)
        self.temb = TimeEmbed(dim)
        self.blocks = nn.ModuleList([DiTBlock(dim, heads) for _ in range(depth)])
        self.nf = nn.LayerNorm(dim, elementwise_affine=False)
        self.adaf = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.head = nn.Linear(dim, dim)
        nn.init.zeros_(self.adaf[-1].weight); nn.init.zeros_(self.adaf[-1].bias)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x, t):
        c = self.temb(t); h = self.tin(x)
        for b in self.blocks:
            h = b(h, c)
        s, sc = self.adaf(c).chunk(2, -1)
        return self.head(modulate(self.nf(h), s, sc))


@torch.no_grad()
def sample_latent(model, n, N, D, device, steps=50):
    """Euler ODE t:1->0 in latent space -> (n,N,D)."""
    x = torch.randn(n, N, D, device=device)
    for i in range(steps):
        t = torch.full((n,), 1 - i / steps, device=device)
        x = x - (1 / steps) * model(x, t)
    return x
