"""DeepONet operators (faithful to L-DeepONet's DON.py) for the forecasting comparison.

  DeepONetOperator : latent -> latent. Branch CNN over the sqrt(d) x sqrt(d) latent 'image' + trunk FNN
                     over the time gap, combined  Sum_p (branch . trunk).  (their p=4, sin activations)
                     This is the L-DeepONet arm's operator (paired with the frozen grid-CAE latent).
  PixelDeepONet    : field -> field, NO autoencoder. Branch CNN over the input field -> per-channel
                     coefficients; trunk over (query coord, dt) -> basis; Sum_p (branch . trunk) -> value.
                     The pixel-space operator baseline (what L-DeepONet's latent version improves over).
"""
import math
import torch
import torch.nn as nn


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class DeepONetOperator(nn.Module):
    def __init__(self, latent, p=4):
        super().__init__()
        self.s = int(round(math.sqrt(latent)))
        assert self.s * self.s == latent, f"latent {latent} must be a perfect square for the branch CNN"
        self.latent, self.p = latent, p
        self.branch = nn.Sequential(                              # (1, s, s) -> (latent*p)
            nn.Conv2d(1, 32, 3), Sin(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, 3), Sin(), nn.BatchNorm2d(16),
            nn.Flatten(), nn.LazyLinear(latent * p))
        self.trunk = nn.Sequential(                              # dt -> (latent*p)
            nn.Linear(1, 100), Sin(), nn.Linear(100, 100), Sin(), nn.Linear(100, latent * p))

    def forward(self, z, dt):                                    # z (B, latent), dt (B,)
        b = self.branch(z.view(-1, 1, self.s, self.s)).view(-1, self.latent, self.p)
        t = self.trunk(dt[:, None]).view(-1, self.latent, self.p)
        return (b * t).sum(-1)                                   # (B, latent)


class PixelDeepONet(nn.Module):
    def __init__(self, in_ch=3, side=64, p=128, n_freq=16):
        super().__init__()
        self.in_ch, self.side, self.p, self.n_freq = in_ch, side, p, n_freq
        self.branch = nn.Sequential(                             # field -> (in_ch * p)
            nn.Conv2d(in_ch, 32, 4, 2, 1), nn.GELU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(),
            nn.Flatten(), nn.LazyLinear(in_ch * p))
        self.trunk = nn.Sequential(                             # (coord fourier + dt) -> p
            nn.Linear(2 * 2 * n_freq + 1, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(), nn.Linear(256, p))

    def _fourier(self, coords):                                 # (N,2) -> (N, 2*2*n_freq)
        f = (2 ** torch.arange(self.n_freq, device=coords.device).float()) * math.pi
        a = coords[..., None] * f
        return torch.cat([torch.sin(a), torch.cos(a)], -1).flatten(-2)

    def forward(self, field, coords, dt):                       # field (B,C,H,W), coords (N,2), dt (B,) -> (B,N,C)
        B, N = field.size(0), coords.size(0)
        b = self.branch(field).view(B, self.in_ch, self.p)
        cf = self._fourier(coords)[None].expand(B, -1, -1)
        tin = torch.cat([cf, dt[:, None, None].expand(B, N, 1)], -1)
        t = self.trunk(tin)                                     # (B,N,p)
        return torch.einsum('bcp,bnp->bnc', b, t)               # (B,N,C)
