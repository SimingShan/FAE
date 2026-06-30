"""2D Fourier Neural Operator (Li et al.) — the dense full-grid forecasting ORACLE/ceiling.
Operates directly on the full field (no autoencoder, no sparsity): field_t -> field_{t+dt}, dt-conditioned
via an extra constant channel. The best-case dense surrogate to bracket the latent operators against."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, m1, m2):
        super().__init__()
        self.in_ch, self.out_ch, self.m1, self.m2 = in_ch, out_ch, m1, m2
        s = 1.0 / (in_ch * out_ch)
        self.w1 = nn.Parameter(s * torch.rand(in_ch, out_ch, m1, m2, 2))    # complex via last dim
        self.w2 = nn.Parameter(s * torch.rand(in_ch, out_ch, m1, m2, 2))

    def _mul(self, x, w):                                                   # x (B,in,m1,m2) complex
        return torch.einsum("bixy,ioxy->boxy", x, torch.view_as_complex(w))

    def forward(self, x):                                                   # x (B,C,H,W) real
        B, _, H, W = x.shape
        xft = torch.fft.rfft2(x)
        out = torch.zeros(B, self.out_ch, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        out[:, :, :self.m1, :self.m2] = self._mul(xft[:, :, :self.m1, :self.m2], self.w1)
        out[:, :, -self.m1:, :self.m2] = self._mul(xft[:, :, -self.m1:, :self.m2], self.w2)
        return torch.fft.irfft2(out, s=(H, W))


class FNO2d(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, width=32, modes=12, n_layers=4, dt_cond=True):
        super().__init__()
        self.dt_cond = dt_cond
        self.lift = nn.Conv2d(in_ch + (1 if dt_cond else 0), width, 1)
        self.spectral = nn.ModuleList([SpectralConv2d(width, width, modes, modes) for _ in range(n_layers)])
        self.pw = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.proj = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, out_ch, 1))

    def forward(self, field, dt):                                          # field (B,C,H,W), dt (B,)
        x = field
        if self.dt_cond:
            dtc = dt[:, None, None, None].expand(-1, 1, field.size(-2), field.size(-1))
            x = torch.cat([field, dtc], 1)
        x = self.lift(x)
        for sp, pw in zip(self.spectral, self.pw):
            x = F.gelu(sp(x) + pw(x))
        return self.proj(x)
