"""Grid convolutional autoencoder — the L-DeepONet encoder, the architecture-axis counterpart to FAE.
Conv downsample a field to a FLAT latent vector, deconv back. Pretrained (recon loss) then FROZEN, so
the forecasting comparison is a single-variable swap: FAE set-latent + SetOperator  vs  this flat latent
+ FlatOperator. Exposes encode(x)->(B,latent) and decode(z)->(B,C,H,W) to mirror the FAE interface."""
import torch
import torch.nn as nn


class GridCAE(nn.Module):
    def __init__(self, in_ch=3, side=64, latent=512, ch=32):
        super().__init__()
        self.side, self.latent, self.in_ch, self.ch = side, latent, in_ch, ch
        self.enc = nn.Sequential(                                  # 64 -> 4 (four stride-2 convs)
            nn.Conv2d(in_ch, ch, 4, 2, 1), nn.GELU(),             # 32
            nn.Conv2d(ch, 2 * ch, 4, 2, 1), nn.GELU(),            # 16
            nn.Conv2d(2 * ch, 4 * ch, 4, 2, 1), nn.GELU(),        # 8
            nn.Conv2d(4 * ch, 8 * ch, 4, 2, 1), nn.GELU())        # 4
        self.f = side // 16
        flat = 8 * ch * self.f * self.f
        self.to_z = nn.Linear(flat, latent)
        self.from_z = nn.Linear(latent, flat)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(8 * ch, 4 * ch, 4, 2, 1), nn.GELU(),   # 8
            nn.ConvTranspose2d(4 * ch, 2 * ch, 4, 2, 1), nn.GELU(),   # 16
            nn.ConvTranspose2d(2 * ch, ch, 4, 2, 1), nn.GELU(),       # 32
            nn.ConvTranspose2d(ch, in_ch, 4, 2, 1))                   # 64

    def encode(self, x):                                          # (B,C,H,W) -> (B, latent)
        return self.to_z(self.enc(x).flatten(1))

    def decode(self, z):                                          # (B, latent) -> (B,C,H,W)
        h = self.from_z(z).view(-1, 8 * self.ch, self.f, self.f)
        return self.dec(h)

    def forward(self, x):
        return self.decode(self.encode(x))
