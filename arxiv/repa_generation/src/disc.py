"""PatchGAN discriminator + hinge losses + a physics spectral loss for RAE stage-1 decoder training.
GAN gives high-frequency sharpness (fluids have shocks/fronts/filaments that MSE over-smooths);
the spectral/gradient term is the physics-native sharpener (penalizes high-k mismatch, no hallucination).
LPIPS is dropped — it's a VGG-on-RGB perceptual net, not meaningful for multi-channel physics fields.
"""
import torch, torch.nn as nn, torch.nn.functional as F


class PatchDisc(nn.Module):
    """Multi-channel PatchGAN (Isola et al.), GroupNorm for stability on physics fields."""
    def __init__(self, in_chans, ch=64, n_layers=3):
        super().__init__()
        layers = [nn.Conv2d(in_chans, ch, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        c = ch
        for _ in range(1, n_layers):
            layers += [nn.Conv2d(c, c * 2, 4, 2, 1), nn.GroupNorm(8, c * 2), nn.LeakyReLU(0.2, True)]
            c *= 2
        layers += [nn.Conv2d(c, 1, 4, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def hinge_d(real, fake):
    return F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()


def hinge_g(fake):
    return -fake.mean()


def spectral_loss(xhat, x):
    """Gradient (finite diff) + FFT-magnitude mismatch -> penalize lost high-k energy.
    Computed in fp32 (torch.fft does not support bf16 under autocast)."""
    xhat = xhat.float(); x = x.float()
    def grad(z):
        return z[..., 1:, :] - z[..., :-1, :], z[..., :, 1:] - z[..., :, :-1]
    gxh, gyh = grad(xhat); gx, gy = grad(x)
    g = F.l1_loss(gxh, gx) + F.l1_loss(gyh, gy)
    fh = torch.fft.rfft2(xhat).abs(); ff = torch.fft.rfft2(x).abs()
    return g + F.l1_loss(fh, ff)


def adaptive_gan_weight(recon_loss, gan_loss, last_layer, eps=1e-6):
    """RAE/VQGAN adaptive weight: balance the GAN grad magnitude to the reconstruction grad."""
    rg = torch.autograd.grad(recon_loss, last_layer, retain_graph=True)[0]
    gg = torch.autograd.grad(gan_loss, last_layer, retain_graph=True)[0]
    return (rg.norm() / (gg.norm() + eps)).clamp(0, 1e4).detach()
