"""RAE = frozen pretrained encoder + trained coordinate decoder (Diffusion Transformers with
Representation Autoencoders, adapted to PDE fields). The encoder's latent IS the stage-2 diffusion
space — unlike REPA, which only *aligns* to it. A UNIFORM coordinate decoder (FAEDecoder) is trained
for ALL encoders, so the only variable is the frozen latent (set for FAE, grid for MAE/JEPA).

  encode(field) -> latent tokens (B,N,D)   [frozen]
  decode(latent, coords) -> field          [trained]
"""
import torch, torch.nn as nn
from src.config import ckpt_file
from src.models.fae import FAE, FAEDecoder
from src.data.well2d import make_coords_2d, fields_to_tokens


def load_encoder(method, tag, seed, in_chans, side, device):
    """Frozen pretrained encoder -> a callable field(B,C,H,W) -> latent tokens (B,N,D), and its width D."""
    ck = torch.load(ckpt_file(method, tag, seed), map_location=device); a = ck["train_args"]
    if method == "fae":
        m = FAE(emb_dim=a["emb_dim"], num_iter=a.get("num_iter", 4), num_latents=a["num_latents"],
                in_chans=in_chans, coord_dim=2).to(device)
        m.load_state_dict(ck["model"]); m.eval()
        coords = make_coords_2d(side, device); idx = torch.arange(side * side, device=device)   # full grid
        enc = lambda x: m.encode_tokens(fields_to_tokens(x, idx), coords[idx])
        D = a["emb_dim"]
    else:
        from scripts.train_baseline import build_model
        m = build_model("mae" if method == "mae" else "ijepa", resolution=side, in_chans=in_chans,
                        embed_dim=a["embed_dim"], depth=a["depth"], patch_size=a["patch_size"]).to(device)
        m.load_state_dict(ck["model"]); m.eval()
        enc = (lambda x: m.forward_encoder(x, 0.0)[0][:, 1:]) if method == "mae" else (lambda x: m.target(x))
        D = a["embed_dim"]
    for p in m.parameters():
        p.requires_grad_(False)
    return m, enc, D


class RAE(nn.Module):
    """Frozen encoder (set/grid latent) + trained coordinate decoder + latent normalization stats."""
    def __init__(self, method, tag, seed, in_chans, side, device):
        super().__init__()
        self.enc_model, self.encode_fn, self.D = load_encoder(method, tag, seed, in_chans, side, device)
        self.decoder = FAEDecoder(emb_dim_in=self.D, dec_dim=self.D, out_chans=in_chans, coord_dim=2).to(device)
        self.side, self.in_chans = side, in_chans
        self.register_buffer("coords", make_coords_2d(side, device))         # (NPIX, 2)
        self.register_buffer("lat_mean", torch.zeros(1, 1, self.D, device=device))
        self.register_buffer("lat_std", torch.ones(1, 1, self.D, device=device))

    @torch.no_grad()
    def encode(self, x):
        return self.encode_fn(x)                                              # (B,N,D), frozen

    def normalize(self, z):
        return (z - self.lat_mean) / self.lat_std

    def denormalize(self, z):
        return z * self.lat_std + self.lat_mean

    def decode(self, z, coords=None):
        """latent (B,N,D) -> field (B,C,H,W). coords None = full grid."""
        c = self.coords if coords is None else coords
        pred = self.decoder(z, c)                                             # (B, Nq, C)
        if coords is None:
            B = z.size(0)
            return pred.reshape(B, self.side, self.side, self.in_chans).permute(0, 3, 1, 2)
        return pred                                                          # raw (B,Nq,C) for sparse-coord recon

    @torch.no_grad()
    def set_latent_stats(self, fields, n=256, chunk=32):
        """Per-dim mean/std of the latent over up to n fields (B,C,H,W), chunked -> normalization buffers."""
        Z = torch.cat([self.encode(fields[i:i + chunk]).reshape(-1, self.D)
                       for i in range(0, min(n, len(fields)), chunk)])
        self.lat_mean.copy_(Z.mean(0).view(1, 1, -1))
        self.lat_std.copy_(Z.std(0).clamp_min(1e-4).view(1, 1, -1))
