"""Model zoo — single place that knows how to load and encode every method.

All benchmark scripts (evaluate, diag_*) load checkpoints through this module
instead of re-implementing per-method loaders and encode functions.

Method registry (G1 benchmark, all ~7M params):

  name               label            kind     representation
  ----------------------------------------------------------------------
  fae_recon          FAE-recon        fae      tokens.mean(1), sparse input
  fae_vicreg         FAE+VICReg       fae      tokens.mean(1), sparse input
  fae_spatiotemporal FAE-T2           fae      tokens.mean(1), sparse input
  mlp                MLPSparseAE      mlp      set-pooled latent, sparse input
  cnn                CNN1DAE          cnn      pooled bottleneck, dense input
  mae                MAE1DAE          mae      pooled patch tokens, dense input
  jepa_perceiver     JEPA-Perceiver   fae      target-branch tokens.mean(1)
  jepa_vit           JEPA-ViT         vit      target-branch patch-mean

Checkpoints live in ``results/checkpoints/g1/<name>.pt``. Each stores at least
{"model": state_dict} (FAE variants also store {"config": arch kwargs};
JEPA ones store {"encoder"/"target": state_dict, "config": ...}).

Historical note: methods were renamed in the 2026-06 cleanup
(v3_recon -> fae_recon, v3_vicreg -> fae_vicreg, v3_spatiotemporal ->
fae_spatiotemporal, jepa_perceiver_sparse -> jepa_perceiver,
jepa_vit1d -> jepa_vit). Old result JSONs keep the old keys.
"""
from __future__ import annotations
import os
from dataclasses import dataclass

import torch

from .fae import FAE
from .baselines import MLPSparseAE, CNN1DAE, MAE1DAE
from .jepa_vit import VisionTransformer1D

X_FULL = 1024
DEFAULT_CKPT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "results", "checkpoints", "g1"))


@dataclass(frozen=True)
class MethodSpec:
    name: str
    label: str
    kind: str          # one of {"fae", "mlp", "cnn", "mae", "vit"}
    branch: str        # which state_dict inside the checkpoint
    sparse_input: bool # accepts an arbitrary sensor subset
    has_decoder: bool  # can reconstruct the field


METHODS = [
    MethodSpec("fae_recon",          "FAE-recon",      "fae", "model",  True,  True),
    MethodSpec("fae_vicreg",         "FAE+VICReg",     "fae", "model",  True,  True),
    MethodSpec("fae_spatiotemporal", "FAE-T2",         "fae", "model",  True,  True),
    MethodSpec("mlp",                "MLPSparseAE",    "mlp", "model",  True,  True),
    MethodSpec("cnn",                "CNN1DAE",        "cnn", "model",  False, True),
    MethodSpec("mae",                "MAE1DAE",        "mae", "model",  True,  True),
    MethodSpec("jepa_perceiver",     "JEPA-Perceiver", "fae", "target", True,  False),
    MethodSpec("jepa_vit",           "JEPA-ViT",       "vit", "target", False, False),
    # Ablation: VICReg trained with the sensing floor raised to 256 — tests the
    # invariance-capacity trade-off hypothesis (dimension cap set by the
    # sparsest training views). Auto-skipped while the checkpoint is absent.
    MethodSpec("fae_vicreg_floor256", "FAE+VICReg-f256", "fae", "model", True, True),
    # Ablation: VICReg alignment weight lowered 25 -> 5 (capacity-cap suspect #2:
    # the invariance objective strength itself).
    MethodSpec("fae_vicreg_sim5",     "FAE+VICReg-s5",   "fae", "model", True, True),
    # Capacity-cap fix: learned 8-query readout trained under VICReg (tests
    # whether the mean-pool readout, not the encoder, caps usable dimension).
    MethodSpec("fae_vicreg_q8",       "FAE+VICReg-q8",   "fae", "model", True, True),
]
METHODS_BY_NAME = {m.name: m for m in METHODS}


def load_method(name: str, ckpt_dir: str = DEFAULT_CKPT_DIR, device: str = "cpu"):
    """Load one benchmark method. Returns (model.eval() with grads off, spec),
    or (None, spec) if the checkpoint file does not exist."""
    spec = METHODS_BY_NAME[name]
    path = os.path.join(ckpt_dir, f"{name}.pt")
    if not os.path.exists(path):
        return None, spec
    ck = torch.load(path, map_location=device, weights_only=False)
    if spec.kind == "fae":
        m = FAE(**ck["config"]).to(device).eval()
        m.load_state_dict(ck.get(spec.branch, ck["model"]))
    elif spec.kind == "mlp":
        m = MLPSparseAE(coord_dim=1, latent_dim=320, enc_emb=640, dec_emb=640).to(device).eval()
        m.load_state_dict(ck["model"])
    elif spec.kind == "cnn":
        m = CNN1DAE().to(device).eval()
        m.load_state_dict(ck["model"])
    elif spec.kind == "mae":
        m = MAE1DAE().to(device).eval()
        m.load_state_dict(ck["model"])
    elif spec.kind == "vit":
        cfg = ck["config"]
        m = VisionTransformer1D(img_size=cfg["img_size"], patch_size=cfg["patch_size"],
                                     embed_dim=cfg["embed_dim"], depth=cfg["depth"],
                                     num_heads=cfg["num_heads"]).to(device).eval()
        m.load_state_dict(ck.get(spec.branch, ck["encoder"]))
    else:
        raise ValueError(f"unknown kind {spec.kind!r}")
    for p in m.parameters():
        p.requires_grad_(False)
    return m, spec


@torch.no_grad()
def encode(model, kind: str, u_field, full_coords, idx=None, n_sensors: int = 256):
    """Pooled latent for a batch of full fields.

    u_field:     (B, X) full-resolution snapshots
    full_coords: (X, 1) coordinate grid on the same device
    idx:         optional sensor index set; default = uniform stride subset of
                 size n_sensors (sparse methods) / full grid (dense methods)
    returns:     (B, D) pooled latent
    """
    X = u_field.size(1)
    B = u_field.size(0)
    if kind in ("fae", "mlp"):
        if idx is None:
            idx = torch.arange(0, X, X // n_sensors, device=u_field.device)[:n_sensors]
        coords_in = full_coords[idx]
        if kind == "fae":
            tok = model.encoder(u_field[:, idx].unsqueeze(-1), coords_in)
            # honor a learned readout if the model has one (else mean-pool)
            if getattr(model, "readout", None) is not None:
                return model.represent(tok)
            return tok.mean(dim=1)
        return model.encoder(u_field[:, idx].unsqueeze(-1),
                                coords_in.unsqueeze(0).expand(B, -1, -1))
    if kind == "cnn":
        feats = model.encoder_conv(u_field.unsqueeze(1))
        return model.latent_proj(feats.mean(dim=-1))
    if kind == "mae":
        if idx is not None:
            u_pad = torch.zeros_like(u_field)
            u_pad[:, idx] = u_field[:, idx]
            return model.encode_full(u_pad.unsqueeze(1))
        return model.encode_full(u_field.unsqueeze(1))
    if kind == "vit":
        return model(u_field, masks=None).mean(dim=1)
    raise ValueError(f"unknown kind {kind!r}")


@torch.no_grad()
def decode_sparse(model, kind: str, u_field, full_coords, n_sensors: int):
    """Encode at a uniform-stride sparse sensor set, decode at the full grid.

    Returns (B, X) prediction, or None for methods without a decoder.
    Dense methods (cnn, mae) receive the sparse field zero-filled to the grid.
    """
    X = u_field.size(1)
    B = u_field.size(0)
    if kind in ("fae", "mlp"):
        idx = torch.arange(0, X, X // n_sensors, device=u_field.device)[:n_sensors]
        coords_in = full_coords[idx]
        if kind == "fae":
            tokens = model.encoder(u_field[:, idx].unsqueeze(-1), coords_in)
            return model.decoder(tokens, full_coords).squeeze(-1)
        z = model.encoder(u_field[:, idx].unsqueeze(-1),
                            coords_in.unsqueeze(0).expand(B, -1, -1))
        pred = model.decoder(z, full_coords.unsqueeze(0).expand(B, -1, -1))
        return pred.squeeze(-1) if pred.dim() == 3 else pred
    if kind in ("cnn", "mae"):
        idx = torch.arange(0, X, X // n_sensors, device=u_field.device)[:n_sensors]
        u_pad = torch.zeros_like(u_field)
        u_pad[:, idx] = u_field[:, idx]
        pred, _ = model(u_pad.unsqueeze(1))
        return pred.squeeze(1)
    return None
