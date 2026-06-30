"""Load frozen encoders from checkpoints for evaluation (probe / sensor sweeps / figures).
One place that rebuilds the exact architecture from a checkpoint's train_args.
"""
import os
import torch
from src.models.fae import FAE
from benchmarks import build_model

DS_CHANS = {"typhoon": 1, "ns": 3, "flowbench": 3, "shear": 4, "sw": 1, "mhd": 7, "rbc": 4}    # FAE checkpoints don't store in_chans


@torch.no_grad()
def mae_ordered_tokens(enc, x):
    """MAE patch tokens in SPATIAL order (bypass forward_encoder's random_masking, which shuffles even at ratio 0).
    Needed so the temporal operator can align tokens t -> t+dt (FAE latents are already fixed-order learned queries)."""
    z = enc.patch_embed(x) + enc.pos_embed[:, 1:, :]
    for blk in enc.blocks:
        z = blk(z)
    return enc.norm(z)


def fae_hw(ckpt, base_side):
    """(H, W) the FAE was trained at (rectangular-aware). `ckpt` = path or a train_args dict."""
    a = ckpt if isinstance(ckpt, dict) else torch.load(ckpt, map_location="cpu")["train_args"]
    return (a["res_h"], a["res_w"]) if a.get("res_h") else (base_side, base_side)


@torch.no_grad()
def load_fae(ckpt, device, base_side=128):
    """-> (model, (H, W)). Rebuilds the exact arch from the checkpoint (no hardcoded heads/freqs)."""
    a = torch.load(ckpt, map_location=device)["train_args"]
    inc = a.get("in_chans") or DS_CHANS.get(a.get("dataset"), 4)
    m = FAE(emb_dim=a["emb_dim"], num_iter=a.get("num_iter", 4), depth_per_iter=a.get("depth_per_iter", 5),
            num_latents=a["num_latents"], num_cross_heads=a.get("num_cross_heads", 4), num_self_heads=a.get("num_self_heads", 8),
            n_freq=a.get("n_freq", 32), max_freq=a.get("max_freq", 32), val_dim=a.get("val_dim", 32), coord_dim=2, in_chans=inc,
            use_local=a.get("use_local", False), local_k=a.get("local_k", 8), local_dim=a.get("local_dim", 48)).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device)["model"]); m.eval()
    return m, fae_hw(a, base_side)


@torch.no_grad()
def load_vit(ckpt, device):
    """-> (model, method). Native-aspect aware (shear MAE/JEPA are 128x256)."""
    a = torch.load(ckpt, map_location=device)["train_args"]; method = a["method"]
    import json as _json                                            # train_baseline builds res/in_chans from META (not from --args) -> read meta to rebuild
    mp = f"data/{a.get('dataset', '')}/meta.json"
    _m = _json.load(open(mp)) if os.path.exists(mp) else {}
    if a.get("res_h"):
        res = (a["res_h"], a["res_w"])
    elif _m:
        res = (_m["H"], _m["W"]) if _m["H"] != _m["W"] else _m["H"]
    else:
        res = a["resolution"]
    inc = a.get("in_chans") or _m.get("C") or DS_CHANS.get(a.get("dataset"), 1)
    m = build_model("mae" if method == "mae" else "ijepa", resolution=res, in_chans=inc,
                    embed_dim=a.get("embed_dim"), depth=a.get("depth"), patch_size=a.get("patch_size")).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device)["model"]); m.eval()
    return m, method
