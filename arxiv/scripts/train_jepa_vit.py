"""Train the JEPA-ViT baseline: faithful I-JEPA port to 1D PDE snapshots.

Recipe (matches Meta I-JEPA `src/train.py`):
  - target encoder = EMA copy; sees the FULL field, targets are extracted
    from its OUTPUT at target-patch positions and layer-normed.
  - context encoder sees only the context patches.
  - smooth_l1_loss(predictor(context), target); per-iteration EMA ramp
    0.996 -> 1.0. No VICReg, no reconstruction.

Model classes live in src/models/jepa_vit.py. The representation used by the
benchmark is the TARGET branch, mean-pooled over patch tokens.
"""
import os, sys, time, argparse, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.g1 import G1FrameDataset
from src.models.jepa_vit import (
    VisionTransformer1D, VisionTransformerPredictor1D, apply_masks, sample_masks,
)


def train(out_path, epochs=15, batch=64, lr=1e-3, gpu=0, workers=4,
          embed_dim=384, depth=8, num_heads=6,
          pred_dim=192, pred_depth=6, pred_heads=6,
          patch_size=16, n_ctx_patches=40, n_tgt_patches=12,
          ema_start=0.996, ema_end=1.0,
          warmup_iters=500, time_subsample=5):
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"=== I-JEPA-1D (ViT)  device={device} ===", flush=True)
    print(f"  patches: {1024//patch_size},  N_ctx={n_ctx_patches},  N_tgt={n_tgt_patches}", flush=True)
    print(f"  encoder: dim={embed_dim} depth={depth} heads={num_heads}", flush=True)
    print(f"  predictor: dim={pred_dim} depth={pred_depth} heads={pred_heads}", flush=True)
    print(f"  EMA τ: {ema_start} → {ema_end} (per-iter)", flush=True)

    ds = G1FrameDataset(time_subsample=time_subsample)
    print(f"  snapshots: {len(ds):,}", flush=True)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers,
                         pin_memory=True, drop_last=True,
                         persistent_workers=(workers > 0))
    ipe = len(loader)

    num_patches = 1024 // patch_size
    encoder = VisionTransformer1D(img_size=1024, patch_size=patch_size,
                                     embed_dim=embed_dim, depth=depth,
                                     num_heads=num_heads).to(device)
    predictor = VisionTransformerPredictor1D(num_patches=num_patches,
                                                embed_dim=embed_dim,
                                                predictor_embed_dim=pred_dim,
                                                depth=pred_depth,
                                                num_heads=pred_heads).to(device)
    target_encoder = VisionTransformer1D(img_size=1024, patch_size=patch_size,
                                              embed_dim=embed_dim, depth=depth,
                                              num_heads=num_heads).to(device)
    target_encoder.load_state_dict(encoder.state_dict())
    for p_ in target_encoder.parameters(): p_.requires_grad_(False)

    n_par_e = sum(p.numel() for p in encoder.parameters())
    n_par_p = sum(p.numel() for p in predictor.parameters())
    print(f"  encoder params: {n_par_e/1e6:.3f}M", flush=True)
    print(f"  predictor params: {n_par_p/1e6:.3f}M", flush=True)

    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=lr, weight_decay=1e-4)
    total_iters = epochs * ipe

    def lr_at(it):
        if it < warmup_iters:
            return lr * (it / max(1, warmup_iters))
        prog = (it - warmup_iters) / max(1, total_iters - warmup_iters)
        return lr * 0.5 * (1.0 + math.cos(math.pi * prog))

    history = []
    t0_train = time.time()
    it_global = 0
    for ep in range(epochs):
        ls = 0.0; n_b = 0
        for u, _cls, _coeff in loader:
            u = u.to(device, non_blocking=True).float()             # (B, X)
            B = u.size(0)
            ctx_mask, tgt_mask = sample_masks(B, num_patches,
                                                n_ctx_patches, n_tgt_patches, device)

            with torch.no_grad():
                h = target_encoder(u, masks=None)                   # (B, P, D)
                h = apply_masks(h, tgt_mask)                        # (B, N_tgt, D)
                h = F.layer_norm(h, (h.size(-1),))                  # target LN

            z = encoder(u, masks=ctx_mask)                           # (B, N_ctx, D)
            z_pred = predictor(z, ctx_mask, tgt_mask)               # (B, N_tgt, D)

            loss = F.smooth_l1_loss(z_pred, h)

            for g in opt.param_groups: g["lr"] = lr_at(it_global)
            opt.zero_grad(); loss.backward(); opt.step()

            frac = it_global / max(total_iters - 1, 1)
            tau = ema_start + (ema_end - ema_start) * frac
            with torch.no_grad():
                for p_o, p_t in zip(encoder.parameters(), target_encoder.parameters()):
                    p_t.mul_(tau).add_(p_o.data, alpha=1.0 - tau)

            ls += loss.item(); n_b += 1; it_global += 1
        ls /= n_b
        elapsed = int(time.time() - t0_train)
        print(f"ep {ep+1:>3d}/{epochs}  loss={ls:.4e}  τ_end={tau:.4f}  "
              f"lr_end={lr_at(it_global-1):.2e}  ({elapsed}s)", flush=True)
        history.append({"epoch": ep + 1, "loss": ls, "tau": tau, "elapsed": elapsed})

    save = {
        "method": "jepa_vit",
        "history": history,
        "n_par_encoder": n_par_e, "n_par_predictor": n_par_p,
        "encoder":  encoder.state_dict(),
        "target":   target_encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "config": {"img_size": 1024, "patch_size": patch_size,
                     "embed_dim": embed_dim, "depth": depth, "num_heads": num_heads,
                     "pred_dim": pred_dim, "pred_depth": pred_depth,
                     "pred_heads": pred_heads,
                     "n_ctx_patches": n_ctx_patches, "n_tgt_patches": n_tgt_patches,
                     "ema_start": ema_start, "ema_end": ema_end,
                     "loss": "smooth_l1", "target_layernorm": True,
                     "time_subsample": time_subsample},
    }
    torch.save(save, out_path)
    print(f"\ndone in {int(time.time() - t0_train)}s  →  {out_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",   required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch",  type=int, default=64)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--gpu",    type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--embed_dim", type=int, default=384)
    ap.add_argument("--depth",     type=int, default=8)
    ap.add_argument("--num_heads", type=int, default=6)
    ap.add_argument("--pred_dim",  type=int, default=192)
    ap.add_argument("--pred_depth", type=int, default=6)
    ap.add_argument("--pred_heads", type=int, default=6)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--n_ctx_patches", type=int, default=40)
    ap.add_argument("--n_tgt_patches", type=int, default=12)
    ap.add_argument("--ema_start", type=float, default=0.996)
    ap.add_argument("--ema_end",   type=float, default=1.0)
    ap.add_argument("--warmup_iters", type=int, default=500)
    ap.add_argument("--time_subsample", type=int, default=5)
    args = ap.parse_args()
    train(args.out, args.epochs, args.batch, args.lr, args.gpu, args.workers,
          args.embed_dim, args.depth, args.num_heads,
          args.pred_dim, args.pred_depth, args.pred_heads,
          args.patch_size, args.n_ctx_patches, args.n_tgt_patches,
          args.ema_start, args.ema_end, args.warmup_iters,
          args.time_subsample)
