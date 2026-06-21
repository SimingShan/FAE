"""GPU sanity for the 8-baseline matrix: build + forward + backward + encode at 224,
print trainable params. Exits 0 only if every config passes."""
import sys, types, traceback, torch
sys.path.insert(0, ".")
import scripts.train_baseline as tb
from benchmarks.jepa import stjepa  # import check

A = types.SimpleNamespace(mask_ratio=0.9, ctx_frac=0.2, tgt_frac=0.06)
IMG = 224
CONFIGS = [("mae", 1), ("ijepa", 1), ("videomae", 4), ("videomae", 8),
           ("videomae", 16), ("stjepa", 4), ("stjepa", 8), ("stjepa", 16)]
ok = True
print(f"device={tb.DEVICE}  img={IMG}")
for method, nf in CONFIGS:
    try:
        m = tb.build_model(method, IMG, nf)
        p = sum(x.numel() for x in m.parameters() if x.requires_grad) / 1e6
        x = (torch.randn(2, 4, nf, IMG, IMG) if method in ("videomae", "stjepa")
             else torch.randn(2, 4, IMG, IMG)).to(tb.DEVICE)
        mr = 0.9 if method == "videomae" else A.mask_ratio
        loss = tb.loss_step(method, m, x, A)
        loss.backward()
        gnorm = sum(q.grad.norm().item() for q in m.parameters() if q.grad is not None)
        z = m.encode(x)
        print(f"  PASS {method:9s} nf={nf:2d}  params={p:5.2f}M  loss={float(loss):.3f} "
              f"gradnorm={gnorm:.1f} encode={tuple(z.shape)}", flush=True)
        del m, x, loss
        if tb.DEVICE == "cuda": torch.cuda.empty_cache()
    except Exception as e:
        ok = False
        print(f"  FAIL {method:9s} nf={nf:2d}: {e}", flush=True)
        traceback.print_exc()
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
