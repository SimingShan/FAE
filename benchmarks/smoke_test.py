"""Smoke test for the benchmark baselines — run: python benchmarks/smoke_test.py

Verifies MAE and AE instantiate, forward/backward, mask correctly, patchify
round-trips exactly, and produce a frozen-probe representation — on 4-channel
128x128 physics fields at ~7M-param parity.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from benchmarks.mae.mae import mae_physics, ae_physics


def main():
    x = torch.randn(2, 4, 128, 128)
    ok = True

    print("=== MAE (mask_ratio=0.75) ===")
    m = mae_physics()
    npar = sum(p.numel() for p in m.parameters()) / 1e6
    loss, pred, mask = m(x, mask_ratio=0.75)
    p = m.patch_embed.patch_size[0]
    exp_pred = (2, m.patch_embed.num_patches, p * p * m.in_chans)
    print(f"  params={npar:.2f}M  patches={m.patch_embed.num_patches}  "
          f"loss={loss.item():.4f}  pred={tuple(pred.shape)} (expect {exp_pred})  "
          f"masked_frac={mask.mean().item():.2f}")
    ok &= 6.0 < npar < 8.0                          # ~7M parity
    ok &= tuple(pred.shape) == exp_pred
    ok &= abs(mask.mean().item() - 0.75) < 0.05
    rt = (m.unpatchify(m.patchify(x)) - x).abs().max().item()
    print(f"  patchify->unpatchify round-trip err={rt:.2e} (expect 0)")
    ok &= rt < 1e-5
    z = m.encode(x)
    print(f"  encode -> {tuple(z.shape)} (frozen probe rep)")
    ok &= z.shape == (2, 256)
    loss.backward()
    print("  backward OK")

    print("=== AE (mask_ratio=0.0, full reconstruction) ===")
    a = ae_physics()
    lossA, predA, maskA = a(x, mask_ratio=0.0)
    print(f"  params={sum(p.numel() for p in a.parameters())/1e6:.2f}M  "
          f"loss={lossA.item():.4f}  masked_frac={maskA.mean().item():.2f} (expect 0.00)")
    ok &= maskA.sum().item() == 0
    lossA.backward()
    print("  backward OK")

    print("\n" + ("ALL SMOKE TESTS PASSED" if ok else "SMOKE TEST FAILURES — check above"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
