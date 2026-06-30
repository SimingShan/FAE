"""FID for uncond REPA runs — treats PDE fields as 3-channel images (user's framing: our setup IS
image generation). torchvision InceptionV3 2048-d pool features + Frechet distance. Absolute values
won't match papers (canonical FID uses a TF-ported Inception); the RANKING across configs is the point.
Paired with spectrum_dist. Same per-channel transform applied to real and generated.
"""
import os, sys, glob, re, collections, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external/REPA"))
from scipy import linalg
import torchvision
from scripts.eval_uncond_gen import build_from_args
from scripts.generate import sample, get_frames, DEVICE

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def inception_model():
    m = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1,
                                        transform_input=False)
    m.fc = nn.Identity(); m.eval().to(DEVICE)                         # -> 2048-d pool features
    for p in m.parameters(): p.requires_grad_(False)
    return m


@torch.no_grad()
def feats(model, fields, bs=128):
    """fields: (N,3,H,W) z-scored. clamp->[0,1]->299->ImageNet-norm (identical for real & gen)."""
    mean, std = _MEAN.to(DEVICE), _STD.to(DEVICE); out = []
    for i in range(0, len(fields), bs):
        x = fields[i:i + bs].to(DEVICE).clamp(-3, 3).add(3).div(6)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        out.append(model((x - mean) / std).cpu().numpy())
    return np.concatenate(out)


def fid(f1, f2):
    mu1, mu2 = f1.mean(0), f2.mean(0)
    c1, c2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    covmean = linalg.sqrtm(c1 @ c2, disp=False)[0]
    if np.iscomplexobj(covmean): covmean = covmean.real
    return float((mu1 - mu2) @ (mu1 - mu2) + np.trace(c1 + c2 - 2 * covmean))


@torch.no_grad()
def gen_samples(path, n=2048, bs=256):
    ck = torch.load(path, map_location=DEVICE); a = ck["args"]; ncls = ck.get("ncls", 1)
    m, C, R = build_from_args(a, ncls); m.load_state_dict(ck["ema"]); m.eval()
    torch.manual_seed(12345)
    label = a["align"] if a["align"] == "none" else f"{a['align']}@lam{a.get('lam')}"
    return torch.cat([sample(m, min(bs, n - i), C, R) for i in range(0, n, bs)]), label


def main():
    R = 64; inc = inception_model()
    tr = get_frames("ns", "train", R); nreal = min(len(tr), 4096)
    real = torch.from_numpy(np.stack([tr[i][0].numpy() for i in range(nreal)]))
    fr = feats(inc, real)
    print(f"=== FID (InceptionV3 2048-d, {nreal} real vs 2048 gen/ckpt) ===", flush=True)
    by = collections.defaultdict(list)
    for path in sorted(glob.glob("results/checkpoints/g1/gen_ns_uncond_*_s*.pt")):
        try:
            g, al = gen_samples(path); v = fid(fr, feats(inc, g))
            seed = (re.search(r"_s(\d+)\.pt$", path) or [None, "?"])[1]
            by[al].append((seed, v)); print(f"  {os.path.basename(path):34s} align={al:6s} FID={v:.2f}", flush=True)
        except Exception as e:
            print(f"  {os.path.basename(path):34s} FAIL {str(e)[:55]}", flush=True)
    print(f"\n  {'align':8s} {'mean FID':>9s} {'std':>7s} {'n':>3s}   per-seed")
    print("  " + "-" * 58)
    agg = [(al, float(np.mean([x for _, x in v])), float(np.std([x for _, x in v])), len(v),
            " ".join(f"s{s}={x:.1f}" for s, x in sorted(v))) for al, v in by.items()]
    for al, m, s, n, ps in sorted(agg, key=lambda r: r[1]):
        print(f"  {al:8s} {m:>9.2f} {s:>7.2f} {n:>3d}   {ps}", flush=True)


if __name__ == "__main__":
    main()
