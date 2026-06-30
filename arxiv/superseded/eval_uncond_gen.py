"""Many-sample (CLAUDE.md rule #3) uncond REPA spectrum_dist eval. Loads every gen_ns_uncond_*_s0.pt,
rebuilds the SiT exactly as generate.py did, samples 2048 fields, and ranks by radial-spectrum distance
to real. The 256-sample in-train number is too noisy for ranking; this is the trustworthy read."""
import os, sys, glob, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external/REPA"))
from models.sit import SiT_models
from scripts.generate import sample, spectrum_dist, get_frames, DEVICE


def build_from_args(a, ncls):
    R = a["resolution"]; C = 3 if a["dataset"] == "ns" else 4
    sz = a["size"].split("-")[1].split("/")[0]
    extra = {"decoder_hidden_size": 384} if sz == "S" else {}
    if a["align"] in ("none", "fae", "fjepa"):
        zdim = 320                                                   # must match generate.py's zdim logic
    else:                                                            # mae/jepa: the matched encoder's width
        zdim = torch.load(a["enc_ckpt"], map_location="cpu").get("train_args", {}).get("embed_dim") or 256
    in_ch = 2 * C if a["mode"] in ("sparse", "all") else C
    m = SiT_models[a["size"]](input_size=R, in_channels=in_ch, num_classes=ncls, class_dropout_prob=0.1,
                              z_dims=[zdim], encoder_depth=a["depth"], fused_attn=False, qk_norm=False, **extra).to(DEVICE)
    return m, C, R


@torch.no_grad()
def eval_ckpt(path, real, n=2048, bs=256):
    ck = torch.load(path, map_location=DEVICE); a = ck["args"]; ncls = ck.get("ncls", 1)
    m, C, R = build_from_args(a, ncls); m.load_state_dict(ck["ema"]); m.eval()
    torch.manual_seed(12345)                                         # common random numbers: identical
    gs = [sample(m, min(bs, n - i), C, R) for i in range(0, n, bs)]  # noise draws across all checkpoints
    label = a["align"] if a["align"] == "none" else f"{a['align']}@lam{a.get('lam')}"
    return spectrum_dist(torch.cat(gs), real), label


def main():
    import re, collections
    R = 64
    tr = get_frames("ns", "train", R)
    real = torch.from_numpy(np.stack([tr[i][0].numpy() for i in range(min(len(tr), 512))])).to(DEVICE)
    print("=== uncond REPA spectrum_dist (2048 samples/seed, aggregated over seeds) ===", flush=True)
    by_align = collections.defaultdict(list)
    for path in sorted(glob.glob("results/checkpoints/g1/gen_ns_uncond_*_s*.pt")):
        try:
            sd, al = eval_ckpt(path, real)
            seed = (re.search(r"_s(\d+)\.pt$", path) or [None, "?"])[1]
            by_align[al].append((seed, sd))
            print(f"  {os.path.basename(path):36s} align={al:6s} sd={sd:.4f}", flush=True)
        except Exception as e:
            print(f"  {os.path.basename(path):36s} FAIL {str(e)[:50]}", flush=True)
    print(f"\n  {'align':8s} {'mean sd':>9s} {'std':>7s} {'n':>3s}   per-seed")
    print("  " + "-" * 60)
    agg = [(al, float(np.mean([s for _, s in v])), float(np.std([s for _, s in v])), len(v),
            " ".join(f"s{sd}={x:.3f}" for sd, x in sorted(v))) for al, v in by_align.items()]
    from src.utils.reslog import log_result
    for al, m, sdv, n, perseed in sorted(agg, key=lambda r: r[1]):
        print(f"  {al:8s} {m:>9.4f} {sdv:>7.4f} {n:>3d}   {perseed}", flush=True)
        lam = al.split("@lam")[1] if "@lam" in al else ""
        log_result(category="gen_uncond", name=al, dataset="ns", res=64, lam=lam, seeds=n,
                   n_samples=2048, metric="spectrum_dist", value=round(m, 4), std=round(sdv, 4))


if __name__ == "__main__":
    main()
