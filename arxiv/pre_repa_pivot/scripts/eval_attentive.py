"""Attentive-probe eval (frozen encoder), single seed, all models + RANDOM-INIT floor.
A learnable query cross-attends over the frozen tokens -> small head -> (logRe, logSc).
Because the probe is trainable, the ONLY meaningful signal is trained-encoder vs
random-init-encoder of the SAME architecture (the random floor is the honest baseline).
Reports R2 on standardized labels for Reynolds and Schmidt."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from src.models import FAE
from src.data.well2d import (ShearFlowSnapshotDataset, ShearFlowWindowDataset,
                             ShearFlowClipDataset, make_coords_2d, fields_to_tokens)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CK = "results/checkpoints/g1/"


class AttnProbe(nn.Module):
    """Additive attention pooling: per-token score -> softmax -> weighted mean -> MLP head.
    Simpler/more stable than multihead cross-attention on small probe sets."""
    def __init__(self, d_in, d=64, n_out=2, p=0.4):
        super().__init__()
        self.drop = nn.Dropout(p)
        self.proj = nn.Linear(d_in, d)
        self.score = nn.Linear(d, 1)
        self.head = nn.Linear(d, n_out)                     # linear head (low capacity)

    def forward(self, tok):
        x = self.proj(self.drop(tok))                       # (B,N,d)
        w = torch.softmax(self.score(x).squeeze(-1), dim=1) # (B,N)
        return self.head(self.drop((w.unsqueeze(-1) * x).sum(1)))


def labels(ds):
    lr = np.asarray(ds.logRe, dtype=np.float32); sc = np.asarray(ds.Sc, dtype=np.float32)
    lsc = np.log10(sc) if sc.max() > 2 else sc
    return np.stack([lr, lsc], 1)


@torch.no_grad()
def extract(model, get_tok, ds, batch):
    Z = []
    for x, _ in DataLoader(ds, batch_size=batch):
        Z.append(get_tok(model, x.to(DEVICE)).float().cpu())
    return torch.cat(Z)


def run_probe(Ztr, Ytr, Zva, Yva, epochs=300, seed=0):
    torch.manual_seed(seed)
    m, s = Ytr.mean(0), Ytr.std(0) + 1e-8
    yt = torch.tensor((Ytr - m) / s).float().to(DEVICE)
    yv = (Yva - m) / s
    Ztr = Ztr.to(DEVICE); Zva = Zva.to(DEVICE); bs = 256
    n = len(Ztr); pr = torch.randperm(n); nval = max(8, n // 5)
    vi, ti = pr[:nval].to(DEVICE), pr[nval:].to(DEVICE)        # probe-val / probe-train (early stop)
    probe = AttnProbe(Ztr.size(-1), d=128).to(DEVICE)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-3)
    best, best_sd, wait = 1e9, None, 0
    for ep in range(epochs):
        probe.train(); p2 = ti[torch.randperm(len(ti), device=DEVICE)]
        for i in range(0, len(ti), bs):
            idx = p2[i:i + bs]
            loss = ((probe(Ztr[idx]) - yt[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            vl = ((probe(Ztr[vi]) - yt[vi]) ** 2).mean().item()
        if vl < best - 1e-4: best, best_sd, wait = vl, {k: v.clone() for k, v in probe.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 25: break
    probe.load_state_dict(best_sd); probe.eval()
    with torch.no_grad():
        pv = probe(Zva).cpu().numpy()
    return r2_score(yv[:, 0], pv[:, 0]), r2_score(yv[:, 1], pv[:, 1])


def build(method, ckpt, R=224):
    cko = torch.load(CK + ckpt, map_location="cpu", weights_only=False)
    st = cko["stats"]
    if method == "fae":
        m = FAE(emb_dim=320, num_iter=4, depth_per_iter=4, num_latents=128, num_cross_heads=4,
                num_self_heads=8, n_freq=32, max_freq=32, coord_dim=2, in_chans=4)
        tr = ShearFlowClipDataset("train", n_seed=6, frame_stride=4, clip_len=3, side=R, stats=st)
        va = ShearFlowClipDataset("valid", n_seed=8, frame_stride=4, clip_len=3, side=R, stats=st)
        coords = make_coords_2d(R, DEVICE); g = torch.Generator().manual_seed(0)
        iA = torch.randperm(R * R, generator=g)[:512].to(DEVICE)
        def gt(model, x): return model.encode_tokens(fields_to_tokens(x[:, :, 0], iA), coords[iA])
        return m, gt, tr, va
    nf = cko["train_args"].get("n_frames", 1)
    if method in ("mae", "ijepa"):
        tr = ShearFlowSnapshotDataset("train", n_seed=6, frame_stride=12, side=R, stats=st)
        va = ShearFlowSnapshotDataset("valid", n_seed=8, frame_stride=12, side=R, stats=st)
    else:
        tr = ShearFlowWindowDataset("train", n_seed=6, n_frames=nf, side=R, stats=st)
        va = ShearFlowWindowDataset("valid", n_seed=8, n_frames=nf, side=R, stats=st)
    import scripts.train_baseline as tb; tb.DEVICE = DEVICE
    m = tb.build_model(method, R, nf, cko["train_args"].get("tubelet", 2), 4)
    GT = {"mae": lambda mo, x: mo.forward_encoder(x, 0.0)[0][:, 1:, :],
          "videomae": lambda mo, x: mo.forward_encoder(x, 0.0)[0][:, 1:, :],
          "ijepa": lambda mo, x: mo.target(x),
          "stjepa": lambda mo, x: mo.target(x)}[method]
    return m, GT, tr, va


def main():
    jobs = [("fae", "faep_twoview_tvb_s0.pt"), ("mae", "mae_shear_mae_s1.pt"),
            ("ijepa", "ijepa_shear_ijepaB_s0.pt"), ("videomae", "videomae_shear_vmae16_09_s0.pt"),
            ("stjepa", "stjepa_shear_stjB16_s0.pt")]
    print(f"{'model':12} {'Re(trained)':12} {'Re(random)':12} {'Sc(trained)':12} {'Sc(random)':12}")
    for method, ckpt in jobs:
        m, gt, tr, va = build(method, ckpt)
        bs = 32 if method in ("videomae", "stjepa") else 128
        Ytr, Yva = labels(tr), labels(va)
        # random-init floor (same arch, fresh weights)
        m_rand = m.to(DEVICE).eval()
        Zr_tr = extract(m_rand, gt, tr, bs); Zr_va = extract(m_rand, gt, va, bs)
        r_re, r_sc = run_probe(Zr_tr, Ytr, Zr_va, Yva)
        # trained
        sd = torch.load(CK + ckpt, map_location="cpu", weights_only=False)["model"]
        m.load_state_dict(sd); m = m.to(DEVICE).eval()
        Zt_tr = extract(m, gt, tr, bs); Zt_va = extract(m, gt, va, bs)
        t_re, t_sc = run_probe(Zt_tr, Ytr, Zt_va, Yva)
        print(f"{method:12} {t_re:<12.3f} {r_re:<12.3f} {t_sc:<12.3f} {r_sc:<12.3f}", flush=True)
    print("=> trained MUST beat random (its honest floor). Re margin = real structure; Sc margin = ?")


if __name__ == "__main__":
    main()
