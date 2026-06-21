"""Unified entry point for the NS / SSLForPDEs experiment suite.

  python scripts/run_ns.py download [--subset N]                  # fetch + verify the dataset
  python scripts/run_ns.py train-fae      --seed 0 [--grad]       # OUR method (FAE), saves a checkpoint
  python scripts/run_ns.py train-vicreg   --seed 0                # THEIR method (collapses on this data; see docs)
  python scripts/run_ns.py probe          --seed 0                # buoyancy linear probe: ours + theirs + floor
  python scripts/run_ns.py rollout --cond {time,buoyancy,rep} --seed 0   # time-stepping (Table 2)

Every stage threads --seed for reproducibility. See docs/NS_EXPERIMENTS.md for the full protocol,
expected numbers, and the data-scale caveat (the released dataset is ~10x smaller than the paper's).
"""
import argparse, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def sh(*cmd):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    sub.add_parser("download").add_argument("--subset", type=int, default=0)
    p = sub.add_parser("train-fae"); p.add_argument("--seed", type=int, default=0); p.add_argument("--grad", action="store_true"); p.add_argument("--epochs", type=int, default=100)
    p = sub.add_parser("train-vicreg"); p.add_argument("--seed", type=int, default=0); p.add_argument("--epochs", type=int, default=100)
    p = sub.add_parser("probe"); p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("rollout"); p.add_argument("--cond", choices=["time", "buoyancy", "rep"], required=True); p.add_argument("--seed", type=int, default=0); p.add_argument("--epochs", type=int, default=20); p.add_argument("--n_traj_per_file", type=int, default=4)
    a = ap.parse_args()

    if a.stage == "download":
        sh(PY, "scripts/download_ns.py", *(["--subset", a.subset] if a.subset else []))
    elif a.stage == "train-fae":
        cmd = [PY, "scripts/train_fae_predict.py", "--dataset", "ns", "--resolution", 128, "--mode", "twoview",
               "--dt_max", 2, "--dt_fixed", 2, "--frame_stride", 4, "--num_iter", 4, "--mcnt", 512, 1024,
               "--n_query", 2048, "--epochs", a.epochs, "--seed", a.seed, "--save",
               "--tag", f"fae_ns_grad_s{a.seed}" if a.grad else f"fae_ns_s{a.seed}"]
        if a.grad: cmd += ["--lam_grad", 0.5]
        sh(*cmd)
    elif a.stage == "train-vicreg":
        sh(PY, "external/SSLForPDEs/navier_stokes.py", "--data-root", os.environ.get("NS_DATA_ROOT", os.path.expanduser("~/scratch/ns_data")),
           "--logging-folder", "results/ssl_logs", "--exp-name", f"vicreg_s{a.seed}", "--epochs", a.epochs)
    elif a.stage == "probe":
        sh(PY, "scripts/eval_ns_probe.py")                 # ours + floor (valid->test ridge)
        sh(PY, "scripts/eval_ns_theirs.py")                # their VICReg backbone, same ridge probe
    elif a.stage == "rollout":
        sh(PY, "scripts/rollout_ns.py", "--mode", a.cond, "--n_traj_per_file", a.n_traj_per_file,
           "--epochs", a.epochs, "--seed", a.seed)


if __name__ == "__main__":
    main()
