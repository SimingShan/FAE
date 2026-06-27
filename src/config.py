"""OmegaConf experiment config: configs/base.yaml (shared budget) merged with a per-method
method_data.yaml. Fairness BY CONSTRUCTION — a method config may only set its own architectural
knobs; overriding a shared-budget key raises. Param budget is asserted before any run, so we never
burn compute on a mismatched config.
"""
import os, subprocess
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# keys a method_data.yaml is FORBIDDEN to set (they come from base.yaml — keeps comparisons fair)
# keys a method_data.yaml is FORBIDDEN to set (must match for fairness). lr/weight_decay/betas are
# METHOD-LEVEL (each method uses its own appropriate recipe) so they are NOT here.
SHARED = ["dataset", "resolution", "in_chans", "n_traj", "n_seed", "frame_stride", "epochs", "batch",
          "seed", "amp", "emb_dim", "param_budget_M", "param_tol_M", "warmup_frac", "scheduler",
          "eval_probe", "eval_split", "eval_pool", "ridge_alphas", "eval_fae_full_grid",
          "eval_n_sensors", "eval_n_traj", "ckpt_dir"]
# NOTE: `mode` is FAE's objective knob (method-specific), NOT shared. The old REPA-gen / RAE shared
# keys were removed when those lines were archived (arxiv/repa_generation/).


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "nogit"


def load_config(path):
    # tree: configs/<dataset>/<method>/[<group>/]<setup>.yaml  +  configs/<dataset>/base.yaml (shared budget).
    # base.yaml is found by walking UP from the config -> any nesting depth (method/, method/group/...) works.
    p = path if os.path.isabs(path) else os.path.join(ROOT, path)
    d = os.path.dirname(p); base_path = None
    while d and d != os.path.dirname(d):
        cand = os.path.join(d, "base.yaml")
        if os.path.exists(cand) and os.path.abspath(cand) != os.path.abspath(p):
            base_path = cand; break
        d = os.path.dirname(d)
    if base_path is None:
        raise FileNotFoundError(f"no base.yaml found above {path}")
    base = OmegaConf.load(base_path)
    cfg = OmegaConf.load(p)
    bad = [k for k in cfg.keys() if k in SHARED]
    if bad:
        raise ValueError(f"{path} sets SHARED-budget key(s) {bad} — forbidden. Change them in the dataset base.yaml.")
    merged = OmegaConf.merge(base, cfg)
    merged.git = git_hash()
    if "method" in merged:                                   # encoder configs only; gen configs have no `method`
        merged.ckpt = f"{merged.ckpt_dir}/{merged.method}_{merged.dataset}{merged.resolution}_s{merged.seed}.pt"
    return merged


def ckpt_path(cfg, seed):
    """Organized checkpoint path mirroring the config tree: <ckpt_dir>/<dataset>/<method>/<tag>_s<seed>.pt."""
    return os.path.join(cfg.ckpt_dir, cfg.dataset, cfg.method, f"{cfg.tag}_s{seed}.pt")


def ckpt_file(method, tag, seed, dataset="ns", ckpt_dir="results/checkpoints"):
    """Resolve a checkpoint by (method, tag, seed) without a full cfg. ABSOLUTE path under ROOT.
    Single source of truth for eval/viz scripts -> renames happen in ONE place."""
    return os.path.join(ROOT, ckpt_dir, dataset, method, f"{tag}_s{seed}.pt")


def encoder_params_M(cfg):
    """Build the ENCODER (what produces the representation) and return its param count in millions."""
    import torch
    if cfg.method == "fae":
        from src.models.fae import FAE
        m = FAE(emb_dim=cfg.emb_dim, num_latents=cfg.num_latents, num_iter=cfg.num_iter,
                depth_per_iter=cfg.depth_per_iter, in_chans=cfg.in_chans, coord_dim=2)
        n = sum(p.numel() for p in m.encoder.parameters())
    else:
        from scripts.train_baseline import build_model
        meth = "mae" if cfg.method == "mae" else "ijepa"
        m = build_model(meth, resolution=cfg.resolution, in_chans=cfg.in_chans,
                        embed_dim=cfg.emb_dim, depth=cfg.depth, patch_size=cfg.patch_size)
        if cfg.method == "mae":
            n = sum(p.numel() for nm, p in m.named_parameters() if not nm.startswith("decoder"))
        else:
            n = sum(p.numel() for p in m.encoder.parameters())
    return n / 1e6


def assert_budget(cfg):
    """Verify encoder params are within tolerance of the shared budget. Raises -> never run a mismatch."""
    got = encoder_params_M(cfg)
    if abs(got - cfg.param_budget_M) > cfg.param_tol_M:
        raise AssertionError(f"[{cfg.method}] encoder {got:.2f}M outside budget "
                             f"{cfg.param_budget_M}±{cfg.param_tol_M}M — fix the config before submitting.")
    return got
