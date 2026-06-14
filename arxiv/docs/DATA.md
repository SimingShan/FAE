# DATA — the G1 benchmark dataset

One testbed: **G1**, four 1D periodic PDE systems with per-trajectory
coefficients. Generated entirely by `data_gen/gen_g1_all.py` (GPU spectral
solvers, continuous coefficient sampling, GRF initial conditions). The
PDEBench-sourced version used before 2026-06-02 is retired (raw files in
`../WFAE_attic/data/1d/`).

## Systems

| system | equation (x ∈ [0,1), periodic) | coefficient | notes |
|---|---|---|---|
| `heat` | u_t = ν u_xx | ν ~ logU[1e-3, 1e-1] | exact spectral propagator |
| `advection` | u_t + β u_x = 0 | β ~ U[0.1, 4] | exact spectral shift; β is **unidentifiable from a single snapshot** (established negative result) |
| `burgers` | u_t + u u_x = ν u_xx | ν ~ logU[1e-4, 1e-2] | shock-forming; dealiased pseudo-spectral |
| `reaction_diffusion` | u_t = D u_xx + r u(1−u), r = 1 | D ~ logU[1e-4, 1e-2] | Fisher-KPP; called **"AC"** in tables and discussion |

## Format

`data/1d/{system}/{system}_g1.h5`:

```
u        (5000, 100, 1024)  float32 — trajectories × frames × grid
coeff    (5000,)            float32 — per-trajectory coefficient
attrs    pde, pde_class (0=heat, 1=advection, 2=burgers, 3=reaction_diffusion),
         coeff_name (nu / beta / nu / D), dt
```

Loader: `src/data/g1.py`
- `G1FrameDataset(time_subsample=10)` — per-snapshot training corpus across
  all 4 systems (~200k frames at subsample 10), items are
  `(frame (1024,), pde_class, coeff)`.
- `load_g1_system(name)` — raw per-system access for evaluation.
- `train_val_split(u, coeff)` — **mandatory** shuffle-then-split.

## Conventions

- Native grid 1024; sparse-encoder inputs are index subsets of that grid;
  coordinates are `idx / 1024 ∈ [0, 1)`.
- Per-snapshot encoding at the trajectory mid-frame for probes/evaluation.
- Standard evaluation sensor budget N=256 (uniform stride) unless the
  diagnostic varies N on purpose.
- Regenerating: `python data_gen/gen_g1_all.py` (optionally `--system`,
  `--gpu`); ~5 min/system on one GPU. Visual sanity: `data_gen/visualize_1d.py`.
