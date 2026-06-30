"""Generate L-DeepONet's spherical shallow-water data (Dedalus v3) for the FAE harness.

Adapted from external/latent-deeponet/code/data/generate-data-shallow-water.py — SAME physics
(Galewsky-type barotropic jet on a rotating sphere), with fixes/parameterization:
  - clears ./<snapdir> between samples (their version re-read a stale snapshots_s1.h5 — a bug),
  - env-parameterized: SW_N_SAMPLES, SW_STOP_HOURS, SW_NPHI/NTHETA, SW_OUT, SW_SNAPDIR,
  - saves params=(alpha,beta) [the inverse-probe target], inputs=initial perturbation, outputs=vorticity.

Run in the `dedalus` conda env (NOT wfae).  Smoke:  SW_N_SAMPLES=2 SW_STOP_HOURS=48 python scripts/gen_sw.py
"""
import os, shutil
import numpy as np
import dedalus.public as d3
import h5py
import logging
logger = logging.getLogger(__name__)

N_SAMPLES = int(os.environ.get("SW_N_SAMPLES", 150))
STOP_HOURS = float(os.environ.get("SW_STOP_HOURS", 360))
NPHI = int(os.environ.get("SW_NPHI", 256))
NTHETA = int(os.environ.get("SW_NTHETA", 256))
OUT = os.environ.get("SW_OUT", os.path.expanduser("~/scratch/sw_data/shallow_water.npz"))
SNAPDIR = os.environ.get("SW_SNAPDIR", "snapshots_sw")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# Simulation units / parameters (verbatim from L-DeepONet)
meter = 1 / 6.37122e6; hour = 1; second = hour / 3600
dealias = 3 / 2
R = 6.37122e6 * meter
Omega = 7.292e-5 / second
nu = 1e5 * meter**2 / second / 32**2
g = 9.80616 * meter / second**2
H = 1e4 * meter
timestep = 600 * second
stop_sim_time = STOP_HOURS * hour
dtype = np.float64

rng = np.random.default_rng(0)
alphas = rng.uniform(0.05, 0.6, N_SAMPLES)
betas = rng.uniform(0.02, 0.3, N_SAMPLES)
params = np.stack([alphas, betas], 1)                      # (N, 2) — the inverse-probe target (deterministic over full N)

START = int(os.environ.get("SW_START", 0))                 # array-chunking: this task does samples [START:END)
END = int(os.environ.get("SW_END", N_SAMPLES))             # each task own SW_OUT + SW_SNAPDIR -> no RAM leak across 150
init_perturbs, outputs = [], []
print(f"=== SW gen: samples [{START}:{END}) of {N_SAMPLES}, {NPHI}x{NTHETA} sphere, {STOP_HOURS}h -> {OUT} ===", flush=True)
for i in range(START, END):
    shutil.rmtree(SNAPDIR, ignore_errors=True)             # FIX: fresh snapshots each sample
    print(f"--- sample {i+1}/{N_SAMPLES}  alpha={alphas[i]:.3f} beta={betas[i]:.3f} ---", flush=True)

    coords = d3.S2Coordinates('phi', 'theta')
    dist = d3.Distributor(coords, dtype=dtype)
    basis = d3.SphereBasis(coords, (NPHI, NTHETA), radius=R, dealias=dealias, dtype=dtype)
    u = dist.VectorField(coords, name='u', bases=basis)
    h = dist.Field(name='h', bases=basis)
    zcross = lambda A: d3.MulCosine(d3.skew(A))

    phi, theta = dist.local_grids(basis)
    lat = np.pi / 2 - theta + 0 * phi
    umax = 80 * meter / second
    lat0 = np.pi / 7; lat1 = np.pi / 2 - lat0
    en = np.exp(-4 / (lat1 - lat0)**2)
    jet = (lat0 <= lat) * (lat <= lat1)
    u['g'][0][jet] = umax / en * np.exp(1 / (lat[jet] - lat0) / (lat[jet] - lat1))

    c = dist.Field(name='c')
    problem = d3.LBVP([h, c], namespace=locals())
    problem.add_equation("g*lap(h) + c = - div(u@grad(u) + 2*Omega*zcross(u))")
    problem.add_equation("ave(h) = 0")
    problem.build_solver().solve()

    lat2 = np.pi / 4; hpert = 120 * meter
    alpha, beta = alphas[i], betas[i]
    h['g'] += hpert * np.cos(lat) * np.exp(-(phi / alpha)**2) * np.exp(-((lat2 - lat) / beta)**2)
    init_perturbs.append(np.copy(h['g']))

    problem = d3.IVP([u, h], namespace=locals())
    problem.add_equation("dt(u) + nu*lap(lap(u)) + g*grad(h) + 2*Omega*zcross(u) = - u@grad(u)")
    problem.add_equation("dt(h) + nu*lap(lap(h)) + H*div(u) = - div(h*u)")
    solver = problem.build_solver(d3.RK222)
    solver.stop_sim_time = stop_sim_time

    snapshots = solver.evaluator.add_file_handler(SNAPDIR, sim_dt=1 * hour, max_writes=500)
    snapshots.add_task(h, name='height')
    snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')
    while solver.proceed:
        solver.step(timestep)

    snapbase = os.path.basename(SNAPDIR.rstrip('/'))
    with h5py.File(os.path.join(SNAPDIR, f"{snapbase}_s1.h5"), 'r') as hf:
        outputs.append(np.array(hf['tasks']['vorticity'])[::5])    # (~72, NPHI, NTHETA)

    np.savez(OUT, params=params[START:i + 1], inputs=np.array(init_perturbs), outputs=np.array(outputs))
    print(f"    saved sample {i} (chunk has {len(outputs)}, vorticity {outputs[-1].shape})", flush=True)

print(f"=== DONE: {len(outputs)} samples -> {OUT} ===", flush=True)
