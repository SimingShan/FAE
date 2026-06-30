"""Generate the 3D advection capstone dataset via APEBench/Exponax (on-the-fly, no download).
n_traj trajectories, each a RANDOM advection SPEED (the probe target) along a fixed diagonal direction,
random Fourier IC, T frames, N^3 grid, 1 channel. Materializes data/adv3d/ as (N_traj,T,1,N,N,N) +
speed labels + meta (vol=True). Train cheap @32^3; the FAE transfers to higher res at eval (resolution-invariant).
Run with the apebench venv:  ~/scratch/apebench_env/bin/python scripts/data/gen_adv3d.py
"""
import os, json, time
import numpy as np
import jax, jax.numpy as jnp
import exponax as ex
from numpy.lib.format import open_memmap

N, T = 32, 20
N_TRAJ, N_TEST = 5000, 500                                   # 5000*20 = 100k samples; 4500 train / 500 test traj
OUT = "/gpfs/radev/scratch/lu_lu/ss5235/WFAE/data/adv3d"
os.makedirs(OUT, exist_ok=True)

key = jax.random.PRNGKey(0)
ks = jax.random.split(key, N_TRAJ + 1)
traj_keys, speed_key = ks[:-1], ks[-1]
speeds = np.asarray(jax.random.uniform(speed_key, (N_TRAJ,), minval=0.5, maxval=2.0))   # the parameter
direction = jnp.array([0.8, 0.5, 0.33]); direction = direction / jnp.linalg.norm(direction)
ic_gen = ex.ic.RandomTruncatedFourierSeries(num_spatial_dims=3, cutoff=4)


@jax.jit
def gen(k, spd):
    ic = ic_gen(N, key=k)
    st = ex.stepper.Advection(num_spatial_dims=3, domain_extent=1.0, num_points=N, dt=0.06, velocity=spd * direction)
    return ex.rollout(st, T - 1, include_init=True)(ic)      # (T,1,N,N,N)


n_train = N_TRAJ - N_TEST
ssum = ssq = cnt = 0.0
for split, idxs in [("train", range(n_train)), ("test", range(n_train, N_TRAJ))]:
    idxs = list(idxs); Ns = len(idxs)
    mm = open_memmap(f"{OUT}/{split}_fields.npy", mode="w+", dtype=np.float32, shape=(Ns, T, 1, N, N, N))
    lab = np.zeros((Ns, 1), np.float32); t0 = time.time()
    for j, i in enumerate(idxs):
        trj = np.asarray(gen(traj_keys[i], speeds[i]))       # (T,1,N,N,N)
        mm[j] = trj; lab[j, 0] = speeds[i]
        if split == "train":
            ssum += trj.sum(); ssq += (trj ** 2).sum(); cnt += trj.size
        if (j + 1) % 500 == 0:
            mm.flush(); print(f"  [{split}] {j+1}/{Ns}  ({time.time()-t0:.0f}s)", flush=True)
    mm.flush(); np.save(f"{OUT}/{split}_labels.npy", lab)
    print(f"  [{split}] {Ns} traj  fields {mm.shape}", flush=True)

mean = ssum / cnt; std = float(np.sqrt(max(ssq / cnt - mean ** 2, 0)) + 1e-6)
json.dump(dict(dataset="adv3d", label_names=["speed"], stride=1, dt_max=4, vol=True,
               H=N, W=N, D=N, C=1, frames_per_traj=T, mean=[float(mean)], std=[std],
               held_out="trajectory (in-distribution; random speed)", n_train=n_train, n_test=N_TEST),
          open(f"{OUT}/meta.json", "w"), indent=2)
print(f"done. mean={mean:.4f} std={std:.4f}  -> data/adv3d/  ({n_train} train / {N_TEST} test traj, 100k samples)", flush=True)
