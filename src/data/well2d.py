"""The Well `shear_flow` 2D datasets + coordinate / token helpers.

Each split (train/valid/test) holds 28 HDF5 files, one per (Reynolds, Schmidt)
combination:

    $THE_WELL_DATA_DIR/shear_flow/data/{split}/shear_flow_Reynolds_*_Schmidt_*.hdf5

Per file: 32 trajectories x 200 timesteps x 256 x 512, with fields
``t0_fields/{tracer,pressure}`` (scalars) and ``t1_fields/velocity`` (2-vector),
assembled into 4 channels ``[tracer, pressure, velocity_x, velocity_y]``.
Re in {1e4, 5e4, 1e5, 5e5} and Sc in {0.1, 0.2, 0.5, 1, 2, 5, 10}. Following Qu
et al. (compression ["log", None]) the probe labels are ``log10(Re)`` and
``raw Sc`` (Reynolds is log-compressed, Schmidt is NOT).

Exports
-------
``ShearFlowSnapshotDataset``  single-frame samples, field ``(4, side, side)``  -> coord_dim 2
``ShearFlowWindowDataset``    ``n_frames`` windows, field ``(4, n_frames, side, side)`` -> coord_dim 3
``make_coords_2d`` / ``make_coords_3d``  normalized-[0,1] coordinate grids in flat
                                         (row-major, frame-outer) order
``fields_to_tokens(fields, idx)``  gather pixel/voxel values at flat indices ``idx``
                                   -> ``(B, N, C)`` sensor tokens

The flat ordering of ``make_coords_*`` matches ``fields.flatten(2)`` exactly, so a
coordinate at flat index ``k`` always corresponds to the value selected by
``fields_to_tokens(..., idx=k)`` — coords and tokens stay aligned by construction.
"""
from __future__ import annotations
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

CHANNELS = ("tracer", "pressure", "velocity_x", "velocity_y")
_DEFAULT_ROOT = "/gpfs/radev/home/ss5235/scratch/the_well_data"


# ----------------------------------------------------------------------
# Coordinate grids + tokenization
# ----------------------------------------------------------------------
def make_coords_2d(n_side: int = 224, device="cpu") -> torch.Tensor:
    """(n_side*n_side, 2) grid in [0, 1], flat index k = i*n_side + j -> (i, j)."""
    g = torch.linspace(0.0, 1.0, n_side, device=device)
    gi, gj = torch.meshgrid(g, g, indexing="ij")
    return torch.stack([gi.flatten(), gj.flatten()], dim=-1)


def make_coords_2d_hw(H: int, W: int, device="cpu") -> torch.Tensor:
    """(H*W, 2) grid in [0,1]^2 for NON-SQUARE fields; flat index k = i*W + j -> (i, j)."""
    gi = torch.linspace(0.0, 1.0, H, device=device)
    gj = torch.linspace(0.0, 1.0, W, device=device)
    ii, jj = torch.meshgrid(gi, gj, indexing="ij")
    return torch.stack([ii.flatten(), jj.flatten()], dim=-1)


def make_coords_3d(n_frames: int, n_side: int = 224, device="cpu") -> torch.Tensor:
    """(n_frames*n_side*n_side, 3) grid in [0, 1], flat index = t*HW + i*n_side + j."""
    t = torch.linspace(0.0, 1.0, n_frames, device=device)
    g = torch.linspace(0.0, 1.0, n_side, device=device)
    tt, gi, gj = torch.meshgrid(t, g, g, indexing="ij")
    return torch.stack([tt.flatten(), gi.flatten(), gj.flatten()], dim=-1)


def fields_to_tokens(fields: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather values at flat indices ``idx`` -> sensor tokens.

    fields: (B, C, *spatial)   idx: (N,) long on the same device
    returns: (B, N, C)
    """
    B, C = fields.shape[:2]
    flat = fields.reshape(B, C, -1)                  # (B, C, P)
    sel = flat.index_select(2, idx)                  # (B, C, N)
    return sel.permute(0, 2, 1).contiguous()         # (B, N, C)


def fields_to_tube_tokens(fields: torch.Tensor, spatial_idx: torch.Tensor) -> torch.Tensor:
    """Gather values at SPATIAL indices for EVERY frame (tube sampling).

    fields: (B, C, T, H, W)   spatial_idx: (N,) long into H*W
    returns: (B, N, T, C)  — for FAE time_mode='aggregate' (per-frame encode).
    """
    B, C, T = fields.shape[:3]
    flat = fields.reshape(B, C, T, -1)               # (B, C, T, H*W)
    sel = flat.index_select(3, spatial_idx)          # (B, C, T, N)
    return sel.permute(0, 3, 2, 1).contiguous()      # (B, N, T, C)


# ----------------------------------------------------------------------
# Shared file IO
# ----------------------------------------------------------------------
def _data_root() -> str:
    return os.environ.get("THE_WELL_DATA_DIR", _DEFAULT_ROOT)


def _split_files(split: str):
    d = os.path.join(_data_root(), "shear_flow", "data", split)
    files = sorted(glob.glob(os.path.join(d, "*.hdf5")) +
                   glob.glob(os.path.join(d, "*.h5")))
    if not files:
        raise FileNotFoundError(
            f"no shear_flow files in {d} — set THE_WELL_DATA_DIR (see docs/MIGRATION.md)")
    return files


def _resize(x: np.ndarray, side) -> np.ndarray:
    """(M, C, H, W) float32 -> (M, C, h, w). ``side`` is an int (square) OR an (h, w) tuple.

    shear_flow is 256x512 (1:2 aspect); a square ``side`` squishes the y axis (fair square grid for
    every method); an (h, w) tuple PRESERVES the native aspect (FAE-only, coordinate-native).
    """
    hw = (side, side) if isinstance(side, int) else tuple(side)
    if x.shape[-2:] == hw:
        return x
    t = torch.from_numpy(x)
    t = F.interpolate(t, size=hw, mode="bilinear", align_corners=False)
    return t.numpy()


def _read_snapshots(path: str, n_seed: int, frame_stride: int, side: int):
    """One file -> (M, 4, side, side) float32, with M = n_traj * n_frames."""
    import h5py
    with h5py.File(path, "r") as h:
        Re = float(h.attrs["Reynolds"])
        Sc = float(h.attrs["Schmidt"])
        n_traj = min(n_seed, h["t0_fields/tracer"].shape[0])
        sl = (slice(0, n_traj), slice(0, None, frame_stride))
        tracer = h["t0_fields/tracer"][sl]            # (nt, nf, 256, 512)
        pressure = h["t0_fields/pressure"][sl]        # (nt, nf, 256, 512)
        velocity = h["t1_fields/velocity"][sl]        # (nt, nf, 256, 512, 2)
    fields = np.stack([tracer, pressure,
                       velocity[..., 0], velocity[..., 1]], axis=2)  # (nt, nf, 4, H, W)
    nt, nf = fields.shape[:2]
    fields = fields.reshape(nt * nf, 4, *fields.shape[3:]).astype(np.float32)
    fields = _resize(fields, side)
    return fields, Re, Sc, nt * nf


def _read_windows(path: str, n_seed: int, n_frames: int, dt: int, side: int):
    """One file -> (W, 4, n_frames, side, side) float32 windows.

    Frames are read at temporal stride ``dt`` then chunked into consecutive
    non-overlapping windows of ``n_frames`` (each window spans dt*(n_frames-1)
    timesteps, so it carries dynamics).
    """
    import h5py
    with h5py.File(path, "r") as h:
        Re = float(h.attrs["Reynolds"])
        Sc = float(h.attrs["Schmidt"])
        n_traj = min(n_seed, h["t0_fields/tracer"].shape[0])
        sl = (slice(0, n_traj), slice(0, None, dt))
        tracer = h["t0_fields/tracer"][sl]
        pressure = h["t0_fields/pressure"][sl]
        velocity = h["t1_fields/velocity"][sl]
    fields = np.stack([tracer, pressure,
                       velocity[..., 0], velocity[..., 1]], axis=2)  # (nt, F, 4, H, W)
    nt, F_read = fields.shape[:2]
    n_win = F_read // n_frames
    if n_win == 0:
        raise ValueError(f"n_frames={n_frames} > frames read ({F_read}); lower dt")
    fields = fields[:, :n_win * n_frames]
    H, W = fields.shape[-2:]
    fields = fields.reshape(nt, n_win, n_frames, 4, H, W)
    fields = fields.transpose(0, 1, 3, 2, 4, 5)        # (nt, n_win, 4, n_frames, H, W)
    fields = fields.reshape(nt * n_win, 4, n_frames, H, W).astype(np.float32)
    Wn = fields.shape[0]
    # resize H,W by folding (channel, frame) onto the interpolate channel axis
    out = _resize(fields.reshape(Wn, 4 * n_frames, H, W), side)
    out = out.reshape(Wn, 4, n_frames, side, side)
    return out, Re, Sc, Wn


# ----------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------
class _ShearFlowBase(Dataset):
    """Holds normalized fields + log10 labels; subclasses fill ``self.fields``."""

    def _finalize(self, fields_list, re_list, sc_list, stats):
        self.fields = np.concatenate(fields_list, axis=0).astype(np.float32, copy=False)
        fields_list.clear()                                         # free per-file arrays
        self.logRe = np.concatenate(re_list).astype(np.float32)
        self.Sc = np.concatenate(sc_list).astype(np.float32)   # raw Schmidt
        ax = (0,) + tuple(range(2, self.fields.ndim))               # all but channel
        if stats is None:
            mean = self.fields.mean(axis=ax, keepdims=True).astype(np.float32)
            std = (self.fields.std(axis=ax, keepdims=True) + 1e-6).astype(np.float32)
        else:
            mean, std = stats
        self.fields -= mean                                         # in-place: no 2x copy
        self.fields /= std
        self.stats = (mean, std)

    def __len__(self):
        return len(self.fields)

    def __getitem__(self, i):
        y = torch.tensor([self.logRe[i], self.Sc[i]], dtype=torch.float32)
        return torch.from_numpy(self.fields[i]), y


class ShearFlowSnapshotDataset(_ShearFlowBase):
    """Single-frame snapshots -> field (4, side, side); label [log10 Re, log10 Sc]."""

    def __init__(self, split: str, n_seed: int = 24, frame_stride: int = 12,
                 side: int = 224, stats=None):
        super().__init__()
        f_list, re_list, sc_list = [], [], []
        for path in _split_files(split):
            fields, Re, Sc, m = _read_snapshots(path, n_seed, frame_stride, side)
            f_list.append(fields)
            re_list.append(np.full(m, np.log10(Re)))
            sc_list.append(np.full(m, Sc))   # raw Schmidt (paper: compression None)
        self._finalize(f_list, re_list, sc_list, stats)


class ShearFlowWindowDataset(_ShearFlowBase):
    """n_frames windows -> field (4, n_frames, side, side); label [log10 Re, log10 Sc]."""

    def __init__(self, split: str, n_seed: int = 24, n_frames: int = 16,
                 side: int = 224, stats=None, dt: int = 4):
        super().__init__()
        f_list, re_list, sc_list = [], [], []
        for path in _split_files(split):
            fields, Re, Sc, w = _read_windows(path, n_seed, n_frames, dt, side)
            f_list.append(fields)
            re_list.append(np.full(w, np.log10(Re)))
            sc_list.append(np.full(w, np.log10(Sc)))
        self._finalize(f_list, re_list, sc_list, stats)


class ShearFlowPairDataset(_ShearFlowBase):
    """Temporal pairs (field_t, field_{t+h}) for latent-prediction SSL.

    Frames are read at ``frame_stride``; ``horizon`` is in stride units, so the
    physical gap is ``horizon * frame_stride`` timesteps. __getitem__ -> (A, B, y)
    with A=field_t, B=field_{t+h}, both (4, side, side); y=[log10 Re, raw Sc].
    """
    def __init__(self, split, n_seed=24, frame_stride=12, horizon=1, side=224, stats=None):
        super().__init__()
        import h5py
        a_list, b_list, re_list, sc_list = [], [], [], []
        for path in _split_files(split):
            with h5py.File(path, "r") as h:
                Re = float(h.attrs["Reynolds"]); Sc = float(h.attrs["Schmidt"])
                nt = min(n_seed, h["t0_fields/tracer"].shape[0])
                sl = (slice(0, nt), slice(0, None, frame_stride))
                tr = h["t0_fields/tracer"][sl]; pr = h["t0_fields/pressure"][sl]
                ve = h["t1_fields/velocity"][sl]
            fields = np.stack([tr, pr, ve[..., 0], ve[..., 1]], axis=2)   # (nt, nf, 4, H, W)
            nt, nf = fields.shape[:2]
            if nf <= horizon:
                continue
            fields = _resize(fields.reshape(-1, 4, *fields.shape[3:]).astype(np.float32), side)
            fields = fields.reshape(nt, nf, 4, side, side)
            A = fields[:, :nf - horizon].reshape(-1, 4, side, side)
            B = fields[:, horizon:].reshape(-1, 4, side, side)
            m = A.shape[0]
            a_list.append(A); b_list.append(B)
            re_list.append(np.full(m, np.log10(Re))); sc_list.append(np.full(m, Sc))
        self.A = np.concatenate(a_list).astype(np.float32)
        self.B = np.concatenate(b_list).astype(np.float32)
        a_list.clear(); b_list.clear()
        self.logRe = np.concatenate(re_list).astype(np.float32)
        self.Sc = np.concatenate(sc_list).astype(np.float32)
        ax = (0, 2, 3)
        if stats is None:
            mean = self.A.mean(axis=ax, keepdims=True).astype(np.float32)
            std = (self.A.std(axis=ax, keepdims=True) + 1e-6).astype(np.float32)
        else:
            mean, std = stats
        self.A -= mean; self.A /= std
        self.B -= mean; self.B /= std
        self.stats = (mean, std)
        self.fields = self.A                                              # len/probe compat

    def __len__(self):
        return len(self.A)

    def __getitem__(self, i):
        y = torch.tensor([self.logRe[i], self.Sc[i]], dtype=torch.float32)
        return torch.from_numpy(self.A[i]), torch.from_numpy(self.B[i]), y


class ShearFlowClipDataset(_ShearFlowBase):
    """Short strided clips for VARIABLE-Δt latent prediction.

    Yields ``(clip, y)`` with clip ``(4, clip_len, side, side)`` of frames spaced
    ``frame_stride`` apart; the trainer samples a pair ``(t0, t0+Δ)`` and a gap Δ
    INSIDE the loop, so one dataset serves the whole horizon distribution.
    """
    def __init__(self, split, n_seed=24, frame_stride=4, clip_len=8, side=224, stats=None, traj_parity=None):
        super().__init__()                                               # traj_parity 0/1 -> disjoint trajectory halves
        import h5py
        rh, rw = (side, side) if isinstance(side, int) else side          # (h, w) tuple -> native-aspect, else square
        c_list, re_list, sc_list = [], [], []
        for path in _split_files(split):
            with h5py.File(path, "r") as h:
                Re = float(h.attrs["Reynolds"]); Sc = float(h.attrs["Schmidt"])
                nt = min(n_seed, h["t0_fields/tracer"].shape[0])
                sl = (slice(0, nt), slice(0, None, frame_stride))
                tr = h["t0_fields/tracer"][sl]; pr = h["t0_fields/pressure"][sl]
                ve = h["t1_fields/velocity"][sl]
            fields = np.stack([tr, pr, ve[..., 0], ve[..., 1]], axis=2)     # (nt, nf, 4, H, W)
            nt, nf = fields.shape[:2]
            ncl = nf // clip_len
            if ncl == 0:
                continue
            fields = fields[:, :ncl * clip_len]
            H, W = fields.shape[-2:]
            fields = _resize(fields.reshape(-1, 4, H, W).astype(np.float32), side)
            fields = fields.reshape(nt * ncl, clip_len, 4, rh, rw)
            clips = fields.transpose(0, 2, 1, 3, 4)                          # (N, 4, clip_len, rh, rw)
            if traj_parity is not None:                                      # trajectory-disjoint split (shear has no test dir)
                keep = (np.arange(nt * ncl) // ncl) % 2 == traj_parity       # clip j -> trajectory j//ncl
                clips = clips[keep]
            c_list.append(clips); m = clips.shape[0]
            re_list.append(np.full(m, np.log10(Re))); sc_list.append(np.full(m, Sc))
        self.clips = np.concatenate(c_list).astype(np.float32)
        c_list.clear()
        self.logRe = np.concatenate(re_list).astype(np.float32)
        self.Sc = np.concatenate(sc_list).astype(np.float32)
        ax = (0, 2, 3, 4)                                                    # all but channel
        if stats is None:
            mean = self.clips.mean(axis=ax, keepdims=True).astype(np.float32)
            std = (self.clips.std(axis=ax, keepdims=True) + 1e-6).astype(np.float32)
        else:
            mean, std = stats
        self.clips -= mean; self.clips /= std
        self.stats = (mean, std)
        self.fields = self.clips                                            # len/probe compat

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        y = torch.tensor([self.logRe[i], self.Sc[i]], dtype=torch.float32)
        return torch.from_numpy(self.clips[i]), y


if __name__ == "__main__":
    # Tiny self-check (1 trajectory, coarse frames) — verifies IO + shapes.
    ds = ShearFlowSnapshotDataset("valid", n_seed=1, frame_stride=50, side=128)
    f, y = ds[0]
    print(f"snapshot: N={len(ds)} field={tuple(f.shape)} label={y.tolist()} "
          f"| field mean={float(f.mean()):+.3f} std={float(f.std()):.3f}")
    print(f"  logRe range [{ds.logRe.min():.2f}, {ds.logRe.max():.2f}] "
          f"Sc range [{ds.Sc.min():.2f}, {ds.Sc.max():.2f}]")
    c2 = make_coords_2d(128)
    idx = torch.arange(10)
    tok = fields_to_tokens(f.unsqueeze(0), idx)
    print(f"  coords2d={tuple(c2.shape)} tokens={tuple(tok.shape)} (expect (1,10,4))")
    dw = ShearFlowWindowDataset("valid", n_seed=1, n_frames=4, side=64, dt=20)
    fw, yw = dw[0]
    print(f"window: N={len(dw)} field={tuple(fw.shape)} (expect (4,4,64,64)) "
          f"coords3d={tuple(make_coords_3d(4, 64).shape)}")
