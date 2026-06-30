"""Draggable 3D view of the spherical shallow-water data: vorticity on the sphere, with a time slider.
Writes a standalone HTML (open in a browser / VS Code preview, drag to rotate, slider for time)."""
import os, sys, argparse
import numpy as np
import plotly.graph_objects as go

ap = argparse.ArgumentParser()
ap.add_argument("--npz", default=os.path.expanduser("~/scratch/sw_data/sw_smoke.npz"))
ap.add_argument("--sample", type=int, default=0)
ap.add_argument("--stride", type=int, default=2, help="spatial downsample for a lighter HTML")
ap.add_argument("--out", default="results/figs/sw_sphere.html")
args = ap.parse_args()

d = np.load(args.npz)
vort = d["outputs"][args.sample][:, ::args.stride, ::args.stride]    # (T, Nphi, Ntheta)
alpha, beta = d["params"][args.sample]
T, Nphi, Nth = vort.shape

phi = np.linspace(0, 2 * np.pi, Nphi)
theta = np.linspace(0, np.pi, Nth)
PHI, TH = np.meshgrid(phi, theta, indexing="ij")
X, Y, Z = np.sin(TH) * np.cos(PHI), np.sin(TH) * np.sin(PHI), np.cos(TH)
cmax = float(np.abs(vort).max())

def surf(k):
    return go.Surface(x=X, y=Y, z=Z, surfacecolor=vort[k], colorscale="RdBu_r",
                      cmin=-cmax, cmax=cmax, colorbar=dict(title="vorticity"))

fig = go.Figure(data=[surf(0)], frames=[go.Frame(data=[surf(k)], name=str(k)) for k in range(T)])
fig.update_layout(
    title=f"Shallow-water vorticity on the sphere — sample {args.sample} (α={alpha:.3f}, β={beta:.3f}), {T} frames",
    scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
    width=820, height=760,
    updatemenus=[dict(type="buttons", x=0.05, y=0.05, buttons=[
        dict(label="▶ play", method="animate", args=[None, dict(frame=dict(duration=300, redraw=True), fromcurrent=True)]),
        dict(label="⏸ pause", method="animate", args=[[None], dict(mode="immediate")])])],
    sliders=[dict(active=0, currentvalue=dict(prefix="t = "),
                  steps=[dict(method="animate", label=str(k),
                              args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                         for k in range(T)])])
os.makedirs(os.path.dirname(args.out), exist_ok=True)
fig.write_html(args.out, include_plotlyjs="cdn")
print(f"wrote {args.out}  ({T} frames, {Nphi}x{Nth} sphere, vort=[-{cmax:.3f},{cmax:.3f}])", flush=True)
