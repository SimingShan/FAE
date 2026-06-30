"""Uniform paper-style plotting — the ONE rule for every WFAE figure.

    from src.utils.plotstyle import apply, COLORS, panels
    apply()
    fig, ax = panels(1)              # one square panel
    fig, (a0, a1) = panels(2)        # two square panels

Design: square panels, NO background grid, top/right spines off, Nature(NPG)-flavored palette,
font 16 / ticks 14. Edit here -> every figure updates.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# Nature Publishing Group (npg / ggsci) qualitative palette
NPG = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
       "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85"]

# fixed method <-> color map (consistent across ALL figures)
COLORS = {"FAE": "#E64B35", "MAE": "#4DBBD5", "JEPA": "#00A087",
          "pixel": "#3C5488", "floor": "#8491B4", "Senseiver": "#7E6148"}

FONT, TICK = 16, 14


def apply():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FONT, "axes.titlesize": FONT, "axes.labelsize": FONT,
        "xtick.labelsize": TICK, "ytick.labelsize": TICK, "legend.fontsize": TICK - 1,
        "axes.grid": False,                          # NO background grid
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 1.1, "lines.linewidth": 2.2, "lines.markersize": 6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 4.5, "ytick.major.size": 4.5,
        "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=NPG),
    })


def panels(ncols=1, nrows=1, side=5.0):
    """fig + flat list of SQUARE axes (each box forced to 1:1 via set_box_aspect)."""
    fig, axes = plt.subplots(nrows, ncols, figsize=(side * ncols, side * nrows), squeeze=False)
    flat = axes.ravel().tolist()
    for a in flat:
        a.set_box_aspect(1)
    return fig, (flat[0] if len(flat) == 1 else flat)
