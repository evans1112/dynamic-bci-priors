"""Figure style (Arial, >= 8 pt, 7.5 in wide) and small drawing helpers."""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
from scipy.interpolate import griddata                   # noqa: E402
from config import FIG                                   # noqa: E402
from lib import eeg                                      # noqa: E402

HC, SCZ = "#2a78d6", "#eb6834"
MKV, LEARN, STATIC, NUM, REP, FULL = "#1baf7a", "#4d4d4d", "#a6a6a6", "#7b5cc4", "#8c6d4f", "#1f1f1f"
INK, INK2 = "#000000", "#4d4d4d"
COL = {"HC": HC, "SCZ": SCZ}
GLAB = {"HC": "Controls", "SCZ": "Patients"}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.labelsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.pad": 2, "ytick.major.pad": 2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK, "legend.frameon": False, "legend.handlelength": 1.8,
    "legend.borderaxespad": 0.2, "legend.labelspacing": 0.3, "lines.linewidth": 1.3, "pdf.fonttype": 42,
    "ps.fonttype": 42, "savefig.dpi": 600, "figure.dpi": 150, "lines.dashed_pattern": [3, 1.6],
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "axes.titlepad": 4, "axes.labelpad": 2})


def inch_axes(fig, x, y, w, h, W, H):
    """Axes from a box in inches (x, y = lower left) on a W x H inch figure."""
    return fig.add_axes([x / W, y / H, w / W, h / H])


def letter(fig, x, y, s, W, H):
    fig.text(x / W, y / H, s, fontsize=11, fontweight="bold", ha="left", va="top")


def mean_sem(ax, x, v, ms=4.5):
    from scipy import stats
    ax.errorbar(x, np.mean(v), stats.sem(v), fmt="o", ms=ms, color=INK, mfc="white", mew=1.0, lw=1.1, capsize=0, zorder=6)


def strip(ax, x, v, color, seed=0, w=0.14, s=11):
    r = np.random.default_rng(int(seed * 10))
    ax.scatter(x + r.uniform(-w, w, len(v)), v, s=s, color=color, alpha=0.8, lw=0, zorder=3)


def ptxt(p):
    if p < 0.001:
        return "p < .001"
    return "p = " + (f"{p:.2f}" if p >= 0.1 else f"{p:.3f}").lstrip("0")


# ---------------------------------------------------------------------------------------------- scalp maps
def montage():
    """Channel names (as in the EEG features), MNE info and 2-D sensor positions."""
    import mne
    from mne.channels.layout import _find_topomap_coords
    ch = list(np.load(sorted(glob.glob(os.path.join(eeg.FEAT, "sub-*.npz")))[0])["ch_names"])
    info = mne.create_info(ch, 250.0, "eeg")
    info.set_montage("standard_1005")
    return ch, info, _find_topomap_coords(info, picks=list(range(len(ch))))


def outline(ax, xy, mask, lw=0.8):
    """Contour around the sensors in mask."""
    gx, gy = np.meshgrid(np.linspace(xy[:, 0].min() - 0.02, xy[:, 0].max() + 0.02, 250),
                         np.linspace(xy[:, 1].min() - 0.02, xy[:, 1].max() + 0.02, 250))
    z = griddata(xy, np.asarray(mask, float), (gx, gy), method="linear")
    ax.contour(gx, gy, z, levels=[0.5], colors=INK, linestyles="solid", linewidths=lw)


def topo(ax, values, info, vlim=3.5):
    import mne
    im, _ = mne.viz.plot_topomap(values, info, axes=ax, show=False, cmap="RdBu_r", vlim=(-vlim, vlim), contours=0,
                                 sensors=False, outlines="head")
    for ln in ax.lines:
        ln.set_linewidth(0.6)
    return im


def colorbar(fig, im, box, W, H, ticks, orientation="vertical"):
    cax = inch_axes(fig, *box, W, H)
    cb = fig.colorbar(im, cax=cax, orientation=orientation)
    cb.set_ticks(ticks); cb.outline.set_linewidth(0.5); cb.ax.tick_params(width=0.5, length=2.5)
    return cb


def save(fig, name):
    for ext, kw in ((".png", {"dpi": 600}), (".pdf", {}), (".tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}})):
        fig.savefig(os.path.join(FIG, name + ext), **kw)
    print("saved", name)
