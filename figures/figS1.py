"""S1 Fig. Parameter recovery for the delta-rule and Markov causal priors and for the full model."""
import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.ticker import FixedLocator, NullFormatter, FormatStrFormatter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DERIV                                # noqa: E402
from lib import style as F                              # noqa: E402
import matplotlib.pyplot as plt                         # noqa: E402

PARS = [("p_common", "$p_{\\mathrm{common}}$", None), ("sigma_A", "$\\sigma_A$", [0.3, 0.5, 0.8]),
        ("sigma_V", "$\\sigma_V$", [0.5, 1, 2]), ("sigma_P", "$\\sigma_P$", [1, 3, 10]), ("mu_P", "$\\mu_P$", None),
        ("kappa", "κ", None), ("eta", "η", None)]
ROWS = [("learn", "Delta-rule causal prior", F.LEARN), ("markov", "Markov causal prior", F.MKV),
        ("markov_num_est", "Full model", F.FULL)]

D = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(DERIV, "recovery", "sub-*.csv")))])
W, H = 7.5, 5.2
fig = plt.figure(figsize=(W, H))
fig.text(0.1 / W, (H - 0.07) / H, "Parameter recovery: data simulated from each participant's fit and refitted",
         fontsize=8.5, fontweight="bold", va="top")
for ri, (model, lab, col) in enumerate(ROWS):
    r = D[D.model == model]
    y = 3.75 - ri * 1.6
    fig.text(0.1 / W, (y + 0.46) / H, lab, rotation=90, ha="left", va="center", fontsize=8.5, fontweight="bold", color=col)
    for ci, (p, plab, logt) in enumerate(PARS):
        if p == "eta" and model != "markov_num_est":
            continue
        ax = F.inch_axes(fig, 0.72 + ci * 0.96, y, 0.76, 0.92, W, H)
        x, yy = r[f"true_{p}"].values, r[f"fit_{p}"].values
        lo, hi = np.nanmin(np.r_[x, yy]), np.nanmax(np.r_[x, yy])
        ax.plot([lo, hi], [lo, hi], color=F.STATIC, lw=0.8, zorder=1)
        ax.scatter(x, yy, s=9, color=col, lw=0, alpha=0.8, zorder=3)
        if logt:
            ax.set_xscale("log"); ax.set_yscale("log")
            for a in (ax.xaxis, ax.yaxis):
                a.set_major_locator(FixedLocator(logt)); a.set_major_formatter(FormatStrFormatter("%g"))
                a.set_minor_formatter(NullFormatter())
        else:
            t = [0, 2.5, 5] if p == "mu_P" else [0, 0.5, 1]
            lab_t = ["0", "2.5", "5"] if p == "mu_P" else ["0", ".5", "1"]
            ax.set_xticks(t); ax.set_yticks(t); ax.set_xticklabels(lab_t); ax.set_yticklabels(lab_t)
        rho = stats.spearmanr(x, yy)[0]
        ax.set_title(f"{plab}  r = {np.floor(rho * 100 + 0.5) / 100:.2f}", fontsize=8, fontweight="normal", loc="left")
        if ci == 0:
            ax.set_ylabel("recovered")
        if ri == len(ROWS) - 1:
            ax.set_xlabel("true")
F.save(fig, "S1_Fig")
