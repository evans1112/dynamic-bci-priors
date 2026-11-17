"""S2 Fig. Pull of the report towards the previous report (results of 06_pull.py)."""
import os
import sys
import pandas as pd
from scipy import stats
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RES                                  # noqa: E402
from lib import style as F                              # noqa: E402
import matplotlib.pyplot as plt                         # noqa: E402

Cv = pd.read_csv(os.path.join(RES, "pull_curve.csv"))
B = pd.read_csv(os.path.join(RES, "pull_betas.csv"))
L = pd.read_csv(os.path.join(RES, "pull_lags.csv"))
TASK = {1: "auditory report", 2: "visual report"}
LS = {1: "solid", 2: "dashed"}

W, H = 7.5, 2.6
fig = plt.figure(figsize=(W, H))
ax_ = lambda x, y, w, h: F.inch_axes(fig, x, y, w, h, W, H)

# A: residual report by previous report
for k, g in enumerate(("HC", "SCZ")):
    ax = ax_(0.6 + k * 1.45, 0.55, 1.2, 1.6)
    for tk in (1, 2):
        m = Cv[(Cv.group == g) & (Cv.task == tk)].groupby("prev_report").res
        ax.errorbar(m.mean().index, m.mean(), m.sem(), color=F.COL[g], ls=LS[tk], marker="o", ms=3.5, mec="white",
                    mew=0.5, lw=1.2, label=TASK[tk])
    ax.axhline(0, color=F.INK2, lw=0.6, ls="dotted")
    ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel("previous report")
    ax.set_ylim(-0.25, 0.32); ax.set_title(F.GLAB[g], loc="left", color=F.COL[g])
    if k == 0:
        ax.set_ylabel("residual report")
        ax.legend(loc="upper left", fontsize=8, handlelength=1.6, borderaxespad=0.1)
    else:
        ax.set_yticklabels([])
F.letter(fig, 0.05, H - 0.05, "A", W, H)

# B: weights of the previous report and the previous numbers
ax = ax_(3.65, 0.55, 2.0, 1.6)
cols = [("prev_report", "report"), ("prev_relevant", "relevant\nnumber"), ("prev_irrelevant", "irrelevant\nnumber")]
for ci, (col, lab) in enumerate(cols):
    for j, g in enumerate(("HC", "SCZ")):
        for tk in (1, 2):
            v = B[(B.group == g) & (B.task == tk)][col]
            ax.errorbar(ci * 3 + j * 1.2 + (tk - 1.5) * 0.45, v.mean(), stats.sem(v), fmt="o" if tk == 1 else "s",
                        color=F.COL[g], ms=3.5, lw=1, mfc=F.COL[g] if tk == 1 else "white")
ax.axhline(0, color=F.INK2, lw=0.6, ls="dotted")
ax.set_xticks([0.6, 3.6, 6.6]); ax.set_xticklabels([c[1] for c in cols])
ax.set_ylabel("weight on residual"); ax.set_title("Previous trial", loc="left")
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=F.INK2, ms=4, label="auditory report"),
                   Line2D([], [], marker="s", ls="", color=F.INK2, mfc="white", ms=4, label="visual report")],
          loc="upper right", fontsize=8, handletextpad=0.2)
F.letter(fig, 3.1, H - 0.05, "B", W, H)

# C: report k trials back
ax = ax_(6.15, 0.55, 1.2, 1.6)
for g in ("HC", "SCZ"):
    for tk in (1, 2):
        m = L[(L.group == g) & (L.task == tk)].groupby("lag").slope
        ax.errorbar(m.mean().index, m.mean(), m.sem(), color=F.COL[g], ls=LS[tk], marker="o", ms=3, lw=1.1)
ax.axhline(0, color=F.INK2, lw=0.6, ls="dotted")
ax.set_xticks([1, 3, 5]); ax.set_xlabel("trials back"); ax.set_ylabel("slope on previous report")
ax.set_title("Lag", loc="left")
F.letter(fig, 5.6, H - 0.05, "C", W, H)
F.save(fig, "S2_Fig")
