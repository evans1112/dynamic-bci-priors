"""Fig 1.
A  model comparison: summed dAIC relative to static priors (left); dAIC from adding the Markov causal prior to the
   static-priors, response-repetition and dynamic-numerosity-prior models, per participant (right)
B  history effect of Rohe et al. (2019): data and simulations of each model
C  conditional tests: previous report held constant; sequences in which the Markov causal prior and the pull towards
   the previous report predict opposite shifts
D  prestimulus occipital alpha -> causal prior in the full model: group t of b per sensor; occipital sensors outlined
   (p of the a priori test below each map)
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DERIV, RES                           # noqa: E402
from lib import data, eeg, style as F                   # noqa: E402
import matplotlib.pyplot as plt                         # noqa: E402

MOD = {  # key: (label, colour, line style); labels shared with the text and the supplement
    "static": ("Static priors", F.STATIC, (0, (1, 1.3))),
    "learn": ("Delta-rule causal prior", F.LEARN, "dashed"),
    "markov": ("Markov causal prior", F.MKV, "solid"),
    "rep": ("Response repetition", F.REP, "solid"),
    "num_est": ("Dynamic numerosity prior", F.NUM, "solid"),
    "markov_rep": ("Markov causal prior + response repetition", F.REP, "dashed"),
    "markov_num_est": ("Full model", F.FULL, "solid")}
SIM = ["static", "rep", "num_est", "markov", "markov_num_est"]

W, H = 7.5, 5.55
fig = plt.figure(figsize=(W, H))
ax_ = lambda x, y, w, h: F.inch_axes(fig, x, y, w, h, W, H)
# grid (inches): row 1 axes 3.20-4.80, row 2 axes 0.68-2.20; title line and sub-title line above each row
Y1, H1, T1 = 3.20, 1.60, 5.22
Y2, H2, T2 = 0.68, 1.52, 2.62
SUB = 0.24                                              # title line -> sub-title line


def head(x, y, letter, title=None):
    F.letter(fig, x, y + 0.02, letter, W, H)
    if title:
        fig.text((x + 0.2) / W, y / H, title, fontsize=8.5, fontweight="bold", va="top")


def sub(x, y, s, color=F.INK, ha="left"):
    fig.text(x / W, y / H, s, fontsize=8, fontweight="bold", color=color, va="top", ha=ha)


# ------------------------------------------------------------------ key (B, C)
fig.legend(handles=[Line2D([], [], color=MOD[m][1], ls=MOD[m][2], lw=1.5, label=MOD[m][0]) for m in SIM],
           loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5, handlelength=1.8, columnspacing=1.4,
           handletextpad=0.4, borderaxespad=0.25)

# ------------------------------------------------------------------ A: model comparison
Fi = pd.read_csv(os.path.join(RES, "fits.csv"))
order = ["learn", "markov", "rep", "num_est", "markov_rep", "markov_num_est"]
head(0.0, T1, "A", "Model comparison")
x0 = 0.14
sub(x0, T1 - SUB, "Each model vs static priors")
ax = ax_(x0, Y1, 2.1, H1)
ys = [6.6, 5.45, 4.3, 3.15, 2.0, 0.42]                  # last row has a two-line label
for m, y in zip(order, ys):
    for j, g in enumerate(("HC", "SCZ")):
        ax.barh(y - j * 0.25, Fi[(Fi.group == g) & (Fi.model == m)].dAIC.sum(), height=0.21, color=F.COL[g], lw=0)
    if m == "markov_num_est":
        ax.text(14, y + 0.2, "(Markov causal + dynamic numerosity prior)", ha="left", va="bottom", fontsize=8)
        ax.text(14, y + 0.72, "Full model", ha="left", va="bottom", fontsize=8, fontweight="bold")
    else:
        ax.text(14, y + 0.2, MOD[m][0], ha="left", va="bottom", fontsize=8)
ax.set_ylim(0.05, 7.4); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.axvline(0, color=F.INK, lw=0.6)
ax.set_xlim(0, 1260); ax.set_xticks([0, 400, 800, 1200]); ax.set_xlabel("summed ΔAIC (positive favours the model)")
ax.legend(handles=[Patch(color=F.HC, label="Controls"), Patch(color=F.SCZ, label="Patients")], loc="upper right",
          handlelength=0.9, handleheight=0.7, handletextpad=0.4, bbox_to_anchor=(1.0, 1.02))

x0 = 2.78
sub(x0 - 0.36, T1 - SUB, "Adding the Markov causal prior to …")
ax = ax_(x0, Y1, 1.7, H1)
pairs = [("static", "markov", "Static\npriors"), ("rep", "markov_rep", "Response\nrepetition"),
         ("num_est", "markov_num_est", "Dynamic\nnumerosity\nprior")]
for k, (a, b, lab) in enumerate(pairs):
    for j, g in enumerate(("HC", "SCZ")):
        P = Fi[Fi.group == g].pivot(index="subject", columns="model", values="aic")
        dv = (P[a] - P[b]).values
        x = k + (j - 0.5) * 0.4
        F.strip(ax, x, dv, F.COL[g], seed=k * 2 + j + 21, w=0.08, s=7)
        F.mean_sem(ax, x, dv, ms=3.6)
        ax.text(x, 19.3, f"{dv.sum():.0f}", ha="center", va="bottom", fontsize=8)
ax.text(-0.47, 19.3, "Σ", ha="left", va="bottom", fontsize=8)
ax.axhline(0, color=F.INK2, lw=0.6, ls="dashed")
ax.set_xticks(range(3)); ax.set_xticklabels([p[2] for p in pairs], linespacing=1.0); ax.tick_params(axis="x", length=0)
ax.set_xlim(-0.5, 2.5); ax.set_ylim(-4, 21.5); ax.set_yticks([0, 5, 10, 15])
ax.set_ylabel("ΔAIC per participant")

# ------------------------------------------------------------------ B: history effect
S = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(DERIV, "sims", "sub-*.csv"))])
S["group"] = S.subject.map(data.group_of())
S.loc[S.measure != "history", "lag"] = 1
Pm = S.groupby(["group", "subject", "source", "measure", "task", "lag"]).value.mean().reset_index()
G = S[S.source != "data"].groupby(["group", "source", "sim", "measure", "task", "lag"]).value.mean().reset_index()
lags = np.arange(1, 6)
xb = 5.14
head(4.6, T1, "B", "History effect")
for k, g in enumerate(("HC", "SCZ")):
    ax = ax_(xb + k * 1.17, Y1, 1.06, H1)
    for m in SIM:
        q = G[(G.group == g) & (G.source == m) & (G.measure == "history")].groupby("lag").value
        lab, col, ls = MOD[m]
        if m == "markov_num_est":
            ax.fill_between(lags, q.quantile(.025), q.quantile(.975), color=col, alpha=0.09, lw=0)
        ax.plot(lags, q.mean(), color=col, ls=ls, lw=1.6 if m in ("markov_num_est", "markov") else 1.2)
    dd = Pm[(Pm.group == g) & (Pm.source == "data") & (Pm.measure == "history")].groupby("lag").value
    ax.errorbar(lags, dd.mean(), dd.sem(), fmt="o", color=F.COL[g], ms=4, mec="white", mew=0.6, lw=1, zorder=6)
    ax.axhline(0, color=F.INK2, lw=0.6, ls="dashed")
    ax.set_xticks(lags); ax.set_xlabel("trials back"); ax.set_ylim(-0.06, 0.13); ax.set_yticks([-0.05, 0, 0.05, 0.1])
    sub(xb + k * 1.17, T1 - SUB, F.GLAB[g], F.COL[g])
    if k == 0:
        ax.set_ylabel("Δ$p_{\\mathrm{common}}$, small − large\nprevious disparity")
    else:
        ax.set_yticklabels([])

# ------------------------------------------------------------------ C: conditional tests
rows = ["static", "rep", "num_est", "markov", "markov_num_est"]
tests = [("strict", "Previous report held constant", (-0.03, 0.06), [0, 0.05],
          "shift towards irrelevant number,\nsmall − large previous disparity"),
         ("loose", "Opposite predictions", (-0.085, 0.035), [-0.05, 0],
          "residual report\n← previous report · causal prior →")]
xc, wc, gap_g, gap_t = 1.36, 0.78, 0.08, 0.34
head(0.0, T2, "C")
for ti, (meas, ttl, lim, ticks, xl) in enumerate(tests):
    xt = xc + ti * (2 * wc + gap_g + gap_t)
    fig.text((xt + wc + gap_g / 2) / W, T2 / H, ttl, ha="center", va="top", fontsize=8.5, fontweight="bold")
    for gi, g in enumerate(("HC", "SCZ")):
        xa = xt + gi * (wc + gap_g)
        ax = ax_(xa, Y2, wc, H2)
        v = Pm[(Pm.group == g) & (Pm.source == "data") & (Pm.measure == meas)].value
        ax.axvspan(v.mean() - stats.sem(v), v.mean() + stats.sem(v), color=F.COL[g], alpha=0.13, lw=0)
        for ri, m in enumerate(rows):
            y = len(rows) - ri
            q = G[(G.group == g) & (G.source == m) & (G.measure == meas)].value
            ax.hlines(y, q.quantile(.025), q.quantile(.975), color=MOD[m][1], lw=3.4, alpha=0.45)
            ax.plot(q.mean(), y, "|", color=MOD[m][1], ms=7.5, mew=1.8)
        ax.errorbar(v.mean(), 0, xerr=stats.sem(v), fmt="o", color=F.COL[g], ms=4.4, mec="white", mew=0.5, lw=1.2)
        ax.axvline(0, color=F.INK2, lw=0.6, ls="dashed")
        ax.set_ylim(-0.6, len(rows) + 0.6); ax.set_xlim(*lim); ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}".replace("-", "−") for t in ticks])
        ax.set_yticks(range(len(rows) + 1))
        if ti == 0 and gi == 0:
            ax.set_yticklabels(["Data"] + [MOD[m][0] for m in rows][::-1])
            ax.get_yticklabels()[0].set_fontweight("bold")
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
        sub(xa + wc / 2, T2 - SUB, F.GLAB[g], F.COL[g], ha="center")
    fig.text((xt + wc + gap_g / 2) / W, (Y2 - 0.2) / H, xl, ha="center", va="top", fontsize=8, linespacing=1.15)

# ------------------------------------------------------------------ D: alpha -> causal prior
ch, info, xy = F.montage()
CB = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(DERIV, "channel_b", "sub-*.csv"))])
CB.loc[CB.p_common < 1e-4, "b"] = np.nan
ST = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(DERIV, "brain_state", "sub-*.csv"))])
ST = ST[(~ST.position) & (ST.base == "markov_num_est")]; ST.loc[ST.p_common < 1e-4, "b"] = np.nan
OCC = np.isin(ch, eeg.OCC)
xd, wd = 5.3, 1.04
head(5.12, T2, "D", "Prestimulus alpha → causal prior")
for k, g in enumerate(("HC", "SCZ")):
    Mb = CB[(CB.feature != "occ_roi") & CB.subject.isin(data.ids(g))].pivot(index="subject", columns="feature", values="b")[ch].dropna()
    ax = ax_(xd + k * (wd + 0.08), Y2 + 0.38, wd, wd)
    im = F.topo(ax, eeg.tvals(Mb.values), info, vlim=3.5)
    F.outline(ax, xy, OCC, lw=0.8)
    ax.scatter(xy[OCC, 0], xy[OCC, 1], s=1.8, color="white", edgecolor=F.INK, linewidth=0.3, zorder=5)
    sub(xd + k * (wd + 0.08) + wd / 2, T2 - SUB, F.GLAB[g], F.COL[g], ha="center")
    v = ST[ST.subject.isin(data.ids(g))].b.dropna()
    ax.text(0.5, -0.02, F.ptxt(stats.ttest_1samp(v, 0).pvalue), transform=ax.transAxes, ha="center", va="top", fontsize=8)
F.colorbar(fig, im, (xd + 0.5, Y2 - 0.12, 1.12, 0.065), W, H, [-3, 0, 3], orientation="horizontal")
fig.text((xd + 1.7) / W, (Y2 - 0.09) / H, "t of b", fontsize=8, va="center")

F.save(fig, "Fig1")
