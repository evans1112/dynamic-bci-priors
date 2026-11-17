"""Does prestimulus occipital alpha shift the causal prior (b) or the numerosity prior (c) on single trials?

z(t): alpha power (8-12.5 Hz, -0.6..-0.2 s) averaged over the occipital sensors of the decile analysis of Rohe et al.
(2019), z-scored within run, 0 on trials without clean EEG.  logit P_alpha(t) = logit P(t) + b z(t), or
mu_alpha(t) = mu_P(t) + c z(t) (transient), or both.
Models: markov (sensory parameters and mu_P fixed at the static fit; free: p_common, kappa, b) and markov_num_est, the
full model (sensory parameters fixed at its fit; free: p_common, mu_P(1), kappa, eta, and b and/or c).  Control: trial
position within the run (z) as a second shift of the logit prior.
Starting points: the original fit and 9 combinations of p_common (0.03, 0.5, 0.97) x kappa (0.05, 0.5, 0.95), because
the likelihood can have separate optima (e.g. p_common near 0 vs near 1); fits with c start from the original fit and
from the solution for b, each with several values of c.
Output: derivatives/brain_state/sub-XX.csv (b fits), derivatives/brain_state_num/sub-XX.csv (c and b + c fits)
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DERIV, N_JOBS                         # noqa: E402
from lib import data, eeg, models as R, state as RS      # noqa: E402

M = R.M
OUT_B = os.path.join(DERIV, "brain_state")
OUT_C = os.path.join(DERIV, "brain_state_num")
for _d in (OUT_B, OUT_C):
    os.makedirs(_d, exist_ok=True)
FREE = {"markov": [0, 5], "markov_num_est": [0, 4, 5, 6]}      # free entries of the parameter vector
FULL = "markov_num_est"


def run(full, free, u, base, d, offP, offMu, dyn_mu=None):
    v = full.copy(); v[free] = u[:len(free)]
    p = R.unpack(base, v[:R.K[base]])
    arrs, mI, tail, wO, mu_lo = R.tables(p, p["eta"] > 0 if dyn_mu is None else dyn_mu)
    ll = RS.run_state(*arrs, mI, tail, wO, mu_lo, R.MU_STEP, d.cond, d.task, d.resp, d.nA, d.nV, d.run_start,
                      p["p_common"], p["kappa"], p["rule"], p["mu_P"], p["eta"], p["target"], p["rho"], M.LAPSE,
                      offP, offMu)
    return -float(ll.sum())


def nll_b(u, base, full, free, d, z, pos):
    """u = [free parameters, b (, weight of trial position)]."""
    off = u[len(free)] * z + (u[len(free) + 1] * pos if pos is not None else 0.0)
    return run(full, free, u, base, d, off, np.zeros(len(z)))


def nll_c(u, full, d, z, both):
    """u = [free parameters of the full model, (b,) c]."""
    free = FREE[FULL]
    b = u[4] if both else 0.0
    return run(full, free, u, FULL, d, b * z, u[-1] * z, dyn_mu=True)


def params(base, full, free, u):
    v = full.copy(); v[free] = u[:len(free)]
    p = R.unpack(base, v[:R.K[base]])
    return dict(p_common=p["p_common"], kappa=p["kappa"], eta=p["eta"], mu_P=p["mu_P"])


def subject(sid):
    if os.path.exists(os.path.join(OUT_C, f"sub-{sid:02d}.csv")):
        return
    M._set_grid(*R.FIT_GRID)
    d = M.SubjectData(data.load_av().query("subject == @sid"))
    z = eeg.alpha_z(sid, d)
    pos = (d.trial_in_run - d.trial_in_run.mean()) / d.trial_in_run.std()
    fits = pd.read_csv(os.path.join(DERIV, "fits", f"sub-{sid:02d}.csv")).set_index("model")
    rows, roi = [], {}
    for base in ("markov", FULL):
        full = np.array(json.loads(fits.loc[base, "x"]))
        if base == "markov":
            full[1:5] = np.array(json.loads(fits.loc["static", "x"]))[1:5]
        free = FREE[base]
        grid = [full[free]]
        for pc in (0.03, 0.5, 0.97):
            for kp in (0.05, 0.5, 0.95):
                g = full[free].copy(); g[0] = M._logit(pc); g[1 if base == "markov" else 2] = M._logit(kp)
                grid.append(g)
        for with_pos in (False, True):
            ps = pos if with_pos else None
            extra = 2 if with_pos else 1
            x0, f0 = M._multi(lambda u, *a: nll_b(np.r_[u, np.zeros(extra)], *a), grid, (base, full, free, d, z, ps), polish=True)
            starts = [np.r_[g, np.zeros(extra)] for g in grid] + [np.r_[x0, s, np.zeros(extra - 1)] for s in (0.0, -0.3, 0.3)]
            u, fv = M._multi(nll_b, starts, (base, full, free, d, z, ps), polish=True)
            rows.append(dict(subject=sid, base=base, position=with_pos, b=u[len(free)], ll0=-f0, ll=-fv,
                             **params(base, full, free, u)))
            if base == FULL and not with_pos:
                roi = dict(u=u, ll0=-f0)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_B, f"sub-{sid:02d}.csv"), index=False)

    # alpha -> numerosity prior (c), and both shifts (b + c), in the full model
    full = np.array(json.loads(fits.loc[FULL, "x"]))
    free = FREE[FULL]
    xr = roi["u"][:4]
    x0, f0 = M._multi(lambda u, *a: nll_c(np.r_[u, 0.0], *a), [full[free], xr], (full, d, z, False), polish=True)
    uc, fc = M._multi(nll_c, [np.r_[x0, c0] for c0 in (0.0, -0.3, 0.3)] + [np.r_[xr, 0.0]], (full, d, z, False), polish=True)
    ubc, fbc = M._multi(nll_c, [np.r_[uc[:4], 0.0, uc[4]], np.r_[xr, roi["u"][4], 0.0], np.r_[x0, 0.0, 0.0]],
                        (full, d, z, True), polish=True)
    rows = [dict(subject=sid, state="c", b=np.nan, c=uc[-1], ll0=-f0, ll=-fc, **params(FULL, full, free, uc)),
            dict(subject=sid, state="bc", b=ubc[4], c=ubc[-1], ll0=-f0, ll=-fbc, **params(FULL, full, free, ubc))]
    pd.DataFrame(rows).to_csv(os.path.join(OUT_C, f"sub-{sid:02d}.csv"), index=False)
    print(sid, "done", flush=True)


if __name__ == "__main__":
    ids = [int(s) for s in sys.argv[1:]] or data.ids()
    Parallel(n_jobs=N_JOBS)(delayed(subject)(s) for s in ids)
