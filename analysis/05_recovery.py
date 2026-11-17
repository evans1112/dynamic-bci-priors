"""Parameter recovery for the delta-rule and Markov causal priors and for the full model.

Per participant and model, N_REC data sets are simulated from the participant's fit with the participant's own trial
sequence (missed trials kept missed) and refitted with the procedure of 03_fit.py.
Usage: python 05_recovery.py [--models learn,markov,markov_num_est] [participants]
Output: derivatives/recovery/sub-XX.csv, results/recovery.csv
"""
import os
import sys
import json
import glob
import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DERIV, RES, N_JOBS                    # noqa: E402
from lib import data, models as R                        # noqa: E402

M = R.M
OUT = os.path.join(DERIV, "recovery")
os.makedirs(OUT, exist_ok=True)
N_REC = int(os.environ.get("N_REC", 2))
MODELS = {"markov_num_est": 0, "learn": 1, "markov": 2}          # seed offsets
PARS = ["p_common", "sigma_A", "sigma_V", "sigma_P", "mu_P", "kappa", "eta"]
lg = M._logit


def refit(name, d, seed):
    xs, _ = M.fit_static(d, n_starts=8, seed=seed)
    if name == "learn":
        return M.fit_dynamic(d, xs, kappas=(0.01, 0.05, 0.2), rule=0)[0]
    xm, _ = M.fit_dynamic(d, xs, kappas=(0.95, 0.8, 0.5), rule=1)
    if name == "markov":
        return xm
    xn, f = M._multi(R.nll, [np.r_[xs, lg(e)] for e in (0.02, 0.1, 0.3)], ("num_est", d), polish=True)
    if R.nll(np.r_[xs, R.NULL], "num_est", d) < f:
        xn = np.r_[xs, R.NULL]
    x, f = M._multi(R.nll, [np.r_[xm, lg(e)] for e in (0.02, 0.1)] + [np.r_[xn[:5], xm[5], xn[5]]], (name, d), polish=True)
    return np.r_[xm, R.NULL] if R.nll(np.r_[xm, R.NULL], name, d) < f else x


def subject(sid, models):
    f = os.path.join(OUT, f"sub-{sid:02d}.csv")
    old = pd.read_csv(f) if os.path.exists(f) else pd.DataFrame(columns=["model"])
    todo = [m for m in models if m not in set(old.model)]
    if not todo:
        return
    M._set_grid(*R.FIT_GRID)
    d = M.SubjectData(data.load_av().query("subject == @sid"))
    fits = pd.read_csv(os.path.join(DERIV, "fits", f"sub-{sid:02d}.csv")).set_index("model")
    rows = []
    for name in todo:
        x_true = np.array(json.loads(fits.loc[name, "x"]))
        p_true = R.unpack(name, x_true)
        for rep in range(N_REC):
            resp = R.sim(name, x_true, d, 104729 * sid + 1000 * MODELS[name] + rep)
            df = d.df.copy()
            df["countingReport"] = np.where((d.resp > 0) & (resp > 0), resp, np.nan)
            x_fit = refit(name, M.SubjectData(df), seed=sid + 100 * rep)
            p_fit = R.unpack(name, x_fit)
            rows.append(dict(subject=sid, model=name, rep=rep, **{f"true_{k}": p_true[k] for k in PARS},
                             **{f"fit_{k}": p_fit[k] for k in PARS}))
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(f, index=False)
    print(sid, "done", flush=True)


def summary():
    D = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(OUT, "sub-*.csv")))], ignore_index=True)
    rows = []
    for m, x in D.groupby("model"):
        for k in PARS:
            if k == "eta" and m != "markov_num_est":
                continue
            t, e = x[f"true_{k}"], x[f"fit_{k}"]
            rows.append(dict(model=m, parameter=k, n=len(x), spearman_r=stats.spearmanr(t, e)[0],
                             median_error=float(np.median(e - t)), median_abs_error=float(np.median(np.abs(t - e)))))
        if m == "markov_num_est":                            # the initial mu_P matters little when eta is high
            for lab, sel in (("mu_P, eta <= 0.5", x.true_eta <= 0.5), ("mu_P, eta > 0.5", x.true_eta > 0.5)):
                rows.append(dict(model=m, parameter=lab, n=int(sel.sum()),
                                 spearman_r=stats.spearmanr(x[sel].true_mu_P, x[sel].fit_mu_P)[0]))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(RES, "recovery.csv"), index=False)
    print(T.round(3).to_string(index=False))


if __name__ == "__main__":
    args = sys.argv[1:]
    models = list(MODELS)
    if "--models" in args:
        i = args.index("--models"); models = args[i + 1].split(","); args = args[:i] + args[i + 2:]
    ids = [int(s) for s in args] or data.ids()
    Parallel(n_jobs=N_JOBS)(delayed(subject)(s, models) for s in ids)
    summary()
