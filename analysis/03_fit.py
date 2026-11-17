"""Fit all behavioural models per participant (maximum likelihood).

static          static priors: p_common, sigma_A, sigma_V, sigma_P, mu_P (8 starting points)
learn, markov   + causal prior updated by the delta rule / the Markov rule (kappa)
rep             + response repetition (rho)
num_est         + dynamic numerosity prior, updated towards the report (eta)
num_stim        + dynamic numerosity prior, updated towards the presented numbers (eta)
markov_rep, markov_num_est (the full model), markov_num_stim, learn_num_est   combinations
Each model starts from the fit of the model(s) it extends, with several values of the new parameter, and the nested
solution is kept if it is better, so no model fits worse than the one it extends.  L-BFGS-B from every start, then
Nelder-Mead from the best; reported log-likelihoods are recomputed on a finer integration grid.
Output: derivatives/fits/sub-XX.csv, results/fits.csv
"""
import os
import sys
import json
import glob
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DERIV, RES, N_JOBS                    # noqa: E402
from lib import data, models as R                        # noqa: E402

M = R.M
OUT = os.path.join(DERIV, "fits")
os.makedirs(OUT, exist_ok=True)
lg = M._logit
PARS = ["p_common", "sigma_A", "sigma_V", "sigma_P", "mu_P", "kappa", "rho", "eta"]


def fit(name, starts, d, nulls):
    x, f = M._multi(R.nll, starts, (name, d), polish=True)
    for x0 in nulls:
        f0 = R.nll(x0, name, d)
        if f0 < f:
            x, f = x0, f0
    return x


def subject(sid):
    f = os.path.join(OUT, f"sub-{sid:02d}.csv")
    if os.path.exists(f):
        return
    M._set_grid(*R.FIT_GRID)
    d = M.SubjectData(data.load_av().query("subject == @sid"))
    X = {}
    X["static"] = xs = M.fit_static(d, n_starts=8, seed=sid)[0]
    X["learn"] = xl = M.fit_dynamic(d, xs, kappas=(0.01, 0.05, 0.2), rule=0)[0]
    X["markov"] = xm = M.fit_dynamic(d, xs, kappas=(0.95, 0.8, 0.5), rule=1)[0]
    X["rep"] = fit("rep", [np.r_[xs, lg(r)] for r in (0.02, 0.08, 0.25)], d, [np.r_[xs, R.NULL]])
    for tg in ("est", "stim"):
        X[f"num_{tg}"] = fit(f"num_{tg}", [np.r_[xs, lg(e)] for e in (0.02, 0.1, 0.3)], d, [np.r_[xs, R.NULL]])
    xr = X["rep"]
    X["markov_rep"] = fit("markov_rep", [np.r_[xm, lg(r)] for r in (0.02, 0.1)] + [np.r_[xr[:5], xm[5], xr[5]]], d,
                          [np.r_[xm, R.NULL]])
    for tg in ("est", "stim"):
        xn = X[f"num_{tg}"]
        X[f"markov_num_{tg}"] = fit(f"markov_num_{tg}", [np.r_[xm, lg(e)] for e in (0.02, 0.1)] +
                                    [np.r_[xn[:5], xm[5], xn[5]]], d, [np.r_[xm, R.NULL]])
    xn = X["num_est"]
    X["learn_num_est"] = fit("learn_num_est", [np.r_[xl, lg(e)] for e in (0.02, 0.1)] + [np.r_[xn[:5], xl[5], xn[5]]] +
                             [np.r_[xn[:5], lg(k), xn[5]] for k in (0.01, 0.05, 0.2)], d,
                             [np.r_[xl, R.NULL], np.r_[xn[:5], R.NULL, xn[5]]])
    rows = []
    for name, x in X.items():
        p = R.unpack(name, x)
        rows.append(dict(subject=sid, model=name, k=R.K[name], n=int(d.valid.sum()),
                         ll=float(R.evaluate(name, x, d, R.FINE_GRID)[0].sum()),
                         **{k: p[k] for k in PARS}, x=json.dumps(list(map(float, x)))))
    pd.DataFrame(rows).to_csv(f, index=False)
    print(sid, "done", flush=True)


def summary():
    F = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(OUT, "sub-*.csv")))], ignore_index=True)
    F["group"] = F.subject.map(data.group_of())
    F["aic"] = 2 * F.k - 2 * F.ll
    F["dAIC"] = F.subject.map(F[F.model == "static"].set_index("subject").aic) - F.aic     # > 0 favours the model
    F.to_csv(os.path.join(RES, "fits.csv"), index=False)


if __name__ == "__main__":
    ids = [int(s) for s in sys.argv[1:]] or data.ids()
    Parallel(n_jobs=N_JOBS)(delayed(subject)(s) for s in ids)
    summary()
