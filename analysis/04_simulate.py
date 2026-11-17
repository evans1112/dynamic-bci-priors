"""Behavioural signatures in the data and in experiments simulated from each fitted model.

For every participant and model, N_SIM experiments are simulated with the participant's own trial sequence (missed
trials kept missed; the updates use the simulated observer's own posterior and report).  Measures, computed in the
same way for the data and for each simulated experiment:
  history       history effect of Rohe et al. (2019), lags 1-5 (lib/history.py)
  strict        previous report held constant: within cells of task x nA x nV x previous report x previous
                task-relevant number, incongruent current trials, y = (report - relevant number) x sign(irrelevant -
                relevant); beta = effect of a small (<= 1) vs large (>= 2) previous disparity (cell fixed effects)
  strict_ctrl   the same with the previous task-irrelevant number as a covariate
  loose         trials on which the Markov causal prior and the pull towards the previous report predict opposite
                shifts: residual report (report minus the participant's mean in the same condition) signed along the
                direction predicted by the causal prior
  attract       slope of the residual report on the previous report, per task (1 auditory, 2 visual)
Output: derivatives/sims/sub-XX.csv (source = data or model, sim, measure, task, lag, value)
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DERIV, N_JOBS                         # noqa: E402
from lib import data, history, models as R               # noqa: E402

M = R.M
OUT = os.path.join(DERIV, "sims")
os.makedirs(OUT, exist_ok=True)
N_SIM = int(os.environ.get("N_SIM", 30))
SEED = {"static": 0, "learn": 1, "markov": 2, "rep": 3, "num_est": 4, "markov_num_est": 7, "learn_num_est": 9}


def signatures(d, resp, x_static):
    out = []
    prev = np.r_[0, resp[:-1]]; prev[d.run_start] = 0
    pnA = np.r_[0, d.nA[:-1]]; pnV = np.r_[0, d.nV[:-1]]
    ok = (resp > 0) & (prev > 0)
    rel = np.where(d.task == 1, d.nA, d.nV); irr = np.where(d.task == 1, d.nV, d.nA)
    prel = np.where(d.task == 1, pnA, pnV); pirr = np.where(d.task == 1, pnV, pnA)
    # previous report held constant
    dint = np.sign(irr - rel)
    m = ok & (dint != 0)
    T = pd.DataFrame(dict(y=(resp - rel)[m] * dint[m], small=(np.abs(pnA - pnV)[m] <= 1).astype(float), pirr=pirr[m],
                          cell=[f"{a}_{b}_{c}_{e}_{f}" for a, b, c, e, f in zip(d.task[m], d.nA[m], d.nV[m], prev[m], prel[m])]))
    Td = T[["y", "small", "pirr"]] - T.groupby("cell")[["y", "small", "pirr"]].transform("mean")
    beta = (Td.small * Td.y).sum() / (Td.small ** 2).sum()
    beta_c = np.linalg.lstsq(Td[["small", "pirr"]].values, Td.y.values, rcond=None)[0][0]
    out += [dict(measure="strict", task=0, lag=1, value=beta), dict(measure="strict_ctrl", task=0, lag=1, value=beta_c)]
    # residual report
    v = resp > 0
    D = pd.DataFrame(dict(r=resp, task=d.task, cond=d.cond))
    res = np.full(len(resp), np.nan)
    res[v] = resp[v] - D[v].groupby(["task", "cond"]).r.transform("mean").values
    # opposite predictions
    d_upd = np.where(np.abs(pnA - pnV) <= 1, 1, -1) * dint
    d_prev = np.sign(prev - rel)
    mL = ok & (dint != 0) & (d_prev != 0) & (d_upd == -d_prev)
    out.append(dict(measure="loose", task=0, lag=1, value=np.mean(res[mL] * d_upd[mL])))
    # pull towards the previous report
    for tk in (1, 2):
        mm = ok & (d.task == tk)
        out.append(dict(measure="attract", task=tk, lag=1, value=np.polyfit(prev[mm] - prev[mm].mean(), res[mm], 1)[0]))
    out += [dict(measure="history", task=0, **r) for r in history.effect(d, resp, x_static)]
    return out


def subject(sid):
    f = os.path.join(OUT, f"sub-{sid:02d}.csv")
    if os.path.exists(f):
        return
    M._set_grid(*R.FIT_GRID)
    d = M.SubjectData(data.load_av().query("subject == @sid"))
    fits = pd.read_csv(os.path.join(DERIV, "fits", f"sub-{sid:02d}.csv")).set_index("model")
    X = {m: np.array(json.loads(fits.loc[m, "x"])) for m in list(SEED)}
    rows = [dict(subject=sid, source="data", sim=0, **o) for o in signatures(d, d.resp, X["static"])]
    for m, j in SEED.items():
        for s in range(N_SIM):
            resp = R.sim(m, X[m], d, 7919 * sid + 1000 * j + s)
            resp = np.where(d.resp > 0, resp, 0)
            rows += [dict(subject=sid, source=m, sim=s, **o) for o in signatures(d, resp, X["static"])]
    pd.DataFrame(rows).to_csv(f, index=False)
    print(sid, "done", flush=True)


if __name__ == "__main__":
    ids = [int(s) for s in sys.argv[1:]] or data.ids()
    Parallel(n_jobs=N_JOBS)(delayed(subject)(s) for s in ids)
