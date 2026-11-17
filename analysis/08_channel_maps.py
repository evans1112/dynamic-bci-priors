"""Scalp maps of the alpha weight b in the full model (Fig 1D; descriptive, inference is on the occipital sensors).

Per participant and sensor, alpha power z-scored within run shifts the logit causal prior by b; b is fitted with all
other parameters fixed at the participant's occipital solution (07_brain_state.py).  The same fit on the occipital
average is included as a check (it reproduces the occipital estimate).
Output: derivatives/channel_b/sub-XX.csv (subject, feature, b, p_common)
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DERIV, N_JOBS                         # noqa: E402
from lib import data, eeg, models as R, state as RS      # noqa: E402

M = R.M
OUT = os.path.join(DERIV, "channel_b")
os.makedirs(OUT, exist_ok=True)
BASE = "markov_num_est"


def roi_solution(sid):
    full = np.array(json.loads(pd.read_csv(os.path.join(DERIV, "fits", f"sub-{sid:02d}.csv")).set_index("model").loc[BASE, "x"]))
    s = pd.read_csv(os.path.join(DERIV, "brain_state", f"sub-{sid:02d}.csv"))
    s = s[(s.base == BASE) & (~s.position)].iloc[0]
    full[0] = M._logit(np.clip(s.p_common, 1e-12, 1 - 1e-12))
    full[4] = M._logit(np.clip((s.mu_P - M.MU_MIN) / (M.MU_MAX - M.MU_MIN), 1e-12, 1 - 1e-12))
    full[5] = M._logit(np.clip(s.kappa, 1e-12, 1 - 1e-12))
    full[6] = M._logit(np.clip(s.eta, 1e-12, 1 - 1e-12))
    return full, s.p_common


def subject(sid):
    f = os.path.join(OUT, f"sub-{sid:02d}.csv")
    if os.path.exists(f):
        return
    M._set_grid(*R.FIT_GRID)
    d = M.SubjectData(data.load_av().query("subject == @sid"))
    full, pc = roi_solution(sid)
    p = R.unpack(BASE, full[:R.K[BASE]])
    arrs, mI, tail, wO, mu_lo = R.tables(p, True)
    _, ch, _ = eeg.features(sid)

    def nll(b, z):
        ll = RS.run_state(*arrs, mI, tail, wO, mu_lo, R.MU_STEP, d.cond, d.task, d.resp, d.nA, d.nV, d.run_start,
                          p["p_common"], p["kappa"], p["rule"], p["mu_P"], p["eta"], p["target"], p["rho"], M.LAPSE,
                          b * z, np.zeros(len(z)))
        return -float(ll.sum())
    rows = []
    for c in ch + ["occ_roi"]:
        z = eeg.alpha_z(sid, d, eeg.OCC if c == "occ_roi" else [c])
        r = minimize_scalar(nll, bounds=(-3, 3), args=(z,), method="bounded", options=dict(xatol=1e-3))
        rows.append(dict(subject=sid, feature=c, b=r.x, p_common=pc))
    pd.DataFrame(rows).to_csv(f, index=False)
    print(sid, "done", flush=True)


if __name__ == "__main__":
    ids = [int(s) for s in sys.argv[1:]] or data.ids()
    Parallel(n_jobs=N_JOBS)(delayed(subject)(s) for s in ids)
