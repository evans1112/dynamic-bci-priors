"""Pull of the report towards the previous report (S2 Fig).

Residual report = report minus the participant's mean report in the same condition (task x nA x nV).  Pairs of
consecutive trials of a run, both answered.  Per participant and task: (a) mean residual by previous report;
(b) OLS of the residual on the previous report, the previous task-relevant and the previous task-irrelevant number
(centred); (c) slope on the report k trials back alone, k = 1-5.
Output: results/pull_curve.csv, pull_betas.csv, pull_lags.csv, pull_tests.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RES                                  # noqa: E402
from lib import data                                    # noqa: E402

TASK = {1: "auditory", 2: "visual"}


def pairs(d, lag):
    d = d.sort_values(["run", "runTrialNumber"]).copy()
    d["R"] = d.countingReport
    d["rel"] = np.where(d.task == 1, d.nA, d.nV); d["irr"] = np.where(d.task == 1, d.nV, d.nA)
    ok = d.R > 0
    d.loc[ok, "res"] = d[ok].R - d[ok].groupby(["task", "nA", "nV"]).R.transform("mean")
    g = d.groupby("run")
    for c in ("R", "rel", "irr"):
        d[f"p_{c}"] = g[c].shift(lag)
    return d[(d.R > 0) & (d.p_R > 0)]


def main():
    df = data.load_av()
    grp = data.group_of()
    curve, betas, lags = [], [], []
    for s in data.ids():
        for lag in range(1, 6):
            for tk, x in pairs(df[df.subject == s], lag).groupby("task"):
                xc = x[["p_R", "p_rel", "p_irr"]] - x[["p_R", "p_rel", "p_irr"]].mean()
                lags.append(dict(subject=s, group=grp[s], task=tk, lag=lag, slope=np.polyfit(xc.p_R, x.res, 1)[0]))
                if lag == 1:
                    co = np.linalg.lstsq(np.column_stack([np.ones(len(x)), xc.values]), x.res.values, rcond=None)[0]
                    betas.append(dict(subject=s, group=grp[s], task=tk, prev_report=co[1], prev_relevant=co[2],
                                      prev_irrelevant=co[3]))
                    for v, y in x.groupby("p_R").res:
                        curve.append(dict(subject=s, group=grp[s], task=tk, prev_report=int(v), res=y.mean()))
    B, L = pd.DataFrame(betas), pd.DataFrame(lags)
    pd.DataFrame(curve).to_csv(os.path.join(RES, "pull_curve.csv"), index=False)
    B.to_csv(os.path.join(RES, "pull_betas.csv"), index=False)
    L.to_csv(os.path.join(RES, "pull_lags.csv"), index=False)
    rows = []
    for (g, tk), x in B.groupby(["group", "task"]):
        for col in ("prev_report", "prev_relevant", "prev_irrelevant"):
            t, p = stats.ttest_1samp(x[col], 0)
            rows.append(dict(group=g, task=TASK[tk], measure=col, mean=x[col].mean(), sem=stats.sem(x[col]), t=t, p=p))
    for (g, tk, lag), x in L.groupby(["group", "task", "lag"]):
        t, p = stats.ttest_1samp(x.slope, 0)
        rows.append(dict(group=g, task=TASK[tk], measure=f"slope, lag {lag}", mean=x.slope.mean(), sem=stats.sem(x.slope),
                         t=t, p=p))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(RES, "pull_tests.csv"), index=False)
    print(T.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
