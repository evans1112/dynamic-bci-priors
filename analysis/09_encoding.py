"""Does prestimulus power track the latent priors of the full model?

Per participant, OLS of band power (per sensor) on the causal prior P(t) and the numerosity prior mu_P(t) (both
z-scored) together with the nuisance regressors (run, trial position, log time since the previous onset, previous RT,
previous numbers of beeps and flashes, previous report); in a second version, + a cubic spline of trial position and
indicators of trials 2-6.  A latent is left out for a participant when it hardly varies (SD of P < .005, of mu_P < .01).
Group level: two-sided cluster permutation test over sensors (5000 sign flips) per band, smallest cluster p corrected
over the four bands (false discovery rate).
Output: results/encoding.csv, results/encoding_unique_variance.csv
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DERIV, RES                           # noqa: E402
from lib import data, eeg, models as R                  # noqa: E402

M = R.M
MODEL = "markov_num_est"


def latents(sid, d):
    x = np.array(json.loads(pd.read_csv(os.path.join(DERIV, "fits", f"sub-{sid:02d}.csv")).set_index("model").loc[MODEL, "x"]))
    _, _, prior, mus, _ = R.evaluate(MODEL, x, d)
    return pd.DataFrame(dict(run=d.df.run.values, runTrialNumber=d.df.runTrialNumber.values, P=prior, mu=mus))


def subject_betas(sid, df):
    F, ch, md = eeg.features(sid)
    md = md.merge(latents(sid, M.SubjectData(df[df.subject == sid])), on=["run", "runTrialNumber"], how="left")
    Y = F.astype(float).reshape(len(F), -1)
    regs = [r for r, lim in (("P", 0.005), ("mu", 0.01)) if md[r].std() > lim]
    out, uv = {}, {}
    for spline in (False, True):
        N = eeg.design(md, [], spline)
        B = eeg.ols(np.column_stack([eeg.zs(md[r]) for r in regs] + [N]), Y)
        for r in ("P", "mu"):
            out[(r, spline)] = B[regs.index(r)].reshape(F.shape[1:]) if r in regs else np.full(F.shape[1:], np.nan)
            if r in regs and not spline:              # share of the latent's variance left after the nuisance
                others = np.column_stack([eeg.zs(md[o]) for o in regs if o != r] + [N])
                e = eeg.zs(md[r]) - others @ np.linalg.lstsq(others, eeg.zs(md[r]), rcond=None)[0]
                uv[r] = float(e.var())
    return ch, out, uv


def main():
    df = data.load_av()
    rows, uvs, adj = [], [], None
    for g in ("HC", "SCZ"):
        ids = data.ids(g)
        per = [subject_betas(s, df) for s in ids]
        adj = adj if adj is not None else eeg.adjacency(per[0][0])
        uvs += [dict(subject=s, group=g, **{f"unique_var_{k}": v for k, v in o[2].items()}) for s, o in zip(ids, per)]
        for r in ("P", "mu"):
            for spline in (False, True):
                B = np.stack([o[(r, spline)] for _, o, _ in per])
                B = B[~np.isnan(B).any(axis=(1, 2))]
                raw = []
                for bi in range(len(eeg.BANDS)):
                    _, cl = eeg.cluster_1samp(B[:, :, bi], adj)
                    raw.append(min((p for _, p in cl), default=1.0))
                fdr = multipletests(raw, method="fdr_bh")[1]
                for bi, band in enumerate(eeg.BANDS):
                    rows.append(dict(group=g, latent=r, nuisance="spline" if spline else "linear", band=band, n=len(B),
                                     p_cluster=raw[bi], p_fdr=fdr[bi]))
                print(g, r, "spline" if spline else "linear", "done", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(RES, "encoding.csv"), index=False)
    pd.DataFrame(uvs).to_csv(os.path.join(RES, "encoding_unique_variance.csv"), index=False)
    print(T.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
