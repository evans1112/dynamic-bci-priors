"""Tables S1-S3 and the other numbers reported in the text.

Output: results/table_S1.csv (model comparison), table_S2.csv (history effect and conditional tests, data vs
simulations), table_S3.csv (brain-state models), in_text.csv (remaining numbers, with a short description)
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DERIV, RES                           # noqa: E402
from lib import data                                    # noqa: E402

GROUPS = ("HC", "SCZ")
LABEL = {"learn": "Delta-rule causal prior", "markov": "Markov causal prior", "rep": "Response repetition",
         "num_est": "Dynamic numerosity prior (report)", "num_stim": "Dynamic numerosity prior (stimuli)",
         "markov_rep": "Markov causal prior + response repetition",
         "markov_num_stim": "Markov causal prior + dynamic numerosity prior (stimuli)",
         "learn_num_est": "Delta-rule causal prior + dynamic numerosity prior (report)",
         "markov_num_est": "Full model: Markov causal prior + dynamic numerosity prior (report)"}
SIM_MODELS = ["static", "rep", "num_est", "markov", "markov_num_est"]
text = []


def note(key, value, what):
    text.append(dict(key=key, value=value, description=what))


def model_comparison():
    F = pd.read_csv(os.path.join(RES, "fits.csv"))
    st = F[F.model == "static"].set_index("subject")
    rows = []
    for m, lab in LABEL.items():
        row = {"model": lab, "k": int(F[F.model == m].k.iloc[0])}
        for g in GROUPS:
            v = F[(F.group == g) & (F.model == m)].set_index("subject")
            row.update({f"{g}_sum_dLL": (v.ll - st.ll.reindex(v.index)).sum(), f"{g}_sum_dAIC": v.dAIC.sum(),
                        f"{g}_n_favour": f"{int((v.dAIC > 0).sum())}/{len(v)}", f"{g}_p": stats.wilcoxon(v.dAIC).pvalue})
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "table_S1.csv"), index=False)

    P = F.pivot_table(index=["group", "subject"], columns="model", values="aic")
    for g in GROUPS:
        for a, b in (("static", "markov"), ("rep", "markov_rep"), ("num_est", "markov_num_est"),
                     ("num_stim", "markov_num_stim"), ("learn_num_est", "markov_num_est")):
            dv = P.loc[g][a] - P.loc[g][b]
            note(f"{g}: {a} -> {b}", f"sum dAIC {dv.sum():.1f}, Wilcoxon p {stats.wilcoxon(dv).pvalue:.3f}",
                 "dAIC of the second model relative to the first (> 0 favours the second)")
    full = F[F.model == "markov_num_est"].merge(data.subjects()[["subject", "panss_pos"]], on="subject")
    for par in ("kappa", "eta"):
        x = [full[full.group == g][par] for g in GROUPS]
        note(f"full model {par}", f"median {x[0].median():.2f} / {x[1].median():.2f}, Mann-Whitney p "
             f"{stats.mannwhitneyu(*x).pvalue:.2f}", "controls / patients")
        sz = full[full.group == "SCZ"]
        r, p = stats.spearmanr(sz[par], sz.panss_pos)
        note(f"full model {par} vs PANSS positive", f"rho {r:.2f}, p {p:.2f}", "patients, Spearman")


def signatures():
    S = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(DERIV, "sims", "sub-*.csv"))])
    S["group"] = S.subject.map(data.group_of())
    S = S[(S.measure != "history") | (S.lag == 1)]
    P = S.groupby(["group", "subject", "source", "measure", "task"]).value.mean().reset_index()
    meas = [("history", 0, "History effect, lag 1"), ("strict", 0, "Previous report held constant (beta)"),
            ("strict_ctrl", 0, "... + previous irrelevant number"), ("loose", 0, "Opposite-prediction sequences"),
            ("attract", 2, "Pull towards previous report, visual"), ("attract", 1, "Pull towards previous report, auditory")]
    rows = []
    for g in GROUPS:
        for ms, tk, lab in meas:
            x = P[(P.group == g) & (P.measure == ms) & (P.task == tk)]
            dd = x[x.source == "data"].set_index("subject").value
            t, p = stats.ttest_1samp(dd, 0)
            row = dict(group=g, measure=lab, data=dd.mean(), data_t=t, data_p=p)
            for m in SIM_MODELS:
                mm = x[x.source == m].set_index("subject").value.reindex(dd.index)
                row[m] = mm.mean(); row[f"{m}_p_vs_data"] = stats.ttest_rel(dd, mm).pvalue
            rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "table_S2.csv"), index=False)


def brain_state():
    A = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(DERIV, "brain_state", "sub-*.csv"))])
    N = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(DERIV, "brain_state_num", "sub-*.csv"))])
    for D in (A, N):
        D["group"] = D.subject.map(data.group_of())
        D.loc[D.p_common < 1e-4, "b"] = np.nan                    # b is not identified when the causal prior is ~0
    spec = [("Markov causal prior", "causal prior (b)", "b", A[(A.base == "markov") & ~A.position]),
            ("Full model", "causal prior (b)", "b", A[(A.base == "markov_num_est") & ~A.position]),
            ("Full model", "numerosity prior (c)", "c", N[N.state == "c"]),
            ("Full model", "both: causal prior (b)", "b", N[N.state == "bc"]),
            ("Full model", "both: numerosity prior (c)", "c", N[N.state == "bc"])]
    rows = []
    for model, shift, prm, D in spec:
        row = dict(model=model, alpha_shifts=shift)
        for g in GROUPS:
            v = D[D.group == g][prm].dropna()
            row.update({f"{g}_n": len(v), f"{g}_mean": v.mean(), f"{g}_p_t": stats.ttest_1samp(v, 0).pvalue,
                        f"{g}_p_wilcoxon": stats.wilcoxon(v).pvalue})
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "table_S3.csv"), index=False)

    full = A[A.base == "markov_num_est"]
    hc, sz = (full[(full.group == g) & ~full.position].b.dropna() for g in GROUPS)
    note("alpha weight, controls vs patients", f"t test p {stats.ttest_ind(hc, sz).pvalue:.2f}", "full model")
    note("alpha weight, controls, negative", f"{int((hc < 0).sum())}/{len(hc)}", "full model")
    pos = full[(full.group == "HC") & full.position].b.dropna()
    note("alpha weight with trial position", f"b {pos.mean():.2f}, p {stats.ttest_1samp(pos, 0).pvalue:.2f}",
         "controls, full model, trial position as a second shift of the logit prior")
    h5 = full[(full.group == "HC") & ~full.position & (full.subject != 5)].b.dropna()
    note("alpha weight without participant 5", f"b {h5.mean():.2f}, p {stats.ttest_1samp(h5, 0).pvalue:.3f}, "
         f"Wilcoxon p {stats.wilcoxon(h5).pvalue:.3f}", "controls, full model; participant 5: p_common ~ 0.001")
    x = full[~full.position]
    for g in GROUPS:
        dll = (x.ll - x.ll0)[x.group == g]
        note(f"{g}: fit gain from alpha", f"sum dLL {dll.sum():.1f}, pooled LR chi2({len(dll)}) = {2 * dll.sum():.1f}, "
             f"p {stats.chi2.sf(2 * dll.sum(), len(dll)):.3f}", "full model with vs without b (one parameter each)")


def sample():
    df = data.load_av()
    for g in GROUPS:
        x = df[df.subject.isin(data.ids(g))]
        runs = x.groupby("subject").run.nunique()
        n = x[x.countingReport > 0].groupby("subject").size()
        note(f"{g}: participants", f"{len(runs)}, audiovisual runs {runs.min()}-{runs.max()}, answered trials mean "
             f"{n.mean():.0f} (range {n.min()}-{n.max()})", "")


if __name__ == "__main__":
    sample()
    model_comparison()
    signatures()
    brain_state()
    T = pd.DataFrame(text)
    T.to_csv(os.path.join(RES, "in_text.csv"), index=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 120)
    print(T.to_string(index=False))
