"""Prestimulus EEG: time-frequency settings, trial selection, regression design and cluster statistics."""
import os
import numpy as np
import pandas as pd
from scipy import stats
from config import DERIV

FREQS = np.logspace(np.log10(3), np.log10(80), 26)
NCYC = np.clip(0.4 * FREQS, 3, 10)
DECIM = 5                                   # 250 Hz -> 50 Hz
T_ANALYSIS = -1.0
BANDS = {"theta": (4, 7.5), "alpha": (8, 12.5), "beta": (13, 30), "gamma": (31, 80)}
WIN = (-0.6, -0.2)
OCC = ["O1", "Oz", "O2", "PO3", "POz", "PO4", "PO7", "PO8"]      # occipital sensors of the decile analysis [Rohe 2019]
NUIS = ["trial_in_run", "log_gap", "prev_countingReportRT", "prev_nA", "prev_nV", "prev_countingReport"]
FEAT = os.path.join(DERIV, "features")


def valid_mask(times):
    """(freq, time) bins whose wavelet (+-3 SD) lies within -1.5..0 s, from -1 s on."""
    sig = NCYC / (2 * np.pi * FREQS)
    return ((times[None, :] + 3 * sig[:, None] <= 0) & (times[None, :] - 3 * sig[:, None] >= -1.5)
            & (times[None, :] >= T_ANALYSIS - 1e-9))


def load_trials(si):
    """Clean epochs whose previous trial (same run) was answered."""
    import mne
    ep = mne.read_epochs(os.path.join(DERIV, "epochs", f"sub-{si:02d}-epo.fif"), preload=True)
    md = ep.metadata.reset_index(drop=True)
    keep = (~md.artifact & (md.trial_in_run > 1) & (md.prev_trial_gap == 1)
            & md.prev_countingReport.notna() & md.prev_onset_gap.notna()).values
    ep = ep[np.where(keep)[0]]
    md = md[keep].reset_index(drop=True)
    md["log_gap"] = np.log(md.prev_onset_gap)
    rt = md.prev_countingReportRT.where(md.prev_countingReportRT < 10)     # 999 = no response in time
    md["prev_countingReportRT"] = rt.fillna(rt.median())
    return ep, md


def features(si):
    """Band power per trial [trial, channel, band], channel names and trial metadata (03_eeg_features.py)."""
    z = np.load(os.path.join(FEAT, f"sub-{si:02d}.npz"))
    return z["F"], list(z["ch_names"]), pd.read_csv(os.path.join(FEAT, f"sub-{si:02d}.csv"))


def alpha_z(si, d, chans=OCC):
    """Alpha power averaged over chans, z-scored within run, for the trials of d (0 where there is no clean EEG)."""
    F, ch, md = features(si)
    md["a"] = F[:, [ch.index(c) for c in chans], 1].mean(1)
    x = d.df[["run", "runTrialNumber"]].merge(md[["run", "runTrialNumber", "a"]], on=["run", "runTrialNumber"], how="left")
    g = x.groupby("run").a
    return ((x.a - g.transform("mean")) / g.transform("std")).fillna(0.0).values


def zs(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else np.full_like(x, np.nan)


def design(md, regs, spline=False):
    """Regressors of interest + nuisance (z-scored) + run dummies; spline: + cubic spline of trial position and
    indicators of trials 2-6."""
    X = [zs(md[r]) for r in regs] + [zs(md[r]) for r in NUIS]
    runs = pd.get_dummies(md.run.astype(int)).values.astype(float)
    X = np.column_stack(X + [runs])
    if spline:
        from sklearn.preprocessing import SplineTransformer
        tb = SplineTransformer(n_knots=8, degree=3).fit_transform(md[["trial_in_run"]].values.astype(float))
        early = np.column_stack([(md.trial_in_run.values == k).astype(float) for k in range(2, 7)])
        X = np.column_stack([X, tb[:, 1:], early])
    return X


def ols(X, Y):
    XtX_inv = np.linalg.pinv(X.T @ X)
    return XtX_inv @ X.T @ Y


def adjacency(ch):
    import mne
    from scipy import sparse
    info = mne.create_info(ch, 250.0, "eeg")
    info.set_montage("standard_1005")
    return sparse.csr_matrix(mne.channels.find_ch_adjacency(info, "eeg")[0])


def cluster_1samp(V, adj, n_perm=5000):
    """V [participants, sensors]: two-sided one-sample cluster permutation test (cluster-forming p < .05)."""
    import mne
    T, cl, pv, _ = mne.stats.permutation_cluster_1samp_test(
        V, threshold=stats.t.ppf(0.975, len(V) - 1), n_permutations=n_perm, adjacency=adj, tail=0, seed=1,
        out_type="mask")
    return T, list(zip(cl, pv))


def tvals(X):
    """One-sample t over axis 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nan_to_num(X.mean(0) / (X.std(0, ddof=1) / np.sqrt(len(X))))
