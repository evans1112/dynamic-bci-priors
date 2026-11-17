"""EEG: match the stimulus markers of every audiovisual run to the behavioural trials, then preprocess.

Alignment: runs whose marker sequence (nA, nV) equals the behavioural sequence are taken as they are; otherwise the two
sequences are aligned with difflib, keeping matching blocks of >= 3 trials.
Preprocessing (per participant, runs concatenated): notch 50/100 Hz, band-pass 0.1-100 Hz, resampling to 250 Hz;
channels with a robust z-score of log variance > 4 interpolated; FCz added back, average reference; FastICA
(30 components) on 1-40 Hz data, components with |r| > 0.5 to the vertical or horizontal eye proxy removed; epochs
-1.5 to 1.0 s, flagged as artefact if the peak-to-peak amplitude in -1.0..0.8 s exceeds 150 uV.
Output: derivatives/epochs/sub-XX-epo.fif, results/eeg_alignment.csv, results/eeg_preprocessing.csv
"""
import os
import sys
import difflib
import numpy as np
import pandas as pd
import mne
from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DERIV, RES                           # noqa: E402
from lib import data                                    # noqa: E402

mne.set_log_level("ERROR")
SFREQ = 250.0
TMIN, TMAX = -1.5, 1.0
REJ_WIN, REJ_PTP = (-1.0, 0.8), 150e-6
EPO = os.path.join(DERIV, "epochs")
os.makedirs(EPO, exist_ok=True)


def behaviour():
    """Audiovisual trials with the previous trial of the same run."""
    df = data.load_av()
    df["trial_in_run"] = df.groupby(["subject", "run"]).cumcount() + 1
    g = df.groupby(["subject", "run"])
    for c in ["nA", "nV", "countingReport", "countingReportRT"]:
        df["prev_" + c] = g[c].shift(1)
    df["prev_absdisp"] = (df.prev_nA - df.prev_nV).abs()
    df["prev_trial_gap"] = df.runTrialNumber - g.runTrialNumber.shift(1)
    return df[["subject", "run", "runTrialNumber", "trial_in_run", "task", "nA", "nV", "countingReport",
               "countingReportRT", "prev_nA", "prev_nV", "prev_countingReport", "prev_countingReportRT",
               "prev_absdisp", "prev_trial_gap"]]


def align(beh):
    rows, report = [], []
    subs = data.subjects()
    for _, s in subs[subs.subject.isin(data.ids())].iterrows():
        si, eid = int(s.subject), s.eeg_id
        for vhdr, task, run in data.av_runs(eid):
            b = beh[(beh.subject == si) & (beh.run == run)].sort_values("runTrialNumber")
            mk = data.read_markers(vhdr)
            if len(mk) == 0 or len(b) == 0 or not (b.task.nunique() == 1 and (mk.task == b.task.iloc[0]).all()):
                report.append(dict(subject=si, run=run, n_markers=len(mk), n_behaviour=len(b), n_matched=0, note="excluded"))
                continue
            seq_b, seq_m = list(zip(b.nA, b.nV)), list(zip(mk.nA, mk.nV))
            if seq_b == seq_m:
                pairs, note = list(zip(range(len(b)), range(len(mk)))), "exact"
            else:
                sm = difflib.SequenceMatcher(a=seq_b, b=seq_m, autojunk=False)
                pairs = [(blk.a + k, blk.b + k) for blk in sm.get_matching_blocks() if blk.size >= 3 for k in range(blk.size)]
                note = "aligned"
            rows += [dict(subject=si, run=run, runTrialNumber=int(b.runTrialNumber.iloc[ib]), vhdr=vhdr,
                          sample=int(mk["sample"].iloc[im])) for ib, im in pairs]
            report.append(dict(subject=si, run=run, n_markers=len(mk), n_behaviour=len(b), n_matched=len(pairs), note=note))
    pd.DataFrame(report).to_csv(os.path.join(RES, "eeg_alignment.csv"), index=False)
    return pd.DataFrame(rows)


def preprocess(si, al, beh):
    mne.set_log_level("ERROR")
    raws, events_all, meta_all, offset = [], [], [], 0
    for (run, vhdr), a in al.groupby(["run", "vhdr"]):
        raw = mne.io.read_raw_brainvision(data.fixed_vhdr(vhdr, os.path.join(DERIV, "vhdr_fixed")), preload=True)
        raw.set_montage("standard_1005")
        raw.notch_filter([50, 100], n_jobs=1)
        raw.filter(0.1, 100.0, n_jobs=1)
        sf0 = raw.info["sfreq"]
        a = a.sort_values("runTrialNumber")
        ev = np.c_[a["sample"].values, np.zeros(len(a), int), a.runTrialNumber.values]
        raw, ev = raw.resample(SFREQ, events=ev)
        raws.append(raw)
        ev[:, 0] += offset
        offset += raw.n_times
        events_all.append(ev)
        m = beh[beh.run == run].set_index("runTrialNumber").loc[a.runTrialNumber.values].reset_index()
        gap = np.r_[np.nan, np.diff(a["sample"].values / sf0)]
        gap[np.r_[True, np.diff(a.runTrialNumber.values) != 1]] = np.nan
        m["prev_onset_gap"] = gap
        meta_all.append(m)
    raw = mne.io.RawArray(np.concatenate([r.get_data() for r in raws], axis=1), raws[0].info)
    raw.set_montage("standard_1005")
    events, meta = np.vstack(events_all), pd.concat(meta_all, ignore_index=True)

    x = raw.copy().filter(1, 40, n_jobs=1).get_data()
    lv = np.log(np.var(x, axis=1) + 1e-30)
    z = (lv - np.median(lv)) / (1.4826 * np.median(np.abs(lv - np.median(lv))))
    bads = [raw.ch_names[i] for i in np.where((z > 4) | (np.var(x, axis=1) < 1e-14))[0]]
    raw.info["bads"] = bads
    if bads:
        raw.interpolate_bads(reset_bads=True)
    raw = mne.add_reference_channels(raw, "FCz")
    raw.set_montage("standard_1005")
    raw.set_eeg_reference("average", projection=False)

    rf = raw.copy().filter(1.0, 40.0, n_jobs=1)
    ica = mne.preprocessing.ICA(n_components=30, method="fastica", random_state=0, max_iter=1000)
    ica.fit(rf, decim=2)
    src = ica.get_sources(rf).get_data()
    d = rf.get_data(picks=["Fp1", "Fp2", "F7", "F8"])
    veog, heog = (d[0] + d[1]) / 2, d[2] - d[3]
    r_v = np.array([abs(np.corrcoef(s, veog)[0, 1]) for s in src])
    r_h = np.array([abs(np.corrcoef(s, heog)[0, 1]) for s in src])
    ica.exclude = [int(i) for i in sorted(set(np.where(r_v > 0.5)[0]) | set(np.where(r_h > 0.5)[0]))]
    ica.apply(raw)

    epochs = mne.Epochs(raw, events, tmin=TMIN, tmax=TMAX, baseline=None, metadata=meta, preload=True,
                        reject=None, reject_by_annotation=False, event_repeated="drop")
    seg = epochs.copy().crop(*REJ_WIN).get_data()
    ptp = (seg.max(axis=2) - seg.min(axis=2)).max(axis=1)
    epochs.metadata["artifact"] = ptp > REJ_PTP
    epochs.save(os.path.join(EPO, f"sub-{si:02d}-epo.fif"), overwrite=True, fmt="single")
    return dict(subject=si, n_epochs=len(epochs), n_artifact=int((ptp > REJ_PTP).sum()), bad_channels=" ".join(bads),
                ica_excluded=" ".join(map(str, ica.exclude)))


if __name__ == "__main__":
    beh = behaviour()
    al = align(beh)
    subs = [int(s) for s in sys.argv[1:]] or sorted(al.subject.unique())
    out = Parallel(n_jobs=int(os.environ.get("EEG_JOBS", "3")))(
        delayed(preprocess)(s, al[al.subject == s], beh[beh.subject == s]) for s in subs)
    pd.DataFrame(out).to_csv(os.path.join(RES, "eeg_preprocessing.csv"), index=False)
