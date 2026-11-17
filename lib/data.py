"""Behavioural data, participant table and EEG files of Rohe et al. (2024).

The 2024 file differs from the 2019 one in two ways that matter here: runTrialNumber counts trials over the whole
session, and trials without a response were deleted.  Both are undone in load_all(), so that the dynamic models see
the trial sequence of the experiment (a missed trial still updates the priors from its stimuli).
"""
import os
import re
import glob
import functools
import numpy as np
import pandas as pd
import scipy.io as sio
from config import DATA, DATA_2019, RES

GROUPS = {1: "HC", 2: "SCZ", 3: "SZA"}


def _find(pattern):
    hits = sorted(glob.glob(os.path.join(DATA, "**", pattern), recursive=True))
    if not hits:
        raise FileNotFoundError(f"{pattern} not found under {DATA}")
    return hits[0]


def _struct_to_df(f):
    m = sio.loadmat(f, squeeze_me=True, struct_as_record=False)
    d = m[[k for k in m if not k.startswith("_")][0]]
    return pd.DataFrame({k: np.asarray(getattr(d, k)) for k in d._fieldnames})


def _load_2024():
    """Trials deleted from the 2024 file are restored from the EEG marker file of the run (stimuli and task of every
    trial), if it has all 160 trials and agrees with every behavioural row of the run; for two control runs with
    incomplete marker files, from the 2019 file (the control data of the two studies are identical)."""
    cache = os.path.join(RES, "behaviour_restored.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    df = _struct_to_df(_find("BehavioralData.mat")).rename(columns={
        "subj": "subject", "subjID": "eeg_id", "numericReport": "countingReport", "numericReportRT": "countingReportRT"})
    df["countingReport"] = df.countingReport.astype(float)
    df["trial_session"] = df.runTrialNumber.astype(int)
    df["runTrialNumber"] = df.trial_session - 160 * (df.run.astype(int) - 1)
    assert df.runTrialNumber.between(1, 160).all()
    old = _struct_to_df(DATA_2019)
    old = old[old.modality == 1]
    add, report = [], []
    for (s, run), g in df[df.modality == 1].groupby(["subject", "run"]):
        if len(g) >= 160:
            continue
        f = [v for v, t, r in av_runs(g.eeg_id.iloc[0]) if r == run]
        mk = read_markers(f[0]) if len(f) == 1 else pd.DataFrame()
        idx = g.runTrialNumber.values.astype(int) - 1
        ok = (len(mk) == 160 and (mk.nA.values[idx] == g.nA.values).all() and (mk.nV.values[idx] == g.nV.values).all()
              and (mk.task.values[idx] == g.task.values).all())
        src = "markers" if ok else ""
        if not ok and g.group.iloc[0] == 1:
            o = old[(old.subject == s) & (old.run == run)].set_index("runTrialNumber")
            pres = g.set_index("runTrialNumber")
            if len(o) == 160 and (o.loc[pres.index, ["nA", "nV", "task"]].values == pres[["nA", "nV", "task"]].values).all():
                mk = pd.DataFrame(dict(nA=o.nA.values, nV=o.nV.values, task=o.task.values)).astype(int)
                ok, src = True, "2019 file"
        miss = sorted(set(range(1, 161)) - set(g.runTrialNumber.astype(int)))
        report.append(dict(subject=s, run=run, n_rows=len(g), n_missing=len(miss), restored=ok, source=src))
        if ok:
            for r in miss:
                add.append(dict(subject=s, eeg_id=g.eeg_id.iloc[0], group=g.group.iloc[0], modality=1,
                                task=mk.task.iloc[r - 1], nA=mk.nA.iloc[r - 1], nV=mk.nV.iloc[r - 1], run=run,
                                runTrialNumber=r, trial_session=160 * (run - 1) + r,
                                countingReport=np.nan, countingReportRT=np.nan))
    df = pd.concat([df, pd.DataFrame(add)], ignore_index=True)
    df = df.sort_values(["subject", "modality", "run", "runTrialNumber"]).reset_index(drop=True)
    pd.DataFrame(report).to_csv(os.path.join(RES, "restored_trials.csv"), index=False)
    df.to_csv(cache, index=False)
    return df


def load_av():
    """Audiovisual trials of all participants, in presentation order."""
    df = _load_2024()
    for c in ["modality", "task", "run", "subject", "runTrialNumber"]:
        df[c] = df[c].astype(int)
    df = df[df.modality == 1].copy()
    df["nA"] = df.nA.astype(int); df["nV"] = df.nV.astype(int)
    return df.sort_values(["subject", "run", "runTrialNumber"]).reset_index(drop=True)


@functools.lru_cache(maxsize=None)
def subjects():
    """subject, group (HC / SCZ / SZA), eeg_id, age, sex, PANSS scores."""
    beh = _load_2024().groupby("subject").agg(group=("group", "first"), eeg_id=("eeg_id", "first")).reset_index()
    beh["group"] = beh.group.map(GROUPS)
    d = sio.loadmat(_find("DemographicalClinicalData.mat"), squeeze_me=True, struct_as_record=False)["DemoClinicalData"]
    cl = pd.DataFrame(dict(eeg_id=np.asarray(d.participantID), age=np.asarray(d.age), sex=np.asarray(d.sex),
                           panss_pos=np.asarray(d.PANSSpositive), panss_neg=np.asarray(d.PANSSnegative),
                           panss_gen=np.asarray(d.PANSSgeneral)))
    return beh.merge(cl, on="eeg_id", how="left")


def ids(*groups):
    """Participants of the analyses: controls and patients with schizophrenia (schizoaffective excluded)."""
    s = subjects()
    return sorted(s[s.group.isin(groups or ("HC", "SCZ"))].subject.astype(int))


def group_of():
    return subjects().set_index("subject").group


# ------------------------------------------------------------------------------------------------ EEG files
@functools.lru_cache(maxsize=None)
def eeg_dir():
    hits = sorted({os.path.dirname(os.path.dirname(f)) for f in
                   glob.glob(os.path.join(DATA, "**", "*_DF_AV_*.vhdr"), recursive=True)})
    if not hits:
        raise FileNotFoundError(f"no EEG files (*_DF_AV_*.vhdr) under {DATA}")
    return hits[0]


def av_runs(eeg_id):
    """Audiovisual runs of one participant: (vhdr path, task 1 = auditory / 2 = visual report, run number)."""
    out = []
    for f in sorted(glob.glob(os.path.join(eeg_dir(), "**", f"{eeg_id}_DF_AV_*.vhdr"), recursive=True)):
        m = re.search(r"_AV_([AV])_(\d+)[a-z]?\.vhdr$", f)
        if m:
            out.append((f, {"A": 1, "V": 2}[m.group(1)], int(m.group(2))))
    return sorted(out, key=lambda r: r[2])


def read_markers(vhdr):
    """Stimulus markers of one run (file order): nA, nV, task, sample."""
    rows = []
    with open(vhdr[:-5] + ".vmrk", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = re.match(r"Mk\d+=Stimulus,DF_C(\d+)_A(\d)V(\d)-([AV]),(\d+),", line)
            if m:
                rows.append(dict(nA=int(m.group(2)), nV=int(m.group(3)), task={"A": 1, "V": 2}[m.group(4)],
                                 sample=int(m.group(5))))
    return pd.DataFrame(rows, columns=["nA", "nV", "task", "sample"])


def fixed_vhdr(vhdr, out_dir):
    """Some header files point to renamed data/marker files or contain malformed markers; a corrected copy of the
    header (and of the marker file, without the malformed lines) is written to out_dir when needed."""
    base = vhdr[:-5]
    txt = open(vhdr, encoding="utf-8", errors="ignore").read()
    want = {"DataFile": base + ".eeg", "MarkerFile": base + ".vmrk"}
    mk = open(base + ".vmrk", encoding="utf-8", errors="ignore").read().splitlines()
    bad = [l for l in mk if re.match(r"Mk\d+=", l) and not re.match(r"Mk\d+=[^,]*,[^,]*,\d+,", l)]
    if bad:
        os.makedirs(out_dir, exist_ok=True)
        want["MarkerFile"] = os.path.join(out_dir, os.path.basename(base) + ".vmrk")
        with open(want["MarkerFile"], "w", encoding="utf-8") as fh:
            fh.write("\n".join(l for l in mk if l not in bad) + "\n")
    if not bad and all(re.search(rf"^{k}={re.escape(os.path.basename(v))}\s*$", txt, re.M) for k, v in want.items()):
        return vhdr
    for k, v in want.items():
        txt = re.sub(rf"^{k}=.*$", f"{k}={v}", txt, flags=re.M)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, os.path.basename(vhdr))
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(txt)
    return out
