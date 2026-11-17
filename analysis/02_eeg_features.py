"""Prestimulus band power per trial.

Epochs are cut at stimulus onset (-1.5..0 s) before the wavelet transform and mirror-padded by 1 s.  Morlet power,
26 log-spaced frequencies 3-80 Hz, cycles = 0.4 x frequency (3 to 10), 20 ms steps; a bin is kept only if its wavelet
(+-3 SD) lies inside -1.5..0 s.  Log power is z-scored over trials in each channel/frequency/time bin and averaged over
the valid bins of each band (theta, alpha, beta, gamma) within -0.6..-0.2 s.
Trials: no artefact, not the first trial of a run, previous trial answered.
Output: derivatives/features/sub-XX.npz (F [trial, channel, band], channel names), sub-XX.csv (trial metadata)
"""
import os
import sys
import numpy as np
import mne
from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib import data, eeg                               # noqa: E402

mne.set_log_level("ERROR")
KEEP = ["subject", "run", "runTrialNumber", "trial_in_run", "task", "nA", "nV", "countingReport",
        "log_gap", "prev_countingReportRT", "prev_nA", "prev_nV", "prev_countingReport", "prev_absdisp"]
os.makedirs(eeg.FEAT, exist_ok=True)


def subject(si):
    ep, md = eeg.load_trials(si)
    ep.crop(-1.5, 0.0)
    x = ep.get_data(picks="eeg")
    pad = int(ep.info["sfreq"])
    x = np.pad(x, ((0, 0), (0, 0), (pad, pad)), mode="reflect")
    times = ep.times[::eeg.DECIM]
    tsel = times >= eeg.T_ANALYSIS - 1e-9
    p0 = pad // eeg.DECIM
    chunks = []
    for c0 in range(0, x.shape[1], 8):
        p = mne.time_frequency.tfr_array_morlet(x[:, c0:c0 + 8], ep.info["sfreq"], eeg.FREQS, n_cycles=eeg.NCYC,
                                                output="power", decim=eeg.DECIM, n_jobs=1)
        chunks.append(np.log10(p[..., p0:p0 + len(times)][..., tsel]).astype(np.float32))
    pw = np.concatenate(chunks, axis=1)
    del chunks, x
    times = times[tsel]
    mask = eeg.valid_mask(times)
    n, nch, nf, nt = pw.shape
    Y = pw.reshape(n, -1).astype(np.float64)
    Y = ((Y - Y.mean(0)) / Y.std(0)).reshape(n, nch, nf, nt)
    tw = (times >= eeg.WIN[0] - 1e-9) & (times <= eeg.WIN[1] + 1e-9)
    F = np.empty((n, nch, len(eeg.BANDS)))
    for k, (lo, hi) in enumerate(eeg.BANDS.values()):
        fi = (eeg.FREQS >= lo) & (eeg.FREQS <= hi)
        mm = mask[np.ix_(fi, tw)]
        sub = Y[:, :, fi][..., tw]
        F[:, :, k] = np.nanmean(np.where(mm[None, None], sub, np.nan).reshape(n, nch, -1), axis=-1)
    np.savez_compressed(os.path.join(eeg.FEAT, f"sub-{si:02d}.npz"), F=F.astype(np.float32), ch_names=np.array(ep.ch_names))
    md[KEEP].to_csv(os.path.join(eeg.FEAT, f"sub-{si:02d}.csv"), index=False)
    return si, n


if __name__ == "__main__":
    subs = [int(s) for s in sys.argv[1:]] or data.ids()
    for si, n in Parallel(n_jobs=int(os.environ.get("EEG_JOBS", "4")))(delayed(subject)(s) for s in subs):
        print(si, n, "trials", flush=True)
