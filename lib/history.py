"""History effect of Rohe et al. (2019): for each lag k = 1-5, trials are split by the numeric disparity |nA - nV| of
the trial k steps back in the same run (small <= 1, large >= 2), and p_common is refitted in each subset on a grid of
400 values, with the other parameters fixed at the participant's fit with static priors."""
import numpy as np
from lib import bci as M

LAGS = [1, 2, 3, 4, 5]
PG = np.linspace(0.0025, 0.9975, 400)


class Counts:
    """Minimal stand-in for SubjectData when only response counts are needed (static fit)."""
    def __init__(self, counts):
        self.counts = counts


def counts_of(cond, task, resp, mask):
    c = np.zeros((16, 2, 4))
    np.add.at(c, (cond[mask], task[mask] - 1, resp[mask] - 1), 1)
    return c


def log_table(v5):
    """log p(report | condition, task) for every p_common on the grid, other parameters from v5."""
    p = M.from_vec(v5)
    pre = M.precompute(p["sigma_A"], p["sigma_V"], p["sigma_P"], p["mu_P"])
    return np.log(np.stack([M.resp_table(*pre, P, M.LAPSE) for P in PG]))      # [400, 16, 2, 4]


def prev_disp(d, k):
    """Numeric disparity of the trial k back in the same run (NaN if there is none)."""
    disp = np.abs(d.nA - d.nV)
    out = np.full(len(disp), np.nan)
    out[k:] = disp[:-k]
    out[d.trial_in_run < k] = np.nan
    return out


def analyse(d, resp, logtab):
    """Per lag: p_common refitted after a small and after a large previous disparity."""
    valid = resp > 0
    out = []
    for k in LAGS:
        pd_ = prev_disp(d, k)
        row = dict(lag=k)
        for name, m in (("small", pd_ <= 1), ("large", pd_ >= 2)):
            mm = m & valid
            ll = (logtab * counts_of(d.cond, d.task, resp, mm)[None]).sum(axis=(1, 2, 3))
            row[f"p_{name}"] = PG[ll.argmax()]
        out.append(row)
    return out


def effect(d, resp, x_static):
    """History effect (small - large) at lags 1-5 for one set of responses (0 = no response); the static fit is redone
    on these responses, starting from x_static."""
    x, _ = M.fit_static(Counts(counts_of(d.cond, d.task, resp, resp > 0)), x0=x_static, polish=False)
    return [dict(lag=r["lag"], value=r["p_small"] - r["p_large"]) for r in analyse(d, resp, log_table(x))]
