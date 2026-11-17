"""Bayesian causal inference (BCI) for the flash/beep counting task, with a causal prior that can change from trial
to trial.

Per trial (Koerding et al. 2007): internal samples x_A ~ N(n_A, sigma_A^2), x_V ~ N(n_V, sigma_V^2), prior over
numbers N(mu_P, sigma_P^2); posterior of a common cause pi1 = P L1 / (P L1 + (1 - P) L2); the estimate is the
model average pi1 s_C1 + (1 - pi1) s_C2 (task-relevant), rounded to the nearest button 1-4, with a fixed lapse.

Causal prior across trials (restarts at p_common at the start of every run):
    delta rule   P(t+1) = P(t) + kappa (E[pi1(t)] - P(t))            kappa = 0 is the static model
    Markov rule  P(t+1) = (1 - kappa) E[pi1(t)] + kappa p_common      kappa = 1 is the static model
E[pi1(t)] is the expected posterior given the stimuli and the observed report, so the likelihood of a trial depends
only on earlier trials.  Models with a dynamic numerosity prior or response repetition are in models.py.
"""
import math
import numpy as np
from numba import njit
from scipy.stats import norm
from scipy.optimize import minimize

LAPSE = 0.01          # fixed lapse (uniform guess); same in every model
SIG_MIN, SIG_MAX = 0.05, 15.0
MU_MIN, MU_MAX = 0.0, 5.0     # prior mean kept near the stimulus range (1-4)
PNAMES_STATIC = ["p_common", "sigma_A", "sigma_V", "sigma_P", "mu_P"]
PNAMES_DYN = PNAMES_STATIC + ["kappa"]


# ----------------------------------------------------------------------------
# parameter transforms (optimiser works on an unconstrained vector)
# ----------------------------------------------------------------------------
def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _expit(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def to_vec(p_common, sigma_A, sigma_V, sigma_P, mu_P, kappa=None):
    v = [_logit(p_common), np.log(sigma_A), np.log(sigma_V), np.log(sigma_P),
         _logit((mu_P - MU_MIN) / (MU_MAX - MU_MIN))]
    if kappa is not None:
        v.append(_logit(kappa))
    return np.array(v, float)


def from_vec(v):
    out = dict(
        p_common=float(_expit(v[0])),
        sigma_A=float(np.clip(np.exp(np.clip(v[1], -20, 20)), SIG_MIN, SIG_MAX)),
        sigma_V=float(np.clip(np.exp(np.clip(v[2], -20, 20)), SIG_MIN, SIG_MAX)),
        sigma_P=float(np.clip(np.exp(np.clip(v[3], -20, 20)), SIG_MIN, SIG_MAX)),
        mu_P=float(MU_MIN + (MU_MAX - MU_MIN) * _expit(v[4])),
    )
    out["kappa"] = float(_expit(v[5])) if len(v) > 5 else 0.0
    return out


# ----------------------------------------------------------------------------
# numerical integration over the internal samples (x_A, x_V)
#
# The response is a step function of the samples (nearest button), so a plain grid gives
# a jagged likelihood and poorly resolved rare responses.  We therefore integrate
#   - the task-relevant sample ("inner") on a uniform grid over +-4 SD, and split the
#     normal probability mass of every grid segment exactly where the (linearly
#     interpolated) estimate crosses a button boundary;
#   - the other sample ("outer") with Gauss-Hermite quadrature.
# This makes the likelihood continuous in the parameters and accurate to ~0.01 log units.
# ----------------------------------------------------------------------------
N_INNER, N_OUTER, Z_MAX = 49, 12, 4.0
COND_NA = np.repeat(np.arange(1, 5), 4).astype(float)     # condition c = (nA-1)*4 + (nV-1)
COND_NV = np.tile(np.arange(1, 5), 4).astype(float)


def _set_grid(n_inner=N_INNER, n_outer=N_OUTER):
    global _ZI, _MI, _TAIL, _ZO, _WO
    _ZI = np.linspace(-Z_MAX, Z_MAX, n_inner)
    cdf = norm.cdf(_ZI)
    _MI = np.ascontiguousarray(np.stack([_ZI, cdf]))      # inner nodes and their normal CDF
    _TAIL = np.array([cdf[0], 1 - cdf[-1]])               # mass beyond the grid ends
    zo, wo = np.polynomial.hermite_e.hermegauss(n_outer)
    _ZO, _WO = zo, wo / wo.sum()


_set_grid()


def _terms(xA, xV, sigma_A, sigma_V, sigma_P, mu_P):
    vA, vV, vP = sigma_A ** 2, sigma_V ** 2, sigma_P ** 2
    den1 = vV * vA + vV * vP + vA * vP
    logL1 = -0.5 * np.log(den1) - 0.5 * ((xV - xA) ** 2 * vP + (xV - mu_P) ** 2 * vA
                                         + (xA - mu_P) ** 2 * vV) / den1
    logL2 = (-0.5 * np.log((vV + vP) * (vA + vP))
             - 0.5 * ((xV - mu_P) ** 2 / (vV + vP) + (xA - mu_P) ** 2 / (vA + vP)))
    sC1 = (xV / vV + xA / vA + mu_P / vP) / (1 / vV + 1 / vA + 1 / vP)
    sA2 = (xA / vA + mu_P / vP) / (1 / vA + 1 / vP)
    sV2 = (xV / vV + mu_P / vP) / (1 / vV + 1 / vP)
    return logL1 - logL2, sC1, sA2, sV2


def precompute(sigma_A, sigma_V, sigma_P, mu_P):
    """
    Everything that does not depend on the causal prior, for the 16 conditions.
    Returns arrays [16, n_outer, n_inner] for the auditory-report layout (inner = x_A)
    and the visual-report layout (inner = x_V): log-likelihood ratio (evidence for a
    common cause), the fused estimate and the task-relevant segregated estimate.
    """
    nA = COND_NA[:, None, None]; nV = COND_NV[:, None, None]
    zo = _ZO[None, :, None]; zi = _ZI[None, None, :]
    llrA, sC1A, sA2, _ = _terms(nA + sigma_A * zi, nV + sigma_V * zo,
                                sigma_A, sigma_V, sigma_P, mu_P)
    llrV, sC1V, _, sV2 = _terms(nA + sigma_A * zo, nV + sigma_V * zi,
                                sigma_A, sigma_V, sigma_P, mu_P)
    shp = llrA.shape
    c = lambda a: np.ascontiguousarray(np.broadcast_to(a, shp))
    return c(llrA), c(sC1A), c(sA2), c(llrV), c(sC1V), c(sV2), _MI, _TAIL, _WO


# ----------------------------------------------------------------------------
# core: response distribution for one (condition, task, prior)
# ----------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _btn(s):
    b = int(np.floor(s + 0.5)) - 1
    if b < 0:
        b = 0
    elif b > 3:
        b = 3
    return b


@njit(cache=True, fastmath=True)
def _trial_core(llr_c, sC1_c, s2_c, mI, tail, wO, P, acc, pi_row, s_row):
    """
    acc[b, 0..3] <- per button b: probability, and the probability-weighted sums of
    pi1, |llr| and |pi1 - 0.5| (divide by acc[b, 0] for conditional expectations).
    """
    nO, nI = llr_c.shape
    for b in range(4):
        for k in range(4):
            acc[b, k] = 0.0
    lo = np.log(P / (1.0 - P))
    for j in range(nO):
        w = wO[j]
        for i in range(nI):
            a = lo + llr_c[j, i]
            if a > 30.0:
                pi = 1.0
            elif a < -30.0:
                pi = 0.0
            else:
                pi = 1.0 / (1.0 + np.exp(-a))
            pi_row[i] = pi
            s_row[i] = pi * sC1_c[j, i] + (1.0 - pi) * s2_c[j, i]
        # tails beyond the grid: constant extrapolation
        for e in range(2):
            i = 0 if e == 0 else nI - 1
            b = _btn(s_row[i]); m = w * tail[e]
            acc[b, 0] += m; acc[b, 1] += m * pi_row[i]
            acc[b, 2] += m * abs(llr_c[j, i]); acc[b, 3] += m * abs(pi_row[i] - 0.5)
        for i in range(nI - 1):
            s0 = s_row[i]; s1 = s_row[i + 1]
            p0 = pi_row[i]; p1 = pi_row[i + 1]
            l0 = llr_c[j, i]; l1 = llr_c[j, i + 1]
            c0 = mI[1, i]; c1 = mI[1, i + 1]
            m = w * (c1 - c0)
            b = _btn(s0); b1 = _btn(s1)
            if b == b1:
                pm = 0.5 * (p0 + p1)
                acc[b, 0] += m; acc[b, 1] += m * pm
                acc[b, 2] += m * 0.5 * (abs(l0) + abs(l1)); acc[b, 3] += m * abs(pm - 0.5)
                continue
            ds = s1 - s0
            ta = 0.0; ca = c0
            z0 = mI[0, i]; dz = mI[0, i + 1] - z0
            up = ds > 0.0
            while True:
                if up:
                    if b == 3:
                        tb = 1.0
                    else:
                        tb = ((b + 1) + 0.5 - s0) / ds        # upper edge of button b+1
                else:
                    if b == 0:
                        tb = 1.0
                    else:
                        tb = ((b + 1) - 0.5 - s0) / ds        # lower edge of button b+1
                if tb >= 1.0:
                    tb = 1.0; cb = c1
                else:                                   # exact normal mass up to the crossing
                    cb = 0.5 * math.erfc(-(z0 + tb * dz) * 0.7071067811865476)
                tm = 0.5 * (ta + tb)
                mm = w * (cb - ca)
                pm = p0 + tm * (p1 - p0)
                lm = l0 + tm * (l1 - l0)
                acc[b, 0] += mm; acc[b, 1] += mm * pm
                acc[b, 2] += mm * abs(lm); acc[b, 3] += mm * abs(pm - 0.5)
                if tb >= 1.0:
                    break
                ta = tb; ca = cb
                b = b + 1 if up else b - 1


@njit(cache=True, fastmath=True)
def run_filter(llrA, sC1A, sA2, llrV, sC1V, sV2, mI, tail, wO,
               cond, task, resp, run_start, p0, kappa, lapse, use_resp, reset_runs, rule):
    """
    One pass through a subject's trials in presentation order.
    cond: 0..15, task: 1 = auditory report, 2 = visual report, resp: 1..4 (0 = missed)
    use_resp: 1 = prior update conditions on the observed response, 0 = on stimuli only
    reset_runs: 1 = prior restarts at p_common at each run, 0 = carried across runs
    rule: 0 = learn rule   P <- P + kappa * (pi1 - P)
          1 = Markov rule  P <- (1 - kappa) * pi1 + kappa * p_common   (kappa = 1 is static)
    Returns per trial: log-lik of the response (0 if missed), prior, full predicted
    response distribution, and the expected pi1, |evidence| and decisiveness.
    """
    T = cond.shape[0]
    nI = llrA.shape[2]
    ll = np.zeros(T); prior = np.zeros(T); pred = np.zeros((T, 4))
    epost = np.zeros(T); eabse = np.zeros(T); edec = np.zeros(T)
    acc = np.zeros((4, 4)); pi_row = np.zeros(nI); s_row = np.zeros(nI)
    P = p0
    for t in range(T):
        if t == 0 or (reset_runs == 1 and run_start[t]):
            P = p0
        if P < 1e-6:
            P = 1e-6
        elif P > 1.0 - 1e-6:
            P = 1.0 - 1e-6
        prior[t] = P
        c = cond[t]
        if task[t] == 1:
            _trial_core(llrA[c], sC1A[c], sA2[c], mI, tail, wO, P, acc, pi_row, s_row)
        else:
            _trial_core(llrV[c], sC1V[c], sV2[c], mI, tail, wO, P, acc, pi_row, s_row)
        tot_post = 0.0; tot_abse = 0.0; tot_dec = 0.0
        for b in range(4):
            pred[t, b] = (1.0 - lapse) * acc[b, 0] + lapse * 0.25
            tot_post += acc[b, 1]; tot_abse += acc[b, 2]; tot_dec += acc[b, 3]
        r = resp[t]
        e_p = tot_post; e_a = tot_abse; e_d = tot_dec
        if r > 0:
            b = r - 1
            pobs = pred[t, b]
            ll[t] = np.log(pobs)
            if use_resp == 1 and acc[b, 0] > 1e-300:
                w = (1.0 - lapse) * acc[b, 0] / pobs   # P(response came from the model)
                e_p = w * acc[b, 1] / acc[b, 0] + (1.0 - w) * tot_post
                e_a = w * acc[b, 2] / acc[b, 0] + (1.0 - w) * tot_abse
                e_d = w * acc[b, 3] / acc[b, 0] + (1.0 - w) * tot_dec
        epost[t] = e_p; eabse[t] = e_a; edec[t] = e_d
        if rule == 0:
            P = P + kappa * (e_p - P)
        else:
            P = (1.0 - kappa) * e_p + kappa * p0
    return ll, prior, pred, epost, eabse, edec


@njit(cache=True, fastmath=True)
def resp_table(llrA, sC1A, sA2, llrV, sC1V, sV2, mI, tail, wO, P, lapse):
    """Static model: predicted response distribution for all 16 conditions x 2 tasks."""
    out = np.zeros((16, 2, 4))
    nI = llrA.shape[2]
    acc = np.zeros((4, 4)); pi_row = np.zeros(nI); s_row = np.zeros(nI)
    for c in range(16):
        _trial_core(llrA[c], sC1A[c], sA2[c], mI, tail, wO, P, acc, pi_row, s_row)
        for b in range(4):
            out[c, 0, b] = (1.0 - lapse) * acc[b, 0] + lapse * 0.25
        _trial_core(llrV[c], sC1V[c], sV2[c], mI, tail, wO, P, acc, pi_row, s_row)
        for b in range(4):
            out[c, 1, b] = (1.0 - lapse) * acc[b, 0] + lapse * 0.25
    return out


# ----------------------------------------------------------------------------
# data container and likelihoods
# ----------------------------------------------------------------------------
class SubjectData:
    """Audiovisual trials of one participant, in presentation order."""

    def __init__(self, df):
        df = df.sort_values(["run", "runTrialNumber"]).reset_index(drop=True)
        self.df = df
        self.nA = df.nA.values.astype(float)
        self.nV = df.nV.values.astype(float)
        self.cond = ((df.nA.values - 1) * 4 + (df.nV.values - 1)).astype(np.int64)
        self.task = df.task.values.astype(np.int64)
        r = df.countingReport.values
        self.resp = np.where(np.isnan(r), 0, r).astype(np.int64)
        self.run = df.run.values.astype(np.int64)
        self.run_start = np.r_[True, self.run[1:] != self.run[:-1]]
        self.trial_in_run = df.groupby("run").cumcount().values.astype(float)
        self.valid = self.resp > 0
        # counts[c, task, r] for the static model
        self.counts = np.zeros((16, 2, 4))
        for c, k, rr in zip(self.cond[self.valid], self.task[self.valid], self.resp[self.valid]):
            self.counts[c, k - 1, rr - 1] += 1


def filter_pass(v, d, use_resp=1, reset_runs=1, rule=0):
    p = from_vec(v)
    pre = precompute(p["sigma_A"], p["sigma_V"], p["sigma_P"], p["mu_P"])
    return run_filter(*pre, d.cond, d.task, d.resp, d.run_start, p["p_common"], p["kappa"],
                      LAPSE, use_resp, reset_runs, rule)


def nll_dynamic(v, d, use_resp=1, reset_runs=1, rule=0):
    ll = filter_pass(v, d, use_resp, reset_runs, rule)[0]
    return -float(ll.sum())


def nll_static(v, d):
    p = from_vec(v)
    pre = precompute(p["sigma_A"], p["sigma_V"], p["sigma_P"], p["mu_P"])
    tab = resp_table(*pre, min(max(p["p_common"], 1e-6), 1 - 1e-6), LAPSE)
    return -float((d.counts * np.log(tab)).sum())


# ----------------------------------------------------------------------------
# fitting: L-BFGS-B from several starts, then a Nelder-Mead polish
# ----------------------------------------------------------------------------
def _lb(fun, x0, args, maxfun=1200):
    return minimize(fun, x0, args=args, method="L-BFGS-B", options=dict(eps=1e-4, maxfun=maxfun))


def _nm(fun, x0, args, maxfev=800):
    return minimize(fun, x0, args=args, method="Nelder-Mead",
                    options=dict(adaptive=True, xatol=1e-3, fatol=1e-4, maxfev=maxfev,
                                 maxiter=maxfev))


def _multi(fun, starts, args, polish=True, maxfev=800):
    best = None
    for s in starts:
        r = _lb(fun, np.asarray(s, float), args)
        if best is None or r.fun < best.fun:
            best = r
    if polish:
        r = _nm(fun, best.x, args, maxfev)
        if r.fun < best.fun:
            best = r
    return best.x, float(best.fun)


def static_starts(rng, n):
    s = [to_vec(0.4, 0.5, 1.0, 2.5, 2.3), to_vec(0.6, 0.4, 0.8, 1.5, 2.5),
         to_vec(0.2, 0.7, 1.5, 4.0, 2.0)]
    while len(s) < n:
        s.append(to_vec(rng.uniform(0.1, 0.9), rng.uniform(0.2, 1.2), rng.uniform(0.3, 2.5),
                        rng.uniform(0.8, 6), rng.uniform(1.0, 4.0)))
    return s[:n]


def fit_static(d, n_starts=8, seed=0, x0=None, polish=True):
    starts = [x0] if x0 is not None else static_starts(np.random.default_rng(seed), n_starts)
    return _multi(nll_static, starts, (d,), polish)


def fit_dynamic(d, x_static, kappas=(0.01, 0.05, 0.2), use_resp=1, reset_runs=1,
                extra_starts=(), polish=True, rule=0):
    """Starts from the static solution with several kappas, so LL(dynamic) >= LL(static)."""
    starts = [np.r_[x_static, _logit(k)] for k in kappas] + list(extra_starts)
    x, f = _multi(nll_dynamic, starts, (d, use_resp, reset_runs, rule), polish)
    k_static = -30.0 if rule == 0 else 30.0             # the value of kappa that is the static model
    f0 = nll_dynamic(np.r_[x_static, k_static], d, use_resp, reset_runs, rule)
    if f0 < f:
        x, f = np.r_[x_static, k_static], f0
    return x, f

