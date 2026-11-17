"""All behavioural models as one trial-by-trial filter (static priors, delta-rule and Markov causal priors, dynamic
numerosity prior, response repetition, and their combinations).

  causal prior       P(t+1) = P + kappa (E[pi1] - P)               delta rule (rule = 0); kappa = 0 is static
                     P(t+1) = (1 - kappa) E[pi1] + kappa p_common    Markov rule (rule = 1); kappa = 1 is static
  numerosity prior   mu_P(t+1) = mu_P + eta (target(t) - mu_P)       target = report of trial t, or the mean of its
                                                                     numbers of beeps and flashes; eta = 0 is static
  repetition         p(r_t) = (1 - rho) p_BCI(r_t) + rho [r_t = r_{t-1}]
All states restart at the start of each run.  E[pi1] is conditioned on the observed report, weighting the probability
that the report came from the BCI observer rather than from a lapse or a repetition.  With eta > 0, response tables
are precomputed on a grid of mu_P (step MU_STEP) and interpolated linearly.
"""
import numpy as np
from numba import njit
from lib import bci as M
from lib.bci import _trial_core

FIT_GRID, FINE_GRID = (49, 12), (241, 48)
MU_STEP = 0.1
MU_GRID = np.round(np.arange(M.MU_MIN, M.MU_MAX + 1e-9, MU_STEP), 6)
NULL = -30.0                                             # logit of a vanishing rate

SPEC = {  # name: (rule, has_kappa, has_rho, has_eta, target)
    "static": (0, False, False, False, 0),
    "learn": (0, True, False, False, 0),
    "markov": (1, True, False, False, 0),
    "rep": (0, False, True, False, 0),
    "num_est": (0, False, False, True, 0),
    "num_stim": (0, False, False, True, 1),
    "markov_rep": (1, True, True, False, 0),
    "markov_num_est": (1, True, False, True, 0),
    "markov_num_stim": (1, True, False, True, 1),
    "learn_num_est": (0, True, False, True, 0),
}
K = {m: 5 + sum(s[1:4]) for m, s in SPEC.items()}


def unpack(name, v):
    rule, hk, hr, he, target = SPEC[name]
    p = M.from_vec(v[:5]); i = 5
    kappa = (M._expit(v[i]) if hk else 0.0); i += hk
    rho = (M._expit(v[i]) if hr else 0.0); i += hr
    eta = (M._expit(v[i]) if he else 0.0)
    if not hk and rule == 1:
        kappa = 1.0
    return dict(p_common=p["p_common"], sigma_A=p["sigma_A"], sigma_V=p["sigma_V"], sigma_P=p["sigma_P"],
                mu_P=p["mu_P"], kappa=float(kappa), rho=float(rho), eta=float(eta), rule=rule, target=target)


def tables(p, dynamic_mu):
    """Precomputed arrays [G, 16, n_outer, n_inner] (G = 1 at mu_P when the numerosity prior is static)."""
    grid = MU_GRID if dynamic_mu else np.array([p["mu_P"]])
    pre = [M.precompute(p["sigma_A"], p["sigma_V"], p["sigma_P"], m) for m in grid]
    arrs = [np.ascontiguousarray(np.stack([q[k] for q in pre])) for k in range(6)]
    return arrs, pre[0][6], pre[0][7], pre[0][8], float(grid[0])


@njit(cache=True, fastmath=True)
def run_general(LA, CA, A2, LV, CV, V2, mI, tail, wO, mu_lo, mu_step, cond, task, resp, nA, nV, run_start,
                p0, kappa, rule, mu0, eta, target, rho, lapse):
    T = cond.shape[0]; G = LA.shape[0]; nI = LA.shape[3]
    ll = np.zeros(T); pred = np.zeros((T, 4)); prior = np.zeros(T); mus = np.zeros(T); epost = np.zeros(T)
    acc0 = np.zeros((4, 4)); acc1 = np.zeros((4, 4)); acc = np.zeros((4, 4))
    pi_row = np.zeros(nI); s_row = np.zeros(nI)
    P = p0; mu = mu0; prev = 0
    for t in range(T):
        if t == 0 or run_start[t]:
            P = p0; mu = mu0; prev = 0
        if P < 1e-6:
            P = 1e-6
        elif P > 1.0 - 1e-6:
            P = 1.0 - 1e-6
        prior[t] = P; mus[t] = mu
        c = cond[t]
        g0 = 0; w1 = 0.0
        if G > 1:
            u = (mu - mu_lo) / mu_step
            if u < 0.0:
                u = 0.0
            if u > G - 1 - 1e-9:
                u = G - 1 - 1e-9
            g0 = int(np.floor(u)); w1 = u - g0
        if task[t] == 1:
            _trial_core(LA[g0, c], CA[g0, c], A2[g0, c], mI, tail, wO, P, acc0, pi_row, s_row)
            if w1 > 0.0:
                _trial_core(LA[g0 + 1, c], CA[g0 + 1, c], A2[g0 + 1, c], mI, tail, wO, P, acc1, pi_row, s_row)
        else:
            _trial_core(LV[g0, c], CV[g0, c], V2[g0, c], mI, tail, wO, P, acc0, pi_row, s_row)
            if w1 > 0.0:
                _trial_core(LV[g0 + 1, c], CV[g0 + 1, c], V2[g0 + 1, c], mI, tail, wO, P, acc1, pi_row, s_row)
        tot_post = 0.0
        for b in range(4):
            for k in range(4):
                acc[b, k] = acc0[b, k] if w1 == 0.0 else (1.0 - w1) * acc0[b, k] + w1 * acc1[b, k]
            tot_post += acc[b, 1]
        rr = rho if prev > 0 else 0.0
        for b in range(4):
            pb = (1.0 - lapse) * acc[b, 0] + lapse * 0.25
            pred[t, b] = (1.0 - rr) * pb + (rr if b == prev - 1 else 0.0)
        r = resp[t]; e_p = tot_post
        if r > 0:
            b = r - 1
            pobs = pred[t, b]
            ll[t] = np.log(pobs)
            if acc[b, 0] > 1e-300:
                fm = (1.0 - rr) * (1.0 - lapse) * acc[b, 0] / pobs     # P(report came from the BCI observer)
                e_p = fm * acc[b, 1] / acc[b, 0] + (1.0 - fm) * tot_post
        epost[t] = e_p
        if rule == 0:
            P = P + kappa * (e_p - P)
        else:
            P = (1.0 - kappa) * e_p + kappa * p0
        if eta > 0.0:
            if target == 0:
                if r > 0:
                    mu = mu + eta * (r - mu)
            else:
                mu = mu + eta * (0.5 * (nA[t] + nV[t]) - mu)
        prev = r
    return ll, pred, prior, mus, epost


def evaluate(name, v, d, grid=None):
    """Per-trial outputs of model `name` at parameters v (optionally on another integration grid)."""
    if grid is not None:
        M._set_grid(*grid)
    try:
        p = unpack(name, v)
        arrs, mI, tail, wO, mu_lo = tables(p, p["eta"] > 0)
        return run_general(*arrs, mI, tail, wO, mu_lo, MU_STEP, d.cond, d.task, d.resp, d.nA, d.nV,
                           d.run_start, p["p_common"], p["kappa"], p["rule"], p["mu_P"], p["eta"], p["target"],
                           p["rho"], M.LAPSE)
    finally:
        if grid is not None:
            M._set_grid(*FIT_GRID)


def nll(v, name, d):
    return -float(evaluate(name, v, d)[0].sum())


@njit(cache=True)
def simulate(sA, sV, sP, mu0, p0, kappa, rule, eta, target, rho, lapse, nA, nV, task, run_start, seed):
    """Generative process: updates use the observer's actual posterior and report."""
    np.random.seed(seed)
    T = nA.shape[0]
    resp = np.zeros(T, np.int64)
    vA, vV, vP = sA ** 2, sV ** 2, sP ** 2
    den1 = vV * vA + vV * vP + vA * vP
    P = p0; mu = mu0; prev = 0
    for t in range(T):
        if t == 0 or run_start[t]:
            P = p0; mu = mu0; prev = 0
        if P < 1e-6:
            P = 1e-6
        elif P > 1.0 - 1e-6:
            P = 1.0 - 1e-6
        xA = nA[t] + sA * np.random.randn(); xV = nV[t] + sV * np.random.randn()
        logL1 = -0.5 * np.log(den1) - 0.5 * ((xV - xA) ** 2 * vP + (xV - mu) ** 2 * vA + (xA - mu) ** 2 * vV) / den1
        logL2 = (-0.5 * np.log((vV + vP) * (vA + vP))
                 - 0.5 * ((xV - mu) ** 2 / (vV + vP) + (xA - mu) ** 2 / (vA + vP)))
        a = np.log(P / (1.0 - P)) + logL1 - logL2
        pi = 1.0 / (1.0 + np.exp(-a)) if a > -30 else 0.0
        sC1 = (xV / vV + xA / vA + mu / vP) / (1 / vV + 1 / vA + 1 / vP)
        if task[t] == 1:
            s2 = (xA / vA + mu / vP) / (1 / vA + 1 / vP)
        else:
            s2 = (xV / vV + mu / vP) / (1 / vV + 1 / vP)
        s = pi * sC1 + (1.0 - pi) * s2
        b = int(np.floor(s + 0.5))
        if b < 1:
            b = 1
        elif b > 4:
            b = 4
        if prev > 0 and np.random.rand() < rho:
            b = prev
        elif np.random.rand() < lapse:
            b = np.random.randint(1, 5)
        resp[t] = b
        if rule == 0:
            P = P + kappa * (pi - P)
        else:
            P = (1.0 - kappa) * pi + kappa * p0
        if eta > 0.0:
            if target == 0:
                mu = mu + eta * (b - mu)
            else:
                mu = mu + eta * (0.5 * (nA[t] + nV[t]) - mu)
        prev = b
    return resp


def sim(name, v, d, seed):
    p = unpack(name, v)
    return simulate(p["sigma_A"], p["sigma_V"], p["sigma_P"], p["mu_P"], p["p_common"], p["kappa"], p["rule"],
                    p["eta"], p["target"], p["rho"], M.LAPSE, d.nA, d.nV, d.task, d.run_start, seed)
