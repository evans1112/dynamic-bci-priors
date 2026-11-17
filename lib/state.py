"""Brain-state versions of the filter in models.py: on each trial, prestimulus alpha z(t) shifts the causal prior
(logit P_alpha = logit P + b z) and/or the numerosity prior (mu_alpha = mu_P + c z).  The causal update proceeds from
the shifted prior; the numerosity shift is transient (mu_P itself is updated as before)."""
import numpy as np
from numba import njit
from lib import models as R
from lib.models import M, _trial_core


@njit(cache=True, fastmath=True)
def run_state(LA, CA, A2, LV, CV, V2, mI, tail, wO, mu_lo, mu_step, cond, task, resp, nA, nV, run_start,
              p0, kappa, rule, mu0, eta, target, rho, lapse, offP, offMu):
    T = cond.shape[0]; G = LA.shape[0]; nI = LA.shape[3]
    ll = np.zeros(T)
    acc0 = np.zeros((4, 4)); acc1 = np.zeros((4, 4)); acc = np.zeros((4, 4))
    pi_row = np.zeros(nI); s_row = np.zeros(nI); pred = np.zeros(4)
    P = p0; mu = mu0; prev = 0
    for t in range(T):
        if t == 0 or run_start[t]:
            P = p0; mu = mu0; prev = 0
        if P < 1e-6:
            P = 1e-6
        elif P > 1.0 - 1e-6:
            P = 1.0 - 1e-6
        a = np.log(P / (1.0 - P)) + offP[t]
        if a > 30.0:
            a = 30.0
        elif a < -30.0:
            a = -30.0
        Pe = 1.0 / (1.0 + np.exp(-a))
        mue = mu + offMu[t]
        c = cond[t]
        g0 = 0; w1 = 0.0
        if G > 1:
            u = (mue - mu_lo) / mu_step
            if u < 0.0:
                u = 0.0
            if u > G - 1 - 1e-9:
                u = G - 1 - 1e-9
            g0 = int(np.floor(u)); w1 = u - g0
        if task[t] == 1:
            _trial_core(LA[g0, c], CA[g0, c], A2[g0, c], mI, tail, wO, Pe, acc0, pi_row, s_row)
            if w1 > 0.0:
                _trial_core(LA[g0 + 1, c], CA[g0 + 1, c], A2[g0 + 1, c], mI, tail, wO, Pe, acc1, pi_row, s_row)
        else:
            _trial_core(LV[g0, c], CV[g0, c], V2[g0, c], mI, tail, wO, Pe, acc0, pi_row, s_row)
            if w1 > 0.0:
                _trial_core(LV[g0 + 1, c], CV[g0 + 1, c], V2[g0 + 1, c], mI, tail, wO, Pe, acc1, pi_row, s_row)
        tot_post = 0.0
        for b in range(4):
            for k in range(4):
                acc[b, k] = acc0[b, k] if w1 == 0.0 else (1.0 - w1) * acc0[b, k] + w1 * acc1[b, k]
            tot_post += acc[b, 1]
        rr = rho if prev > 0 else 0.0
        for b in range(4):
            pred[b] = (1.0 - rr) * ((1.0 - lapse) * acc[b, 0] + lapse * 0.25) + (rr if b == prev - 1 else 0.0)
        r = resp[t]; e_p = tot_post
        if r > 0:
            b = r - 1
            ll[t] = np.log(pred[b])
            if acc[b, 0] > 1e-300:
                fm = (1.0 - rr) * (1.0 - lapse) * acc[b, 0] / pred[b]
                e_p = fm * acc[b, 1] / acc[b, 0] + (1.0 - fm) * tot_post
        if rule == 0:
            P = Pe + kappa * (e_p - Pe)
        else:
            P = (1.0 - kappa) * e_p + kappa * p0
        if eta > 0.0:
            if target == 0:
                if r > 0:
                    mu = mu + eta * (r - mu)
            else:
                mu = mu + eta * (0.5 * (nA[t] + nV[t]) - mu)
        prev = r
    return ll

