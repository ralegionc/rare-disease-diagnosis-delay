"""Survival estimators, written from scratch in numpy.

Why not lifelines or scikit-survival: this package needs left truncation,
competing risks and a semi-Markov multi-state fit simultaneously, and the
combination is exactly where the wrappers stop agreeing with each other. The
implementations here are short enough to audit line by line, and carry no
compiled dependencies, so the whole analysis runs anywhere numpy does.

Everything takes and returns *years*. Every routine supports **left truncation**
via ``entry``, because in this problem left truncation is not an edge case: a
patient whose symptoms began in 2004 but whose claims start in 2008 enters the
risk set at age-of-symptoms 4, and treating them as entering at 0 is the single
easiest way to invent a shorter diagnostic delay than actually occurred.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

__all__ = [
    "KMResult",
    "kaplan_meier",
    "quantile",
    "rmst",
    "logrank",
    "nelson_aalen",
    "aalen_johansen",
    "fit_weibull",
    "fit_parametric",
    "WeibullFit",
]


# ---------------------------------------------------------------------------
# Kaplan-Meier
# ---------------------------------------------------------------------------


@dataclass
class KMResult:
    time: np.ndarray
    survival: np.ndarray
    var: np.ndarray  # Greenwood variance of S(t)
    at_risk: np.ndarray
    events: np.ndarray

    def ci(self, alpha: float = 0.05):
        """Log-log transformed confidence band (stays inside [0, 1])."""
        z = stats.norm.ppf(1 - alpha / 2)
        s = np.clip(self.survival, 1e-12, 1 - 1e-12)
        se_loglog = np.sqrt(np.clip(self.var, 0, None)) / (s * np.abs(np.log(s)))
        theta = np.exp(z * se_loglog)
        lo = s ** (1 / theta)
        hi = s**theta
        # Degenerate points (S == 1 exactly) get a trivial band.
        deg = ~np.isfinite(se_loglog)
        lo = np.where(deg, self.survival, lo)
        hi = np.where(deg, self.survival, hi)
        return np.clip(lo, 0, 1), np.clip(hi, 0, 1)

    def at(self, t: float) -> float:
        idx = np.searchsorted(self.time, t, side="right") - 1
        return 1.0 if idx < 0 else float(self.survival[idx])


def _as_arrays(*arrs):
    out = [np.asarray(a, dtype=float).ravel() for a in arrs]
    n = len(out[0])
    for a in out:
        if len(a) != n:
            raise ValueError("all inputs must have the same length")
    return out


def kaplan_meier(time, event, entry=None) -> KMResult:
    """Product-limit estimator with optional left truncation.

    ``entry[i]`` is the time at which subject *i* came under observation on the
    analysis clock. The risk set at ``t`` is ``entry < t <= time``, which is the
    delayed-entry form -- not ``time >= t``.
    """
    time, event = _as_arrays(time, event)
    entry = np.zeros_like(time) if entry is None else np.asarray(entry, float).ravel()
    if np.any(entry > time):
        raise ValueError("entry times must not exceed exit times")
    if np.any(time < 0):
        raise ValueError("negative survival times")

    tt = np.unique(time[event == 1])
    if tt.size == 0:
        return KMResult(
            np.array([0.0]), np.array([1.0]), np.array([0.0]),
            np.array([float(len(time))]), np.array([0.0]),
        )

    n_risk = np.empty(tt.size)
    d = np.empty(tt.size)
    for i, t in enumerate(tt):
        n_risk[i] = np.sum((entry < t) & (time >= t))
        d[i] = np.sum((time == t) & (event == 1))

    ok = n_risk > 0
    tt, n_risk, d = tt[ok], n_risk[ok], d[ok]
    surv = np.cumprod(1.0 - d / n_risk)

    # Greenwood
    with np.errstate(divide="ignore", invalid="ignore"):
        inc = np.where(n_risk - d > 0, d / (n_risk * (n_risk - d)), 0.0)
    var = surv**2 * np.cumsum(inc)

    time_out = np.concatenate([[0.0], tt])
    surv_out = np.concatenate([[1.0], surv])
    var_out = np.concatenate([[0.0], var])
    risk_out = np.concatenate([[n_risk[0]], n_risk])
    ev_out = np.concatenate([[0.0], d])
    return KMResult(time_out, surv_out, var_out, risk_out, ev_out)


def quantile(km: KMResult, q: float = 0.5, alpha: float = 0.05):
    """Quantile of the survival distribution, with a log-log CI.

    Returns ``(estimate, lo, hi)``; any component is ``nan`` when the curve
    never falls that far -- which for a rare disease with heavy censoring is a
    real and reportable outcome, not a failure.
    """
    target = 1.0 - q
    idx = np.argmax(km.survival <= target) if np.any(km.survival <= target) else -1
    est = float(km.time[idx]) if idx >= 0 else np.nan
    lo_band, hi_band = km.ci(alpha)
    lo = float(km.time[np.argmax(hi_band <= target)]) if np.any(hi_band <= target) else np.nan
    hi = float(km.time[np.argmax(lo_band <= target)]) if np.any(lo_band <= target) else np.nan
    return est, lo, hi


def rmst(km: KMResult, horizon: float):
    """Restricted mean survival time up to ``horizon``, with its standard error.

    RMST is the workhorse here rather than the mean, because the mean requires
    the tail and the tail is exactly what censoring removes. RMST is also what
    makes the delay decomposition additive: mean time spent in each state up to
    a common horizon sums to the mean total time up to that horizon.
    """
    t = np.concatenate([km.time, [horizon]])
    s = np.concatenate([km.survival, [km.survival[-1]]])
    keep = t <= horizon
    t = np.concatenate([t[keep], [horizon]])
    s = np.concatenate([s[keep], [s[keep][-1]]])
    dt = np.diff(t)
    area = float(np.sum(s[:-1] * dt))

    # Variance (Klein & Moeschberger 4.5.4): integral of the tail area squared.
    var = 0.0
    ev_times = km.time[1:]
    for i, ti in enumerate(ev_times, start=1):
        if ti > horizon or km.at_risk[i] <= km.events[i]:
            continue
        tail_t = np.concatenate([km.time[km.time >= ti], [horizon]])
        tail_s = np.concatenate([km.survival[km.time >= ti], [km.survival[-1]]])
        keep_t = tail_t <= horizon
        tail_t = np.concatenate([tail_t[keep_t], [horizon]])
        tail_s = np.concatenate([tail_s[keep_t], [tail_s[keep_t][-1]]])
        a = float(np.sum(tail_s[:-1] * np.diff(tail_t)))
        var += a**2 * km.events[i] / (km.at_risk[i] * (km.at_risk[i] - km.events[i]))
    return area, float(np.sqrt(var))


# ---------------------------------------------------------------------------
# Two-sample and k-sample tests
# ---------------------------------------------------------------------------


def logrank(time, event, group, entry=None):
    """k-sample log-rank test. Returns ``(chi2, df, p)``."""
    time, event = _as_arrays(time, event)
    group = np.asarray(group).ravel()
    entry = np.zeros_like(time) if entry is None else np.asarray(entry, float).ravel()
    levels = np.unique(group)
    k = len(levels)
    if k < 2:
        raise ValueError("logrank needs at least two groups")

    tt = np.unique(time[event == 1])
    obs = np.zeros(k)
    exp = np.zeros(k)
    V = np.zeros((k, k))
    for t in tt:
        n_g = np.array([np.sum((group == g) & (entry < t) & (time >= t)) for g in levels], float)
        d_g = np.array(
            [np.sum((group == g) & (time == t) & (event == 1)) for g in levels], float
        )
        n, d = n_g.sum(), d_g.sum()
        if n <= 1 or d == 0:
            continue
        e_g = n_g * d / n
        obs += d_g
        exp += e_g
        c = d * (n - d) / (n - 1)
        V += c * (np.diag(n_g / n) - np.outer(n_g, n_g) / n**2)

    z = (obs - exp)[:-1]
    Vr = V[:-1, :-1]
    try:
        chi2 = float(z @ np.linalg.solve(Vr, z))
    except np.linalg.LinAlgError:  # pragma: no cover - degenerate strata
        chi2 = float(z @ np.linalg.pinv(Vr) @ z)
    df = k - 1
    return chi2, df, float(stats.chi2.sf(chi2, df))


# ---------------------------------------------------------------------------
# Cumulative hazard and competing risks
# ---------------------------------------------------------------------------


def nelson_aalen(time, event, entry=None):
    """Nelson-Aalen cumulative hazard. Returns ``(t, H, var)``."""
    time, event = _as_arrays(time, event)
    entry = np.zeros_like(time) if entry is None else np.asarray(entry, float).ravel()
    tt = np.unique(time[event == 1])
    H, V = [], []
    h = v = 0.0
    for t in tt:
        n = np.sum((entry < t) & (time >= t))
        d = np.sum((time == t) & (event == 1))
        if n > 0:
            h += d / n
            v += d / n**2
        H.append(h)
        V.append(v)
    return (
        np.concatenate([[0.0], tt]),
        np.concatenate([[0.0], H]),
        np.concatenate([[0.0], V]),
    )


def aalen_johansen(time, cause, entry=None, causes=None):
    """Cause-specific cumulative incidence functions.

    ``cause`` is 0 for censored and 1..K for the competing event types.

    This matters for ALS specifically. Roughly a third of simulated ALS
    patients die before anyone writes down G12.21, and death is not
    "censoring" -- a dead patient is not still waiting for a diagnosis.
    Reporting ``1 - KM`` there overstates the probability of eventual
    diagnosis, sometimes badly, because it extrapolates the diagnosis hazard
    over person-time that does not exist.
    """
    time = np.asarray(time, float).ravel()
    cause = np.asarray(cause, int).ravel()
    entry = np.zeros_like(time) if entry is None else np.asarray(entry, float).ravel()
    causes = tuple(sorted(set(cause[cause > 0]))) if causes is None else tuple(causes)

    tt = np.unique(time[cause > 0])
    n_risk = np.array([np.sum((entry < t) & (time >= t)) for t in tt], float)
    d_all = np.array([np.sum((time == t) & (cause > 0)) for t in tt], float)
    ok = n_risk > 0
    tt, n_risk, d_all = tt[ok], n_risk[ok], d_all[ok]

    # Overall (all-cause) survival is the correct weight for each increment.
    s_overall = np.concatenate([[1.0], np.cumprod(1.0 - d_all / n_risk)])[:-1]

    out = {}
    for c in causes:
        d_c = np.array([np.sum((time == t) & (cause == c)) for t in tt], float)
        cif = np.cumsum(s_overall * d_c / n_risk)
        out[c] = np.concatenate([[0.0], cif])
    return np.concatenate([[0.0], tt]), out


# ---------------------------------------------------------------------------
# Parametric sojourn fits
# ---------------------------------------------------------------------------


@dataclass
class WeibullFit:
    shape: float
    scale: float
    loglik: float
    n: int
    n_events: int
    converged: bool
    family: str = "weibull"

    @property
    def aic(self) -> float:
        return 2 * self._k() - 2 * self.loglik

    def _k(self) -> int:
        return 1 if self.family == "exponential" else 2

    def median(self) -> float:
        # For the log-normal parameterisation ``scale`` is mu (a log-scale
        # location, which can be negative) and ``shape`` is sigma, so the
        # Weibull formula does not apply and silently returns nonsense.
        if self.family == "lognormal":
            return float(np.exp(self.scale))
        if self.family == "exponential":
            return self.scale * np.log(2.0)
        return self.scale * np.log(2.0) ** (1.0 / self.shape)

    def mean(self) -> float:
        from math import gamma

        if self.family == "lognormal":
            return float(np.exp(self.scale + 0.5 * self.shape**2))
        return self.scale * gamma(1.0 + 1.0 / self.shape)

    def survival(self, t):
        t = np.asarray(t, float)
        if self.family == "lognormal":
            return 1.0 - stats.norm.cdf((np.log(np.maximum(t, 1e-12)) - self.scale) / self.shape)
        return np.exp(-((np.maximum(t, 0.0) / self.scale) ** self.shape))


def _neg_loglik(params, t, e, family):
    if family == "lognormal":
        mu, log_sigma = params
        sigma = np.exp(log_sigma)
        z = (np.log(np.maximum(t, 1e-9)) - mu) / sigma
        ll = np.sum(e * (stats.norm.logpdf(z) - np.log(sigma) - np.log(np.maximum(t, 1e-9))))
        ll += np.sum((1 - e) * stats.norm.logsf(z))
        return -ll
    log_shape, log_scale = params
    k, lam = np.exp(log_shape), np.exp(log_scale)
    t = np.maximum(t, 1e-9)
    z = (t / lam) ** k
    ll = np.sum(e * (np.log(k) - np.log(lam) + (k - 1) * np.log(t / lam))) - np.sum(z)
    return -ll


def fit_weibull(time, event, family: str = "weibull") -> WeibullFit:
    """MLE for right-censored sojourn times.

    ``family`` is one of ``weibull``, ``exponential`` (shape fixed at 1) or
    ``lognormal``. Fitting all three and comparing AIC is the point: an
    exponential sojourn means the wait is memoryless -- each visit is an
    independent lottery ticket -- while a Weibull shape below 1 means the
    hazard of being diagnosed *falls* the longer you have gone undiagnosed,
    which is the quantitative signature of a wrong label sticking.
    """
    t = np.asarray(time, float).ravel()
    e = np.asarray(event, float).ravel()
    if t.size == 0 or e.sum() == 0:
        return WeibullFit(np.nan, np.nan, np.nan, int(t.size), 0, False, family)

    t = np.maximum(t, 1e-6)
    m = float(np.mean(t[e == 1])) if e.sum() else float(np.mean(t))
    if family == "lognormal":
        x0 = np.array([np.log(max(m, 1e-3)), np.log(0.8)])
    else:
        x0 = np.array([0.0, np.log(max(m, 1e-3))])

    if family == "exponential":
        def obj(p):
            return _neg_loglik(np.array([0.0, p[0]]), t, e, "weibull")

        res = optimize.minimize(obj, x0[1:], method="Nelder-Mead")
        shape, scale = 1.0, float(np.exp(res.x[0]))
    else:
        res = optimize.minimize(
            _neg_loglik, x0, args=(t, e, family), method="Nelder-Mead",
            options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-6},
        )
        if family == "lognormal":
            scale, shape = float(res.x[0]), float(np.exp(res.x[1]))
        else:
            shape, scale = float(np.exp(res.x[0])), float(np.exp(res.x[1]))

    return WeibullFit(
        shape=shape, scale=scale, loglik=float(-res.fun), n=int(t.size),
        n_events=int(e.sum()), converged=bool(res.success), family=family,
    )


def fit_parametric(time, event, families=("exponential", "weibull", "lognormal")):
    """Fit several families and rank them by Akaike weight.

    Reporting the weights rather than a bare argmin keeps the model-selection
    step honest: when two families are within a couple of AIC units, the data
    do not distinguish them and the write-up should say so.
    """
    fits = {f: fit_weibull(time, event, f) for f in families}
    aics = np.array([fits[f].aic for f in families], float)
    finite = np.isfinite(aics)
    weights = dict.fromkeys(families, np.nan)
    if finite.any():
        d = aics - np.nanmin(aics[finite])
        w = np.where(finite, np.exp(-0.5 * d), 0.0)
        w = w / w.sum()
        weights = dict(zip(families, w))
    best = min((f for f in families if np.isfinite(fits[f].aic)),
               key=lambda f: fits[f].aic, default=None)
    return fits, weights, best
