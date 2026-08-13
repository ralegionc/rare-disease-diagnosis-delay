"""Estimator correctness against closed forms and known references.

These are the tests that matter most, because everything downstream inherits
whatever these get wrong, and a survival bug does not crash -- it produces a
plausible curve.
"""

from __future__ import annotations

import numpy as np
import pytest

from rdd.survival import (
    aalen_johansen,
    fit_parametric,
    fit_weibull,
    kaplan_meier,
    logrank,
    nelson_aalen,
    quantile,
    rmst,
)


def test_km_matches_hand_computed_example():
    # Klein & Moeschberger style worked example.
    t = np.array([1, 2, 3, 4, 5], float)
    e = np.array([1, 0, 1, 1, 0], float)
    km = kaplan_meier(t, e)
    # S(1) = 4/5; S(3) = 4/5 * 2/3; S(4) = that * 1/2
    assert km.survival[1] == pytest.approx(0.8)
    assert km.survival[2] == pytest.approx(0.8 * 2 / 3)
    assert km.survival[3] == pytest.approx(0.8 * 2 / 3 * 0.5)


def test_km_with_no_censoring_is_the_ecdf():
    rng = np.random.default_rng(0)
    t = rng.exponential(2.0, 400)
    km = kaplan_meier(t, np.ones_like(t))
    for q in (0.25, 0.5, 0.75):
        assert km.at(np.quantile(t, q)) == pytest.approx(1 - q, abs=0.03)


def test_km_recovers_exponential_survival_under_heavy_censoring():
    rng = np.random.default_rng(1)
    true_t = rng.exponential(5.0, 4000)
    cens = rng.exponential(5.0, 4000)
    obs = np.minimum(true_t, cens)
    ev = (true_t <= cens).astype(float)
    km = kaplan_meier(obs, ev)
    for t in (1.0, 2.0, 4.0):
        assert km.at(t) == pytest.approx(np.exp(-t / 5.0), abs=0.04)


def test_left_truncation_changes_the_answer():
    """A delayed-entry risk set must not include not-yet-entered subjects."""
    rng = np.random.default_rng(2)
    n = 3000
    entry = rng.uniform(0, 3, n)
    t = entry + rng.exponential(4.0, n)
    e = np.ones(n)
    naive = kaplan_meier(t, e)
    correct = kaplan_meier(t, e, entry=entry)
    # Ignoring truncation makes early survival look too high, because subjects
    # are credited with person-time before they were observable.
    assert correct.at(2.0) < naive.at(2.0)


def test_km_rejects_entry_after_exit():
    with pytest.raises(ValueError):
        kaplan_meier([1.0], [1.0], entry=[2.0])


def test_quantile_returns_nan_when_curve_never_reaches_it():
    t = np.array([1.0, 2.0, 3.0])
    e = np.array([1.0, 0.0, 0.0])
    est, lo, hi = quantile(kaplan_meier(t, e), 0.5)
    assert np.isnan(est)


def test_rmst_of_uncensored_data_is_the_truncated_mean():
    rng = np.random.default_rng(3)
    t = rng.exponential(3.0, 3000)
    km = kaplan_meier(t, np.ones_like(t))
    area, se = rmst(km, 5.0)
    assert area == pytest.approx(np.mean(np.minimum(t, 5.0)), abs=0.08)
    assert se >= 0


def test_logrank_null_is_not_significant():
    rng = np.random.default_rng(4)
    t = rng.exponential(3.0, 800)
    g = rng.integers(0, 2, 800)
    chi2, df, p = logrank(t, np.ones_like(t), g)
    assert df == 1
    assert p > 0.05


def test_logrank_detects_a_real_difference():
    rng = np.random.default_rng(5)
    a = rng.exponential(2.0, 400)
    b = rng.exponential(6.0, 400)
    t = np.concatenate([a, b])
    g = np.concatenate([np.zeros(400), np.ones(400)])
    chi2, df, p = logrank(t, np.ones_like(t), g)
    assert p < 1e-10


def test_nelson_aalen_tracks_the_true_cumulative_hazard():
    rng = np.random.default_rng(6)
    t = rng.exponential(4.0, 3000)
    tt, H, V = nelson_aalen(t, np.ones_like(t))
    for target in (1.0, 3.0):
        idx = np.searchsorted(tt, target) - 1
        assert H[idx] == pytest.approx(target / 4.0, abs=0.12)


def test_aalen_johansen_cifs_sum_to_one_minus_survival():
    rng = np.random.default_rng(7)
    n = 3000
    t1 = rng.exponential(3.0, n)
    t2 = rng.exponential(6.0, n)
    t = np.minimum(t1, t2)
    cause = np.where(t1 <= t2, 1, 2)
    tt, cifs = aalen_johansen(t, cause)
    total = cifs[1] + cifs[2]
    km = kaplan_meier(t, np.ones_like(t))
    for target in (1.0, 2.0, 4.0):
        i = max(np.searchsorted(tt, target) - 1, 0)
        assert total[i] == pytest.approx(1 - km.at(target), abs=0.02)


def test_aalen_johansen_is_below_one_minus_km_for_one_cause():
    """The whole reason the estimator exists."""
    rng = np.random.default_rng(8)
    n = 2000
    t1 = rng.exponential(4.0, n)
    t2 = rng.exponential(2.0, n)  # strong competing risk
    t = np.minimum(t1, t2)
    cause = np.where(t1 <= t2, 1, 2)
    tt, cifs = aalen_johansen(t, cause)
    km_naive = kaplan_meier(t, (cause == 1).astype(float))
    i = max(np.searchsorted(tt, 3.0) - 1, 0)
    assert cifs[1][i] < 1 - km_naive.at(3.0) - 0.05


def test_weibull_mle_recovers_known_parameters():
    rng = np.random.default_rng(9)
    shape, scale = 1.7, 4.0
    t = scale * rng.weibull(shape, 4000)
    f = fit_weibull(t, np.ones_like(t))
    assert f.shape == pytest.approx(shape, rel=0.08)
    assert f.scale == pytest.approx(scale, rel=0.08)


def test_weibull_mle_handles_right_censoring():
    rng = np.random.default_rng(10)
    shape, scale = 0.8, 5.0
    true_t = scale * rng.weibull(shape, 5000)
    cens = rng.exponential(6.0, 5000)
    obs = np.minimum(true_t, cens)
    ev = (true_t <= cens).astype(float)
    f = fit_weibull(obs, ev)
    assert f.shape == pytest.approx(shape, rel=0.12)
    assert f.scale == pytest.approx(scale, rel=0.20)


def test_lognormal_median_is_not_the_weibull_formula():
    """Regression: the log-normal branch once returned negative medians."""
    rng = np.random.default_rng(11)
    t = np.exp(rng.normal(0.4, 0.7, 3000))
    f = fit_weibull(t, np.ones_like(t), family="lognormal")
    assert f.median() > 0
    assert f.median() == pytest.approx(np.exp(0.4), rel=0.10)


def test_akaike_weights_prefer_the_generating_family():
    rng = np.random.default_rng(12)
    t = rng.exponential(3.0, 3000)
    fits, weights, best = fit_parametric(t, np.ones_like(t))
    assert best in ("exponential", "weibull")
    assert weights["exponential"] + weights["weibull"] > 0.9
    assert sum(weights.values()) == pytest.approx(1.0)


def test_fit_with_no_events_degrades_gracefully():
    f = fit_weibull(np.array([1.0, 2.0]), np.array([0.0, 0.0]))
    assert not f.converged
    assert np.isnan(f.shape)
