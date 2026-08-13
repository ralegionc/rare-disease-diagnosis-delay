"""Quality gates.

These are the tests that would catch a change which quietly stops the pipeline
from working. Two of them assert that the *bias is still there*: if the
retrospective design ever stopped understating delay on this simulator, the
simulator would have stopped reproducing the problem the project is about, and
that is a failure even though it looks like an improvement.
"""

from __future__ import annotations

import numpy as np
import pytest

from rdd import profiles as prof
from rdd.bias import (
    competing_risk_comparison,
    design_comparison,
    lookback_sweep,
    truncate_history,
)
from rdd.validate import (
    QUALITY_GATES,
    check_gates,
    delay_recovery,
    onset_proxy_accuracy,
    phenotype_accuracy,
    recovery_scores,
)


@pytest.fixture(scope="module")
def recovery(recovery_dataset, study):
    return delay_recovery(recovery_dataset, study)


def test_lookback_sweep_saturates(medium_dataset, study):
    """A short history cannot show a long delay -- for any disease."""
    sw = lookback_sweep(medium_dataset, prof.get("FABRY"), windows=(3, 5, 8, 14),
                        study=study).dropna(subset=["median_delay"])
    assert len(sw) >= 3
    short = sw.loc[sw["history_years"] == 3, "median_delay"]
    long = sw.loc[sw["history_years"] == 14, "median_delay"]
    if len(short) and len(long):
        assert float(short.iloc[0]) < float(long.iloc[0])
    # And nothing may exceed the history it was measured in.
    assert (sw["median_delay"] <= sw["history_years"] + 1e-6).all()


def test_truncate_history_shrinks_the_window(medium_dataset, study):
    sub, sub_study = truncate_history(medium_dataset, 4.0, study)
    assert sub.claims["day"].min() >= sub_study.db_start
    assert len(sub.claims) < len(medium_dataset.claims)


def test_retrospective_design_understates_delay(recovery):
    """The headline claim of the study, asserted as a test."""
    ok = recovery.dropna(subset=["understatement_fraction"])
    assert len(ok) >= 4
    assert ok["understatement_fraction"].mean() >= QUALITY_GATES[
        "min_retrospective_understatement"
    ]
    assert (ok["retrospective_median"] < ok["true_median_from_onset"]).mean() >= 0.8


def test_bias_is_mostly_measurement_not_selection(recovery):
    ok = recovery.dropna(subset=["selection_component", "measurement_component"])
    assert abs(ok["measurement_component"].median()) > abs(
        ok["selection_component"].median()
    )


def test_delay_ranking_survives_the_bias(recovery):
    """Levels are badly wrong; the ordering is not. That distinction is the
    most useful thing a claims study of this kind can offer."""
    s = recovery_scores(recovery)
    assert s["pearson_r"] >= QUALITY_GATES["min_delay_correlation"]
    assert s["spearman_rho"] >= QUALITY_GATES["min_delay_rank_correlation"]


def test_quality_gates_pass(recovery_dataset, study, recovery):
    phen = phenotype_accuracy(recovery_dataset, study)
    proxy = onset_proxy_accuracy(recovery_dataset, study)
    fails = check_gates(recovery_scores(recovery), phen, proxy)
    assert not fails, "; ".join(fails)


def test_onset_proxy_is_late_not_early(medium_dataset, study):
    proxy = onset_proxy_accuracy(medium_dataset, study)
    assert (proxy["median_proxy_error_years"] > 0).all(), (
        "the proxy cannot precede the first claim; a negative value means the "
        "origin is being taken from outside the observed record"
    )


@pytest.mark.parametrize("key", ["ALS", "LOPD"])
def test_naive_curve_never_falls_below_aalen_johansen(medium_dataset, study, key):
    cr = competing_risk_comparison(medium_dataset, prof.get(key), study)
    assert len(cr)
    assert (cr["overstatement"] >= -1e-9).all(), "1-KM must never fall below AJ"
    assert cr["overstatement"].max() > 0.005


def test_competing_risk_correction_is_material_for_pompe(medium_dataset, study):
    cr = competing_risk_comparison(medium_dataset, prof.get("LOPD"), study)
    rel = (cr["overstatement"] / cr["aj_diagnosed"].replace(0, np.nan)).max()
    assert rel > 0.10


def test_high_mortality_disease_shows_the_smaller_correction(medium_dataset, study):
    """Counter-intuitive but real: ALS mortality selects *before* cohort entry.

    Patients who die undiagnosed fast never accumulate a symptom signature, so
    they never enter the risk set and no competing-risk estimator can see them.
    A small correction is not reassurance.
    """
    def rel_max(key):
        cr = competing_risk_comparison(medium_dataset, prof.get(key), study)
        return float((cr["overstatement"] / cr["aj_diagnosed"].replace(0, np.nan)).max())

    assert rel_max("ALS") < rel_max("LOPD")


def test_prospective_design_is_not_conditioned_on_the_outcome(medium_dataset, study):
    d = design_comparison(medium_dataset, prof.get("hEDS"), study)
    assert d["n_suspicion_design"] > d["n_diagnosed_design"]
    assert 0.0 < d["prospective_event_frac"] < 1.0
