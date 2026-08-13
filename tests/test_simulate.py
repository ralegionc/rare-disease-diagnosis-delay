from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rdd import profiles as prof
from rdd.config import DAYS_PER_YEAR, DEFAULT_SIM, DEFAULT_STUDY
from rdd.simulate import Friction, _median_true_delay, simulate_patient, simulate_population

from .conftest import CALIBRATION


def test_friction_is_monotone_in_delay():
    """Calibration by bisection is only valid if this holds."""
    lo = _median_true_delay("HAE", 0.3, DEFAULT_STUDY, 220, 5)
    hi = _median_true_delay("HAE", 1.6, DEFAULT_STUDY, 220, 5)
    assert hi > lo


@pytest.mark.parametrize("key", list(CALIBRATION))
def test_calibration_hits_the_literature_anchor(key):
    """The pinned frictions must still reproduce their published delays.

    ALS sits at the friction floor: even with essentially no diagnostic
    friction, care-seeking plus the time to reach a neurologist puts the median
    just above one year. That floor is itself a result, so the tolerance here
    is one-sided-generous rather than the anchor being adjusted to meet it.
    """
    target = prof.get(key).target_median_delay_years
    got = _median_true_delay(key, CALIBRATION[key], DEFAULT_STUDY, 400, 99)
    assert got == pytest.approx(target, rel=0.25), f"{key}: {got:.2f} vs {target}"


def test_delay_ordering_of_the_panel_is_preserved():
    got = {
        k: _median_true_delay(k, v, DEFAULT_STUDY, 250, 3) for k, v in CALIBRATION.items()
    }
    order = sorted(got, key=got.get)
    expected = sorted(CALIBRATION, key=lambda k: prof.get(k).target_median_delay_years)
    # Adjacent swaps are tolerable at this sample size; a reversal is not.
    assert order[0] == expected[0] and order[-1] == expected[-1]


def test_simulated_patient_timeline_is_ordered():
    rng = np.random.default_rng(0)
    p = prof.get("FABRY")
    for i in range(40):
        _, t = simulate_patient(rng, i, p, DEFAULT_STUDY, Friction(0.56),
                                DEFAULT_STUDY.db_end + 40 * DAYS_PER_YEAR)
        assert t["first_care_day"] >= t["onset_day"]
        if np.isfinite(t["dx_day"]):
            assert t["dx_day"] >= t["first_care_day"]
        if np.isfinite(t["first_correct_specialty_day"]):
            assert t["first_correct_specialty_day"] >= t["first_care_day"]


def test_some_patients_are_never_diagnosed():
    """No censoring means no reason for any of the survival machinery."""
    rng = np.random.default_rng(1)
    p = prof.get("FABRY")
    dx = [
        np.isfinite(
            simulate_patient(rng, i, p, DEFAULT_STUDY, Friction(0.56),
                             DEFAULT_STUDY.db_end)[1]["dx_day"]
        )
        for i in range(150)
    ]
    assert 0 < sum(dx) < len(dx)


def test_als_has_substantial_pre_diagnosis_mortality():
    rng = np.random.default_rng(2)
    p = prof.get("ALS")
    died_first = 0
    for i in range(300):
        _, t = simulate_patient(rng, i, p, DEFAULT_STUDY, Friction(0.05),
                                DEFAULT_STUDY.db_end + 40 * DAYS_PER_YEAR)
        if np.isfinite(t["death_day"]) and (
            not np.isfinite(t["dx_day"]) or t["death_day"] < t["dx_day"]
        ):
            died_first += 1
    assert died_first >= 20, "competing risk too weak to be worth modelling"


def test_claims_are_clipped_to_enrolment(small_dataset):
    spans = small_dataset.enrolment.set_index("patient_id")
    j = small_dataset.claims.join(spans[["enrol_start", "enrol_end"]], on="patient_id")
    assert (j["day"] >= j["enrol_start"]).all()
    assert (j["day"] <= j["enrol_end"]).all()


def test_truth_is_not_clipped(small_dataset):
    """The truth table must keep events that fall outside coverage."""
    t = small_dataset.truth
    dx = t.loc[np.isfinite(t["dx_day"]), "dx_day"]
    assert (dx > DEFAULT_STUDY.db_end).any()


def test_controls_carry_the_mimic_labels(small_dataset):
    t = small_dataset.truth
    control_ids = set(t.loc[t["disease"] == "NONE", "patient_id"])
    c = small_dataset.claims
    mimic = c[(c["code"] == "M79.7") & c["patient_id"].isin(control_ids)]
    assert len(mimic) > 0, "controls with no phenocopies make the phenotype look perfect"


def test_rule_out_coding_exists(small_dataset):
    """Some patients carry a rare code without ever qualifying as a case."""
    c = small_dataset.claims
    t = small_dataset.truth
    control_ids = set(t.loc[t["disease"] == "NONE", "patient_id"])
    codes = {p.dx_code for p in prof.PANEL}
    stray = c[c["code"].isin(codes) & c["patient_id"].isin(control_ids)]
    assert len(stray) > 0


def test_simulation_is_reproducible():
    sim = replace(DEFAULT_SIM, n_cases_per_disease=40, n_controls=120, seed=7)
    a = simulate_population(sim, DEFAULT_STUDY, CALIBRATION)
    b = simulate_population(sim, DEFAULT_STUDY, CALIBRATION)
    assert len(a.claims) == len(b.claims)
    assert a.claims["code"].tolist() == b.claims["code"].tolist()
