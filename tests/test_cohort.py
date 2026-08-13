from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rdd import profiles as prof
from rdd.cohort import (
    _qualifying_index,
    apply_phenotype,
    build_cohort,
    continuous_spans,
    onset_proxy,
    signature_codes,
)
from rdd.config import DEFAULT_STUDY


def test_continuous_spans_bridges_small_gaps_but_not_large_ones():
    enrol = pd.DataFrame(
        {
            "patient_id": [1, 1, 2, 2],
            "enrol_start": [0, 40, 0, 500],
            "enrol_end": [30, 200, 100, 900],
            "birth_day": [-1, -1, -1, -1],
            "sex": ["F", "F", "M", "M"],
            "death_day": [np.nan] * 4,
        }
    )
    out = continuous_spans(enrol, DEFAULT_STUDY).set_index("patient_id")
    assert out.at[1, "obs_start"] == 0 and out.at[1, "obs_end"] == 200  # bridged
    assert out.at[2, "obs_end"] - out.at[2, "obs_start"] == 400  # longest span only


def test_qualifying_index_rejects_an_isolated_rule_out_code():
    """A single code four years early must not become the index date."""
    days = np.array([0, 1500, 1560])
    idx = _qualifying_index(days, DEFAULT_STUDY)
    assert idx == 1500


def test_qualifying_index_needs_the_spacing():
    days = np.array([0, 5])  # same episode, not a confirmed diagnosis
    assert _qualifying_index(days, DEFAULT_STUDY) is None


def test_qualifying_index_accepts_a_proper_pair():
    days = np.array([100, 200])
    assert _qualifying_index(days, DEFAULT_STUDY) == 100


def test_onset_proxy_requires_persistence():
    days = np.array([0.0])
    codes = np.array(["M25.50"], dtype=object)
    assert np.isnan(onset_proxy(days, codes, DEFAULT_STUDY))


def test_onset_proxy_requires_two_distinct_codes():
    """Three claims of the same generic code are not a disease signature."""
    days = np.array([0.0, 100.0, 200.0])
    same = np.array(["M25.50"] * 3, dtype=object)
    assert np.isnan(onset_proxy(days, same, DEFAULT_STUDY))
    mixed = np.array(["M25.50", "R53.83", "M25.50"], dtype=object)
    assert onset_proxy(days, mixed, DEFAULT_STUDY) == 0.0


def test_onset_proxy_skips_an_isolated_early_claim():
    days = np.array([0.0, 3000.0, 3050.0, 3100.0])
    codes = np.array(["R53.83", "M25.50", "R53.83", "M24.40"], dtype=object)
    assert onset_proxy(days, codes, DEFAULT_STUDY) == 3000.0


def test_signature_set_includes_the_misdiagnoses():
    sig = signature_codes(prof.get("hEDS"))
    assert "M79.7" in sig, "fibromyalgia is part of the hEDS odyssey, not outside it"


def test_phenotype_has_high_precision(small_dataset, study):
    truth = small_dataset.truth
    for p in prof.PANEL:
        found = set(apply_phenotype(small_dataset, p, study).ids())
        actual = set(truth.loc[truth["disease"] == p.key, "patient_id"])
        if not found:
            continue
        ppv = len(found & actual) / len(found)
        assert ppv >= 0.90, f"{p.key} PPV {ppv:.3f}"


def test_cohorts_obey_the_survival_contract(small_dataset, study):
    for p in prof.PANEL:
        for design in ("suspicion", "diagnosed"):
            co = build_cohort(small_dataset, p, study, design=design)
            if co.empty:
                continue
            assert (co["exit"] >= 0).all()
            assert co["event"].isin([0, 1]).all()
            assert co["cause"].isin([0, 1, 2]).all()
            # event and cause must agree
            assert ((co["cause"] == 1) == (co["event"] == 1)).all()


def test_diagnosed_design_contains_only_events(small_dataset, study):
    co = build_cohort(small_dataset, prof.get("hEDS"), study, design="diagnosed")
    if len(co):
        assert co["event"].min() == 1


def test_suspicion_design_contains_censored_patients(small_dataset, study):
    co = build_cohort(small_dataset, prof.get("hEDS"), study, design="suspicion")
    assert len(co) and co["event"].mean() < 1.0


def test_bad_design_name_is_rejected(small_dataset, study):
    with pytest.raises(ValueError):
        build_cohort(small_dataset, prof.get("ALS"), study, design="whatever")


def test_distinct_code_rule_keeps_the_risk_set_from_exploding(small_dataset, study):
    """Without it, the hEDS suspicion cohort recruits most of the controls."""
    co = build_cohort(small_dataset, prof.get("hEDS"), study, design="suspicion")
    assert len(co) < 0.75 * small_dataset.n_patients
