from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rdd import profiles as prof
from rdd.schema import ClaimsDataset, SchemaError, validate_claims


def test_panel_spans_a_decade_of_delay():
    """The panel exists for contrast; a collapsed panel proves nothing."""
    anchors = [p.target_median_delay_years for p in prof.PANEL]
    assert max(anchors) - min(anchors) >= 10.0


def test_every_profile_has_a_literature_reference():
    for p in prof.PANEL:
        assert p.reference.strip(), f"{p.key} has no reference"


def test_referral_mix_is_a_distribution():
    for p in prof.PANEL:
        assert sum(p.referral_mix.values()) == pytest.approx(1.0)


def test_referral_mix_must_sum_to_one():
    with pytest.raises(ValueError):
        prof.DiseaseProfile(
            key="X", name="x", dx_code="A00", onset_age_mean=10, onset_age_sd=1,
            female_fraction=0.5, symptoms=prof.ALS.symptoms, misdx=prof.ALS.misdx,
            recognizing_specialties=("NEUROLOGY",), recognition_prob=0.5,
            referral_mix={"NEUROLOGY": 0.4}, confirm_codes=("1",),
            care_seeking_median_years=1, care_seeking_logsd=1, visit_rate=1,
            death_hazard=0.1, target_median_delay_years=1, reference="r",
        )


def test_symptom_specificity_is_bounded():
    with pytest.raises(ValueError):
        prof.Symptom("A00", "x", prob=0.5, onset_lag_years=0, specificity=1.7,
                     owning_specialty="NEUROLOGY")


def test_misdiagnosis_dwell_must_be_positive():
    with pytest.raises(ValueError):
        prof.Misdiagnosis("A00", "x", prob=0.5, dwell_years=0.0, specialty="NEUROLOGY")


def test_every_profile_has_at_least_one_specific_symptom():
    """A disease with no actionable finding could never be diagnosed at all."""
    for p in prof.PANEL:
        assert p.specific_symptom_codes, f"{p.key} has no symptom above 0.30 specificity"


def test_recognizing_specialty_is_reachable_by_referral():
    """Otherwise the simulated patient can never arrive anywhere useful."""
    for p in prof.PANEL:
        reachable = set(p.referral_mix) & set(p.recognizing_specialties)
        assert reachable, f"{p.key}: no recognising specialty appears in referral_mix"


def test_get_rejects_unknown_disease():
    with pytest.raises(KeyError):
        prof.get("NOT_A_DISEASE")


def _minimal_claims():
    return pd.DataFrame(
        {
            "patient_id": [1, 1], "claim_id": [1, 1], "day": [10, 10],
            "code": ["M25.50", "99213"], "code_type": ["DX", "PX"],
            "position": [1, 2], "specialty": ["PRIMARY_CARE", "PRIMARY_CARE"],
            "setting": ["OFFICE", "OFFICE"],
        }
    )


def _minimal_enrolment():
    return pd.DataFrame(
        {
            "patient_id": [1], "enrol_start": [0], "enrol_end": [1000],
            "birth_day": [-10000], "sex": ["F"], "death_day": [np.nan],
        }
    )


def test_schema_round_trips(tmp_path):
    ds = ClaimsDataset(_minimal_claims(), _minimal_enrolment(), source="test")
    ds.write(tmp_path)
    back = ClaimsDataset.read(tmp_path)
    assert back.source == "test"
    assert len(back.claims) == 2
    assert back.n_patients == 1


def test_schema_rejects_unknown_code_type():
    bad = _minimal_claims()
    bad.loc[0, "code_type"] = "XX"
    with pytest.raises(SchemaError):
        validate_claims(bad)


def test_schema_rejects_missing_columns():
    with pytest.raises(SchemaError):
        validate_claims(_minimal_claims().drop(columns=["setting"]))


def test_schema_rejects_backwards_enrolment():
    bad = _minimal_enrolment()
    bad.loc[0, "enrol_end"] = -5
    with pytest.raises(SchemaError):
        ClaimsDataset(_minimal_claims(), bad)
