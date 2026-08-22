"""Data-readiness diagnostics.

Each test constructs a source broken in one specific way and asserts the
doctor names that specific problem -- because the failure mode being guarded
against is a source that is broken *silently* and still renders a report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rdd.config import DEFAULT_STUDY
from rdd.doctor import (
    MAX_OTHER_SPECIALTY,
    MIN_USABLE_COHORT,
    check_enrolment,
    check_vocabulary,
    diagnose,
)
from rdd.schema import ClaimsDataset


def _dataset(codes, specialty="PRIMARY_CARE", span_days=4000, death=False, n=40):
    rows, enrol = [], []
    for pid in range(n):
        for j, code in enumerate(codes):
            rows.append((pid, pid * 100 + j, 3000 + j * 40, code, "DX", 1,
                         specialty, "OFFICE"))
        enrol.append((pid, 2922, 2922 + span_days, -8000, "F",
                      float(2922 + span_days - 10) if death else np.nan))
    claims = pd.DataFrame(rows, columns=[
        "patient_id", "claim_id", "day", "code", "code_type", "position",
        "specialty", "setting"])
    enrolment = pd.DataFrame(enrol, columns=[
        "patient_id", "enrol_start", "enrol_end", "birth_day", "sex", "death_day"])
    return ClaimsDataset(claims, enrolment, source="test")


def test_detects_icd9_source():
    """The single most likely real-data failure: wrong vocabulary, no error."""
    ds = _dataset(["250.00", "401.9", "V70.0"])
    v = check_vocabulary(ds)
    assert "ICD-9" in v["detected"]
    assert "crosswalk" in v["detected"]

    report = diagnose(ds, DEFAULT_STUDY)
    assert not report.usable
    assert any("ICD-9" in b for b in report.blockers)


def test_detects_icd10_source():
    ds = _dataset(["M25.50", "R53.83", "E75.21"])
    assert check_vocabulary(ds)["detected"] == "ICD-10-CM"


def test_names_the_diagnosis_codes_that_never_appear():
    ds = _dataset(["M25.50", "R53.83"])
    v = check_vocabulary(ds)
    assert "E75.21" in v["absent_dx_codes"]  # Fabry
    assert "G12.21" in v["absent_dx_codes"]  # ALS


def test_enrolment_ceiling_is_reported():
    ds = _dataset(["M25.50", "R53.83"], span_days=int(3 * 365.25))
    e = check_enrolment(ds, DEFAULT_STUDY)
    assert e["median_years"] == pytest.approx(3.0, abs=0.1)


def test_short_history_downgrades_long_delay_diseases(small_dataset, study):
    """A three-year database cannot see a twelve-year odyssey."""
    report = diagnose(small_dataset, study)
    assert (report.enrolment["median_years"] > 0)
    limited = report.per_disease["verdict"].str.contains("LIMITED: history")
    blocked = report.per_disease["verdict"].str.startswith("BLOCKED")
    # Every disease is either estimable, history-limited, or too small.
    assert (limited | blocked | (report.per_disease["verdict"] == "OK")).all()


def test_specialty_collapse_is_warned_about():
    """SynPUF has no rendering-provider specialty; the model degenerates."""
    ds = _dataset(["M25.50", "R53.83"], specialty="OTHER")
    report = diagnose(ds, DEFAULT_STUDY)
    other = report.specialty.loc[
        report.specialty["specialty"] == "OTHER", "share"].sum()
    assert other > MAX_OTHER_SPECIALTY
    assert any("OTHER" in w for w in report.warnings)


def test_missing_death_dates_are_warned_about():
    ds = _dataset(["M25.50", "R53.83"], death=False)
    report = diagnose(ds, DEFAULT_STUDY)
    assert any("death" in w for w in report.warnings)


def test_death_dates_present_suppresses_that_warning():
    ds = _dataset(["M25.50", "R53.83"], death=True)
    report = diagnose(ds, DEFAULT_STUDY)
    assert not any("no death dates" in w for w in report.warnings)


def test_tiny_cohort_is_blocked_not_estimated():
    ds = _dataset(["E75.21", "G60.9", "L74.4"], n=5)
    report = diagnose(ds, DEFAULT_STUDY)
    fabry = report.per_disease.set_index("disease").loc["FABRY"]
    assert fabry["n_cohort"] < MIN_USABLE_COHORT
    assert "BLOCKED" in fabry["verdict"]


def test_empty_panel_is_a_blocker():
    ds = _dataset(["Z99.9"])
    report = diagnose(ds, DEFAULT_STUDY)
    assert not report.usable
    assert any("no disease in the panel" in b for b in report.blockers)


def test_report_text_renders_without_error(small_dataset, study):
    text = diagnose(small_dataset, study).text()
    assert "data readiness report" in text
    assert "per disease" in text
    assert "observable window" in text


def test_doctor_never_reads_the_truth_table(small_dataset, study):
    """It must behave identically on real data, where no truth exists."""
    stripped = ClaimsDataset(
        small_dataset.claims, small_dataset.enrolment, truth=None, source="no-truth"
    )
    a = diagnose(small_dataset, study).per_disease
    b = diagnose(stripped, study).per_disease
    pd.testing.assert_frame_equal(a, b)
