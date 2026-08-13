from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rdd.adapters.generic import load_generic, map_setting, map_specialty
from rdd.adapters.omop import OMOP_QUERIES


def test_specialty_mapping_reports_unmapped_share():
    s = pd.Series(["Neurology", "Family Practice", "Vexillology", "NEUROLOGY"])
    mapped, unmapped = map_specialty(s)
    assert mapped.tolist() == ["NEUROLOGY", "PRIMARY_CARE", "OTHER", "NEUROLOGY"]
    assert unmapped == pytest.approx(0.25)


def test_setting_mapping_handles_place_of_service_codes():
    s = pd.Series(["11", "23", "inpatient", "nonsense"])
    assert map_setting(s).tolist() == ["OFFICE", "ER", "INPATIENT", "OFFICE"]


def test_generic_adapter_round_trip(tmp_path: Path):
    claims = pd.DataFrame(
        {
            "patient_id": [1, 1, 2],
            "claim_id": [10, 10, 11],
            "service_date": ["2015-01-05", "2015-01-05", "2016-03-02"],
            "code": ["m25.50", "99213", "M79.7"],
            "code_type": ["DX", "PX", "DX"],
            "position": [1, 2, 1],
            "specialty": ["Rheumatology", "Rheumatology", "Family Practice"],
            "setting": ["11", "11", "11"],
        }
    )
    enrol = pd.DataFrame(
        {
            "patient_id": [1, 2],
            "enrol_start": ["2014-01-01", "2014-01-01"],
            "enrol_end": ["2018-01-01", "2018-01-01"],
            "birth_date": ["1980-05-01", "1975-02-02"],
            "sex": ["F", "M"],
        }
    )
    claims.to_csv(tmp_path / "claims.csv", index=False)
    enrol.to_csv(tmp_path / "enrolment.csv", index=False)

    ds, notes = load_generic(tmp_path)
    assert ds.n_patients == 2
    assert ds.claims["code"].iloc[0] == "M25.50"  # upper-cased
    assert ds.claims["specialty"].iloc[0] == "RHEUMATOLOGY"
    assert ds.enrolment["death_day"].isna().all()
    assert any("death" in n for n in notes), "missing death date must be declared"


def test_omop_queries_reference_the_standard_tables():
    q = OMOP_QUERIES["claims"]
    for table in ("condition_occurrence", "procedure_occurrence", "visit_occurrence",
                  "provider", "concept"):
        assert table in q
    assert "observation_period" in OMOP_QUERIES["enrolment"]
    assert "death" in OMOP_QUERIES["enrolment"]


def test_pipeline_runs_end_to_end(tmp_path, monkeypatch, small_dataset):
    """Smoke test: analyse -> figures -> report against a tiny dataset."""
    import rdd.config as cfg
    import rdd.pipeline as pipe
    from rdd.report import write_report

    tab, fig = tmp_path / "tables", tmp_path / "figures"
    tab.mkdir(); fig.mkdir()
    monkeypatch.setattr(pipe, "TABDIR", tab)
    monkeypatch.setattr(pipe, "SUMMARY_PATH", tab / "summary.json")
    monkeypatch.setattr(pipe, "FIGDIR", fig)
    monkeypatch.setattr(cfg, "TABDIR", tab)

    import rdd.report as rep
    monkeypatch.setattr(rep, "TABDIR", tab)
    monkeypatch.setattr(rep, "OUT", tmp_path)

    summary = pipe.stage_analyse(small_dataset)
    assert "recovery_scores" in summary
    assert (tab / "cohort_summary.csv").exists()
    assert (tab / "delay_recovery.csv").exists()

    path = write_report(tmp_path / "report.md")
    text = path.read_text(encoding="utf-8")
    assert "[SYNTHETIC]" in text and "[METHOD]" in text
    assert "not** been run on real claims" in text


def test_no_network_calls_anywhere():
    """The whole pipeline must run offline; this guards against a regression."""
    import ast

    src = Path(__file__).resolve().parents[1] / "src" / "rdd"
    banned = {"requests", "urllib", "urllib3", "httpx", "aiohttp", "socket"}
    for py in src.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                names = {(node.module or "").split(".")[0]}
            else:
                continue
            assert not (names & banned), f"{py.name} imports {names & banned}"
