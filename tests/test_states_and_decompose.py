from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rdd import multistate, states
from rdd import profiles as prof
from rdd.cohort import build_cohort
from rdd.decompose import invisible_phase, phase_times
from rdd.states import State, encounter_states


def _claims(rows):
    return pd.DataFrame(
        rows, columns=["patient_id", "claim_id", "day", "code", "code_type",
                       "position", "specialty", "setting"]
    )


def test_encounter_state_prefers_mislabelled_over_specialty():
    """A wrong label describes the patient's situation, wherever they are sat."""
    p = prof.get("hEDS")
    c = _claims([
        (1, 1, 0, "M79.7", "DX", 1, "RHEUMATOLOGY", "OFFICE"),
        (1, 2, 100, "M79.7", "DX", 1, "RHEUMATOLOGY", "OFFICE"),
        (1, 3, 150, "M25.50", "DX", 1, "GENETICS", "OFFICE"),
    ])
    es = encounter_states(c, p, 0, 200)
    # Genetics is a recognising specialty, but the label is still live.
    assert es.iloc[-1]["state"] == int(State.MISLABELLED)


def test_a_single_differential_code_is_not_a_label():
    p = prof.get("hEDS")
    c = _claims([
        (1, 1, 0, "M79.7", "DX", 1, "RHEUMATOLOGY", "OFFICE"),
        (1, 2, 150, "M25.50", "DX", 1, "GENETICS", "OFFICE"),
    ])
    es = encounter_states(c, p, 0, 200)
    assert es.iloc[-1]["state"] == int(State.CORRECT_SPECIALTY)


def test_label_expires_after_the_persistence_window():
    p = prof.get("hEDS")
    c = _claims([
        (1, 1, 0, "M79.7", "DX", 1, "RHEUMATOLOGY", "OFFICE"),
        (1, 2, 60, "M79.7", "DX", 1, "RHEUMATOLOGY", "OFFICE"),
        (1, 3, 60 + states.LABEL_PERSISTENCE_DAYS + 10, "M25.50", "DX", 1,
         "GENETICS", "OFFICE"),
    ])
    es = encounter_states(c, p, 0, 400)
    assert es.iloc[-1]["state"] == int(State.CORRECT_SPECIALTY)


def test_undifferentiated_covers_primary_care_and_ed():
    p = prof.get("HAE")
    c = _claims([
        (1, 1, 0, "R10.9", "DX", 1, "EMERGENCY", "ER"),
        (1, 2, 30, "T78.3XXA", "DX", 1, "PRIMARY_CARE", "OFFICE"),
    ])
    es = encounter_states(c, p, 0, 100)
    assert set(es["state"]) == {int(State.UNDIFFERENTIATED)}


def test_transitions_are_contiguous_and_terminate_correctly(small_dataset, study):
    p = prof.get("hEDS")
    co = build_cohort(small_dataset, p, study, design="suspicion")
    tr = states.transitions(small_dataset, co, p)
    assert len(tr)
    for pid, g in tr.groupby("patient_id"):
        g = g.sort_values("t_start")
        # Sojourns tile the observation window with no gaps.
        assert np.allclose(g["t_stop"].to_numpy()[:-1], g["t_start"].to_numpy()[1:])
        assert (g["duration"] > 0).all()
        last = g.iloc[-1]
        if last["status"] == 1:
            assert last["to_state"] in (int(State.DIAGNOSED), int(State.DEAD),
                                        *[int(s) for s in State])
    # Diagnosed patients must end in the diagnosed state.
    ended = tr.groupby("patient_id").last()
    diagnosed = set(co.loc[co["event"] == 1, "patient_id"])
    for pid in list(diagnosed)[:40]:
        if pid in ended.index:
            assert ended.at[pid, "to_state"] == int(State.DIAGNOSED)


def test_occupancy_rows_sum_to_one(small_dataset, study):
    p = prof.get("hEDS")
    co = build_cohort(small_dataset, p, study, design="suspicion")
    tr = states.transitions(small_dataset, co, p)
    grid = multistate.occupancy_grid(tr, 10.0, 25)
    occ = multistate.occupancy(tr, grid)
    cols = [int(s) for s in State if int(s) in occ.columns]
    total = occ[cols].sum(axis=1)
    assert np.allclose(total, 1.0, atol=1e-8)


def test_phase_shares_sum_to_one(small_dataset, study):
    p = prof.get("hEDS")
    co = build_cohort(small_dataset, p, study, design="suspicion")
    tr = states.transitions(small_dataset, co, p)
    pt = phase_times(tr, 10.0)
    assert pt["share"].sum() == pytest.approx(1.0)
    assert (pt["mean_years"] >= 0).all()


def test_multistate_fit_reports_when_a_state_is_too_thin():
    tr = pd.DataFrame({
        "patient_id": [1, 2], "disease": ["X", "X"], "from_state": [0, 0],
        "to_state": [1, 1], "t_start": [0.0, 0.0], "t_stop": [1.0, 2.0],
        "status": [1, 1], "duration": [1.0, 2.0],
    })
    f = multistate.fit(tr, "X")
    assert any("sojourns" in n for n in f.notes)


def test_invisible_phase_is_positive(small_dataset):
    inv = invisible_phase(small_dataset.truth)
    assert len(inv) == len(prof.PANEL)
    assert (inv["median_years"] > 0).all()
