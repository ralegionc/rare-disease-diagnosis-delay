"""Reconstructing the odyssey as a state process from claim lines.

The simulator has a state machine; claims do not. What claims have is a
sequence of encounters with a specialty, a setting and some codes, and the
analyst's job is to infer which phase of the odyssey each encounter belongs to.
This module does that inference, and it is deliberately the only place in the
package that knows how to.

States
------
======================  ====================================================
``UNDIFFERENTIATED``    symptomatic; primary care or ED only; no active label
``SPECIALIST``          under a specialist who cannot make this diagnosis
``MISLABELLED``         carrying an active differential-set diagnosis
``CORRECT_SPECIALTY``   under a specialty that could make the diagnosis
``DIAGNOSED``           absorbing
``DEAD``                absorbing
======================  ====================================================

Two honest caveats, both consequences of billing data rather than of this code:

*The process is only observed at encounters.* A transition is dated to the
first encounter that reveals it, so every sojourn is interval-censored with
interval width equal to the gap between visits. For a patient seen six times a
year that is a ~2-month resolution; the recovery check in :mod:`rdd.validate`
measures what it costs.

*"Mislabelled" is inferred from code persistence, not from intent.* A
differential code appearing once is a clinician thinking out loud. We require
it to recur, and we let it persist for :data:`LABEL_PERSISTENCE_DAYS` past its
last appearance, because a label that is working generates no claims -- the
absence of billing is what a settled wrong diagnosis looks like.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import pandas as pd

from .config import DAYS_PER_YEAR
from .profiles import DiseaseProfile

#: How long a wrong label is assumed to keep operating after its last claim.
LABEL_PERSISTENCE_DAYS = 180

#: A differential code must appear this many times to count as a label.
LABEL_MIN_CLAIMS = 2

NON_SPECIALIST = {"PRIMARY_CARE", "EMERGENCY", "OTHER"}


class State(IntEnum):
    UNDIFFERENTIATED = 0
    SPECIALIST = 1
    MISLABELLED = 2
    CORRECT_SPECIALTY = 3
    DIAGNOSED = 4
    DEAD = 5


PRE_DIAGNOSIS_STATES = (
    State.UNDIFFERENTIATED,
    State.SPECIALIST,
    State.MISLABELLED,
    State.CORRECT_SPECIALTY,
)

STATE_LABELS = {
    State.UNDIFFERENTIATED: "Undifferentiated care",
    State.SPECIALIST: "Non-diagnostic specialist",
    State.MISLABELLED: "Carrying a wrong label",
    State.CORRECT_SPECIALTY: "At the right specialty",
    State.DIAGNOSED: "Diagnosed",
    State.DEAD: "Died undiagnosed",
}


def _label_intervals(dx: pd.DataFrame, profile: DiseaseProfile, hi: float):
    """Intervals during which the patient is carrying a wrong label."""
    misdx = set(profile.misdx_codes)
    rows = dx[dx["code"].isin(misdx)]
    out = []
    for _, g in rows.groupby("code", sort=False):
        days = np.sort(g["day"].unique())
        if days.size < LABEL_MIN_CLAIMS:
            continue
        out.append((float(days[0]), min(float(days[-1]) + LABEL_PERSISTENCE_DAYS, hi)))
    return out


def _in_any(t: float, intervals) -> bool:
    return any(lo <= t <= hi for lo, hi in intervals)


def encounter_states(
    claims: pd.DataFrame, profile: DiseaseProfile, t0: float, t_exit: float
) -> pd.DataFrame:
    """Assign a state to every encounter in ``[t0, t_exit]``."""
    w = claims[(claims["day"] >= t0) & (claims["day"] <= t_exit)]
    if w.empty:
        return pd.DataFrame(columns=["day", "state"])
    dx = w[w["code_type"] == "DX"]
    intervals = _label_intervals(dx, profile, t_exit)

    enc = (
        w.groupby("claim_id")
        .agg(day=("day", "first"), specialty=("specialty", "first"))
        .sort_values("day")
    )
    states = []
    for day, spec in zip(enc["day"], enc["specialty"]):
        if _in_any(float(day), intervals):
            s = State.MISLABELLED
        elif spec in profile.recognizing_specialties:
            s = State.CORRECT_SPECIALTY
        elif spec not in NON_SPECIALIST:
            s = State.SPECIALIST
        else:
            s = State.UNDIFFERENTIATED
        states.append(int(s))
    return pd.DataFrame({"day": enc["day"].to_numpy(float), "state": states})


def transitions(
    dataset, cohort: pd.DataFrame, profile: DiseaseProfile
) -> pd.DataFrame:
    """Build the long-format transition table the multi-state model consumes.

    One row per sojourn: ``patient_id, from_state, to_state, t_start, t_stop,
    status``. ``status`` is 1 when the sojourn ended in an observed transition
    and 0 when it ended in censoring, so every downstream estimator can treat
    this as an ordinary right-censored survival dataset per origin state.
    """
    claims = dataset.claims
    grouped = dict(
        tuple(claims[claims["patient_id"].isin(cohort["patient_id"])].groupby("patient_id"))
    )
    rows = []
    for r in cohort.itertuples():
        sub = grouped.get(r.patient_id)
        if sub is None:
            continue
        t_exit_day = r.onset_proxy_day + r.exit * DAYS_PER_YEAR
        es = encounter_states(sub, profile, r.onset_proxy_day, t_exit_day)
        if es.empty:
            continue

        # Collapse consecutive encounters sharing a state into one sojourn.
        seq_day = [float(es["day"].iloc[0])]
        seq_state = [int(es["state"].iloc[0])]
        for d, s in zip(es["day"].iloc[1:], es["state"].iloc[1:]):
            if int(s) != seq_state[-1]:
                seq_day.append(float(d))
                seq_state.append(int(s))

        terminal = (
            int(State.DIAGNOSED) if r.cause == 1
            else int(State.DEAD) if r.cause == 2
            else None
        )
        for i, (d, s) in enumerate(zip(seq_day, seq_state)):
            if i + 1 < len(seq_day):
                stop, to_state, status = seq_day[i + 1], seq_state[i + 1], 1
            else:
                stop = t_exit_day
                to_state = terminal
                status = 1 if terminal is not None else 0
            rows.append(
                (
                    r.patient_id,
                    profile.key,
                    s,
                    to_state if to_state is not None else -1,
                    (d - r.onset_proxy_day) / DAYS_PER_YEAR,
                    (stop - r.onset_proxy_day) / DAYS_PER_YEAR,
                    status,
                )
            )

    trans = pd.DataFrame(
        rows,
        columns=[
            "patient_id", "disease", "from_state", "to_state",
            "t_start", "t_stop", "status",
        ],
    )
    if trans.empty:
        return trans
    trans["duration"] = (trans["t_stop"] - trans["t_start"]).clip(lower=1e-4)
    return trans


def state_at(trans: pd.DataFrame, t: float) -> pd.Series:
    """State of each patient at odyssey-time ``t``; NaN if out of observation."""
    hit = trans[(trans["t_start"] <= t) & (trans["t_stop"] > t)]
    return hit.set_index("patient_id")["from_state"]
