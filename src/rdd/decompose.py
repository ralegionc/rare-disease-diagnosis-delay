"""Where the years actually go.

"Patients wait 5-7 years" is not an actionable finding, because it does not say
which part of the pathway to intervene on. This module splits the observable
delay into phases and reports each phase's share.

The decomposition is an occupancy integral. For a horizon :math:`\\tau`,

.. math::  E[\\text{time in state } s \\text{ before } \\tau] = \\int_0^\\tau P_s(t)\\,dt

and because the state space is exhaustive, the phase times sum to :math:`\\tau`.
Restricting to the pre-diagnosis states and renormalising gives each phase's
share of the delay. Using an occupancy integral rather than per-patient totals
is what makes this work under censoring: a patient followed for two years
contributes to :math:`P_s(t)` for :math:`t < 2` and simply drops out after,
instead of contributing a spuriously short total.

The invisible phase
-------------------
One phase cannot be decomposed from claims at all: symptom onset to first care
contact. No claim is generated while a patient is unwell and has not presented,
so this interval is structurally unobservable in administrative data --
not merely noisy, *absent*. :func:`invisible_phase` measures it against the
simulator's ground truth, and its size is the honest upper bound on what any
claims-only study of diagnostic delay can ever see.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DAYS_PER_YEAR
from .multistate import occupancy, occupancy_grid
from .states import PRE_DIAGNOSIS_STATES, STATE_LABELS, State


def phase_times(trans: pd.DataFrame, horizon: float, n_grid: int = 241) -> pd.DataFrame:
    """Restricted mean time in each pre-diagnosis state, up to ``horizon``."""
    if trans.empty:
        return pd.DataFrame()
    grid = occupancy_grid(trans, horizon, n_grid)
    occ = occupancy(trans, grid)
    if occ.empty:
        return pd.DataFrame()

    t = occ["t"].to_numpy()
    dt = np.diff(t, prepend=t[0])
    rows = []
    for s in PRE_DIAGNOSIS_STATES:
        col = int(s)
        area = float(np.sum(occ[col].to_numpy() * dt)) if col in occ else 0.0
        rows.append(
            {
                "state": col,
                "phase": STATE_LABELS[State(col)],
                "mean_years": area,
            }
        )
    out = pd.DataFrame(rows)
    total = out["mean_years"].sum()
    out["share"] = out["mean_years"] / total if total > 0 else np.nan
    out["horizon"] = horizon
    out["min_n_observed"] = int(occ["n_observed"].min())
    return out


def decompose_panel(trans_by_disease: dict, horizon: float) -> pd.DataFrame:
    frames = []
    for key, trans in trans_by_disease.items():
        pt = phase_times(trans, horizon)
        if pt.empty:
            continue
        pt.insert(0, "disease", key)
        frames.append(pt)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def invisible_phase(truth: pd.DataFrame, cohort: pd.DataFrame | None = None) -> pd.DataFrame:
    """Onset -> first care contact, measured from ground truth.

    Reported separately and clearly marked, because no estimator in this
    package could have produced it. It is the part of the odyssey that happens
    before the health system knows the patient exists.
    """
    t = truth[truth["disease"] != "NONE"].copy()
    if cohort is not None and len(cohort):
        t = t[t["patient_id"].isin(cohort["patient_id"])]
    t["patient_side_years"] = (t["first_care_day"] - t["onset_day"]) / DAYS_PER_YEAR
    g = t.groupby("disease")["patient_side_years"]
    return pd.DataFrame(
        {
            "n": g.size(),
            "median_years": g.median(),
            "p25": g.quantile(0.25),
            "p75": g.quantile(0.75),
        }
    ).reset_index()


def observed_vs_true_origin(truth: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """How late the claims-based onset proxy is, relative to true onset.

    This is the measurement-error term that propagates into every delay
    estimate in the literature. It has a *sign*: the proxy can never precede
    the first claim, so it is late-biased by construction, and every delay
    estimate built on it is short-biased by the same amount.
    """
    m = cohort.merge(
        truth[["patient_id", "onset_day", "first_care_day", "dx_day"]],
        on="patient_id",
        how="inner",
    )
    m = m[np.isfinite(m["onset_day"])]
    if m.empty:
        return pd.DataFrame()
    m["proxy_error_years"] = (m["onset_proxy_day"] - m["onset_day"]) / DAYS_PER_YEAR
    m["proxy_vs_first_care_years"] = (
        m["onset_proxy_day"] - m["first_care_day"]
    ) / DAYS_PER_YEAR
    g = m.groupby("disease")
    return pd.DataFrame(
        {
            "n": g.size(),
            "median_proxy_error_years": g["proxy_error_years"].median(),
            "p25": g["proxy_error_years"].quantile(0.25),
            "p75": g["proxy_error_years"].quantile(0.75),
            "median_vs_first_care": g["proxy_vs_first_care_years"].median(),
        }
    ).reset_index()
