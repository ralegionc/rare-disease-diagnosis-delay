"""Semi-Markov multi-state model of the odyssey.

Semi-Markov, not Markov, and the distinction is the substantive claim of the
model: the hazard of leaving a state depends on *how long you have been in it*,
not on how long ago your symptoms started. A patient who has carried a
fibromyalgia label for three years is not in the same position as one who
acquired it yesterday and happens to be at the same point on the odyssey clock.
Clock resets at every transition.

What comes out
--------------
``transition_counts``
    where patients actually go, which is worth looking at before any model.
``intensities``
    Nelson-Aalen cumulative transition intensities per (from, to) pair.
``sojourn_fits``
    parametric sojourn distributions per origin state, with Akaike weights over
    exponential / Weibull / log-normal. A Weibull shape below 1 in the
    ``MISLABELLED`` state is the quantitative signature of a wrong label
    settling in: the longer it has held, the less likely it is to be revised.
``occupancy``
    probability of being in each state as a function of odyssey time, which is
    what :mod:`rdd.decompose` integrates to apportion the delay.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .states import PRE_DIAGNOSIS_STATES, STATE_LABELS, State
from .survival import fit_parametric, nelson_aalen


@dataclass
class MultiStateFit:
    disease: str
    transition_counts: pd.DataFrame
    sojourn_fits: dict
    sojourn_weights: dict
    intensities: dict
    n_patients: int
    n_sojourns: int
    notes: list[str] = field(default_factory=list)

    def sojourn_table(self) -> pd.DataFrame:
        rows = []
        for s, fits in self.sojourn_fits.items():
            w = self.sojourn_weights[s]
            best = max(w, key=lambda f: w[f]) if w else None
            f = fits.get(best)
            rows.append(
                {
                    "state": int(s),
                    "state_label": STATE_LABELS[State(s)],
                    "n_sojourns": f.n if f else 0,
                    "n_ended": f.n_events if f else 0,
                    "best_family": best,
                    "akaike_weight": float(w.get(best, np.nan)) if best else np.nan,
                    "shape": float(f.shape) if f else np.nan,
                    "median_years": float(f.median()) if f else np.nan,
                }
            )
        return pd.DataFrame(rows).sort_values("state").reset_index(drop=True)


def transition_counts(trans: pd.DataFrame) -> pd.DataFrame:
    obs = trans[trans["status"] == 1]
    tab = (
        obs.groupby(["from_state", "to_state"])
        .size()
        .rename("n")
        .reset_index()
        .pivot(index="from_state", columns="to_state", values="n")
        .fillna(0)
        .astype(int)
    )
    return tab


def fit(trans: pd.DataFrame, disease: str = "") -> MultiStateFit:
    """Fit sojourn distributions and transition intensities."""
    if trans.empty:
        return MultiStateFit(disease, pd.DataFrame(), {}, {}, {}, 0, 0,
                             ["no transitions available"])

    fits, weights, intens, notes = {}, {}, {}, []
    for s in PRE_DIAGNOSIS_STATES:
        sub = trans[trans["from_state"] == int(s)]
        if len(sub) < 20:
            notes.append(f"state {int(s)} has only {len(sub)} sojourns; fit skipped")
            fits[int(s)], weights[int(s)] = {}, {}
            continue
        f, w, _ = fit_parametric(sub["duration"].to_numpy(), sub["status"].to_numpy())
        fits[int(s)], weights[int(s)] = f, w

        for to in sorted(sub.loc[sub["status"] == 1, "to_state"].unique()):
            e = ((sub["to_state"] == to) & (sub["status"] == 1)).astype(int).to_numpy()
            if e.sum() < 5:
                continue
            t, H, V = nelson_aalen(sub["duration"].to_numpy(), e)
            intens[(int(s), int(to))] = (t, H, V)

    return MultiStateFit(
        disease=disease,
        transition_counts=transition_counts(trans),
        sojourn_fits=fits,
        sojourn_weights=weights,
        intensities=intens,
        n_patients=int(trans["patient_id"].nunique()),
        n_sojourns=int(len(trans)),
        notes=notes,
    )


def occupancy(trans: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    """Probability of occupying each state at each time on ``grid``.

    Estimated as the fraction of patients *still under observation* who are in
    each state. Under independent censoring this is consistent, and unlike a
    naive fraction-of-everyone it does not drift toward whatever state the
    long-followed patients happen to be in.

    The denominator is reported alongside, because for rare diseases it
    collapses fast and any occupancy curve past the point where it falls below
    a few dozen patients should be read as decoration, not evidence.
    """
    if trans.empty:
        return pd.DataFrame()

    last = trans.groupby("patient_id").agg(
        t_end=("t_stop", "max"), status=("status", "last"), final=("to_state", "last")
    )
    rows = []
    for t in grid:
        active = trans[(trans["t_start"] <= t) & (trans["t_stop"] > t)]
        cur = active.groupby("patient_id")["from_state"].first()
        # Patients who have already reached an absorbing state stay there.
        done = last[(last["t_end"] <= t) & (last["status"] == 1)]
        absorbed = done.loc[done["final"].isin([int(State.DIAGNOSED), int(State.DEAD)]), "final"]
        at_risk = len(cur) + len(absorbed)
        if at_risk == 0:
            continue
        counts = pd.concat([cur, absorbed]).value_counts()
        row = {"t": float(t), "n_observed": int(at_risk)}
        for s in State:
            row[int(s)] = float(counts.get(int(s), 0)) / at_risk
        rows.append(row)
    return pd.DataFrame(rows)


def occupancy_grid(trans: pd.DataFrame, horizon: float, n: int = 121) -> np.ndarray:
    hi = min(horizon, float(trans["t_stop"].max())) if not trans.empty else horizon
    return np.linspace(0.0, max(hi, 1e-3), n)
