"""Three biases that decide the answer before any modelling happens.

Every number this package produces is an estimate of a latent quantity from a
proxy, under censoring, in a database with finite history. Each of those three
facts bends the estimate in a known direction, and all three bend it the same
way -- toward *understating* diagnostic delay. That coherence is the reason
published claims estimates of the diagnostic odyssey should be read as lower
bounds, and it is the main thing this module exists to demonstrate.

1. The look-back ceiling
------------------------
A database with :math:`W` years of history cannot show a delay longer than
:math:`W`. As :math:`W` shrinks the estimated median does not become noisy, it
becomes *wrong in a specific direction*: it saturates just below :math:`W`.
:func:`lookback_sweep` traces the saturation curve. The practical consequence
is that two studies of the same disease in databases of different vintage will
disagree, and the one with the shorter history will report the shorter delay --
which will look like an improvement in care over time.

2. Conditioning on being diagnosed
----------------------------------
The retrospective design keeps only patients who were eventually diagnosed
inside the window. Long odysseys are systematically the ones that run off the
end of the data, so removing them removes the right tail.
:func:`design_comparison` puts the retrospective estimate, the censoring-aware
prospective estimate, and ground truth side by side.

3. Treating death as censoring
------------------------------
For a disease with substantial mortality before diagnosis, ``1 - KM`` answers
"what fraction would be diagnosed if nobody died", which is not a question
anyone asked. :func:`competing_risk_comparison` contrasts it with the
Aalen-Johansen cumulative incidence.

There is a trap inside this one worth stating explicitly, because it inverts
the intuition. The disease with the *highest* mortality in the panel -- ALS --
shows the *smallest* competing-risk correction in the analysed data, while
late-onset Pompe shows a correction several times larger. The reason is that
ALS kills fast enough that most patients who would have died undiagnosed never
accumulate a symptom signature and so never enter the cohort at all. Mortality
has already done its selecting upstream of the estimator, where no
competing-risk method can see it. A small Aalen-Johansen correction is
therefore not reassurance that death did not matter; it can equally mean death
mattered so much that the affected patients are missing from the denominator.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .cohort import build_cohort
from .config import DAYS_PER_YEAR, DEFAULT_STUDY, StudyConfig
from .profiles import DiseaseProfile
from .schema import ClaimsDataset
from .survival import aalen_johansen, kaplan_meier, quantile


# ---------------------------------------------------------------------------
# 1. Look-back ceiling
# ---------------------------------------------------------------------------


def truncate_history(ds: ClaimsDataset, years: float, study: StudyConfig) -> tuple:
    """Emulate a database that only holds ``years`` of history."""
    cutoff = study.db_end - years * DAYS_PER_YEAR
    claims = ds.claims[ds.claims["day"] >= cutoff].copy()
    enrol = ds.enrolment.copy()
    enrol["enrol_start"] = np.maximum(enrol["enrol_start"], int(cutoff))
    enrol = enrol[enrol["enrol_end"] > enrol["enrol_start"]]
    claims = claims[claims["patient_id"].isin(enrol["patient_id"])]
    trimmed = ClaimsDataset(
        claims=claims, enrolment=enrol, truth=ds.truth,
        source=f"{ds.source} | history={years}y",
    )
    trimmed_study = replace(
        study,
        db_start=int(cutoff),
        lookback_days=int(min(study.lookback_days, max(years - 1, 0.5) * DAYS_PER_YEAR)),
    )
    return trimmed, trimmed_study


def lookback_sweep(
    ds: ClaimsDataset,
    profile: DiseaseProfile,
    windows=(3, 5, 8, 11, 14, 16),
    study: StudyConfig = DEFAULT_STUDY,
) -> pd.DataFrame:
    """Estimated median delay as a function of available database history."""
    rows = []
    for w in windows:
        sub, sub_study = truncate_history(ds, w, study)
        try:
            co = build_cohort(sub, profile, sub_study, design="diagnosed")
        except Exception:  # pragma: no cover - degenerate truncation
            co = pd.DataFrame()
        if co.empty:
            rows.append({"history_years": w, "n": 0, "median_delay": np.nan,
                         "p90_delay": np.nan})
            continue
        rows.append(
            {
                "history_years": w,
                "n": int(len(co)),
                "median_delay": float(co["exit"].median()),
                "p90_delay": float(co["exit"].quantile(0.90)),
                "max_delay": float(co["exit"].max()),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Design comparison
# ---------------------------------------------------------------------------


def true_median_delay(truth: pd.DataFrame, disease: str) -> float:
    t = truth[(truth["disease"] == disease) & np.isfinite(truth["dx_day"])]
    if t.empty:
        return np.nan
    return float(np.median((t["dx_day"] - t["onset_day"]) / DAYS_PER_YEAR))


def true_median_delay_from_first_care(truth: pd.DataFrame, disease: str) -> float:
    """Ground truth on the clock a claims study can actually start."""
    t = truth[(truth["disease"] == disease) & np.isfinite(truth["dx_day"])]
    if t.empty:
        return np.nan
    return float(np.median((t["dx_day"] - t["first_care_day"]) / DAYS_PER_YEAR))


def design_comparison(
    ds: ClaimsDataset, profile: DiseaseProfile, study: StudyConfig = DEFAULT_STUDY
) -> dict:
    """Retrospective vs prospective vs truth, for one disease."""
    dxc = build_cohort(ds, profile, study, design="diagnosed")
    spc = build_cohort(ds, profile, study, design="suspicion")

    out = {
        "disease": profile.key,
        "anchor_years": profile.target_median_delay_years,
        "n_diagnosed_design": int(len(dxc)),
        "n_suspicion_design": int(len(spc)),
    }
    out["retrospective_median"] = float(dxc["exit"].median()) if len(dxc) else np.nan

    if len(spc):
        km = kaplan_meier(spc["exit"].to_numpy(), spc["event"].to_numpy())
        est, lo, hi = quantile(km, 0.5)
        out["prospective_km_median"] = est
        out["prospective_km_lo"] = lo
        out["prospective_km_hi"] = hi
        out["prospective_event_frac"] = float(spc["event"].mean())
    else:
        out.update(
            prospective_km_median=np.nan, prospective_km_lo=np.nan,
            prospective_km_hi=np.nan, prospective_event_frac=np.nan,
        )

    if ds.has_truth:
        out["true_median_from_onset"] = true_median_delay(ds.truth, profile.key)
        out["true_median_from_first_care"] = true_median_delay_from_first_care(
            ds.truth, profile.key
        )
        # Restricted to the patients the retrospective design actually kept --
        # this isolates selection from measurement error.
        if len(dxc):
            t = ds.truth[ds.truth["patient_id"].isin(dxc["patient_id"])]
            t = t[np.isfinite(t["dx_day"])]
            out["true_median_in_analysed_sample"] = (
                float(np.median((t["dx_day"] - t["onset_day"]) / DAYS_PER_YEAR))
                if len(t) else np.nan
            )
    return out


def design_comparison_panel(ds, profiles, study=DEFAULT_STUDY) -> pd.DataFrame:
    rows = [design_comparison(ds, p, study) for p in profiles]
    df = pd.DataFrame(rows)
    if "true_median_from_onset" in df:
        df["retrospective_bias"] = df["retrospective_median"] - df["true_median_from_onset"]
        df["prospective_bias"] = df["prospective_km_median"] - df["true_median_from_onset"]
    return df


# ---------------------------------------------------------------------------
# 3. Competing risks
# ---------------------------------------------------------------------------


def competing_risk_comparison(
    ds: ClaimsDataset, profile: DiseaseProfile, study: StudyConfig = DEFAULT_STUDY,
    horizon: float = 8.0,
) -> pd.DataFrame:
    """``1 - KM`` versus Aalen-Johansen for the probability of being diagnosed."""
    co = build_cohort(ds, profile, study, design="suspicion")
    if co.empty:
        return pd.DataFrame()

    km = kaplan_meier(co["exit"].to_numpy(), co["event"].to_numpy())
    t_aj, cifs = aalen_johansen(
        co["exit"].to_numpy(), co["cause"].to_numpy(), causes=(1, 2)
    )
    grid = np.linspace(0, horizon, 80)
    naive = np.array([1.0 - km.at(t) for t in grid])

    def step(t_ref, y, t):
        idx = np.searchsorted(t_ref, t, side="right") - 1
        return 0.0 if idx < 0 else float(y[idx])

    aj_dx = np.array([step(t_aj, cifs.get(1, np.zeros_like(t_aj)), t) for t in grid])
    aj_death = np.array([step(t_aj, cifs.get(2, np.zeros_like(t_aj)), t) for t in grid])
    return pd.DataFrame(
        {
            "t": grid,
            "naive_1_minus_km": naive,
            "aj_diagnosed": aj_dx,
            "aj_died_undiagnosed": aj_death,
            "overstatement": naive - aj_dx,
        }
    )
