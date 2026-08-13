"""Cohort construction: turning claim lines into an analysable odyssey.

Two study designs are implemented, because the choice between them is the
single largest fork in the published literature and they do not agree.

**Diagnosed-anchored (retrospective).** Find patients who eventually carry the
diagnosis, then look backwards for the first symptom claim. This is what almost
every published claims study of diagnostic delay does. It is also
systematically optimistic, for two compounding reasons: nobody who is *never*
diagnosed is in the denominator, and the look-back can only reach as far as the
patient's enrolment record, so a delay longer than the available history is
structurally unobservable. :mod:`rdd.bias` measures both.

**Suspicion-anchored (prospective).** Define entry by a symptom signature, then
follow forward to diagnosis or censoring. Never-diagnosed patients contribute
censored person-time, so the estimator is not conditioned on the outcome. The
price is contamination: phenocopies -- people with fibromyalgia who do not have
hEDS -- enter the risk set and dilute the curve.

Neither design is correct. The useful move is to run both, understand the sign
of each bias, and report the pair as bounds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DAYS_PER_YEAR, DEFAULT_STUDY, StudyConfig
from .profiles import DiseaseProfile
from .schema import ClaimsDataset

#: A symptom claim landing this close to the start of coverage almost certainly
#: is not the first one that ever happened.
LEFT_TRUNCATION_GRACE_DAYS = 180

NON_SPECIALIST = {"PRIMARY_CARE", "EMERGENCY", "OTHER"}


# ---------------------------------------------------------------------------
# Enrolment
# ---------------------------------------------------------------------------


def continuous_spans(enrolment: pd.DataFrame, study: StudyConfig = DEFAULT_STUDY) -> pd.DataFrame:
    """Collapse coverage into the longest continuous span per patient.

    Short coverage gaps (a job change, a retroactive reinstatement) are bridged;
    real gaps are not, because a claim cannot appear during one and an absent
    diagnosis during a gap is not evidence of absent disease.
    """
    df = enrolment.sort_values(["patient_id", "enrol_start"])
    out = []
    for pid, grp in df.groupby("patient_id", sort=False):
        cur_start = cur_end = None
        spans = []
        for s, e in zip(grp["enrol_start"], grp["enrol_end"]):
            if cur_start is None:
                cur_start, cur_end = s, e
            elif s - cur_end <= study.max_enrolment_gap_days:
                cur_end = max(cur_end, e)
            else:
                spans.append((cur_start, cur_end))
                cur_start, cur_end = s, e
        if cur_start is not None:
            spans.append((cur_start, cur_end))
        best = max(spans, key=lambda t: t[1] - t[0])
        out.append((pid, best[0], best[1]))
    return pd.DataFrame(out, columns=["patient_id", "obs_start", "obs_end"])


# ---------------------------------------------------------------------------
# Phenotype
# ---------------------------------------------------------------------------


def _qualifying_index(days: np.ndarray, study: StudyConfig):
    """Earliest day starting a group that satisfies the repeat-diagnosis rule.

    Returns ``None`` when no group qualifies. The scan is greedy inside each
    candidate window, which is exact here: to maximise the count of claims
    pairwise ``>= spacing`` apart within a fixed window, always take the
    earliest admissible next claim.
    """
    for i, start in enumerate(days):
        window = days[(days >= start) & (days - start <= study.max_dx_claim_span_days)]
        count, anchor = 1, window[0]
        for d in window[1:]:
            if d - anchor >= study.min_dx_claim_spacing_days:
                count += 1
                anchor = d
        if count >= study.min_dx_claims:
            return int(start)
    return None


@dataclass
class Phenotype:
    """Result of applying the claims case definition."""

    index: pd.DataFrame  # patient_id, index_day, rule
    n_screened: int

    def ids(self) -> np.ndarray:
        return self.index["patient_id"].to_numpy()


def apply_phenotype(
    dataset: ClaimsDataset, profile: DiseaseProfile, study: StudyConfig = DEFAULT_STUDY
) -> Phenotype:
    """Identify cases and their index date.

    Rule (the standard rare-disease claims definition):

    * ``>=min_dx_claims`` claims carrying the diagnosis code, at least
      ``min_dx_claim_spacing_days`` apart and all within
      ``max_dx_claim_span_days``; **or**
    * one such claim plus a confirmatory procedure within
      ``confirm_window_days``.

    Two details in that rule do real work.

    The index date is the **first claim of the qualifying group**, not the claim
    that satisfies the rule. Using the confirming claim would shift every index
    date later by the spacing requirement and inflate delay by construction --
    a mistake that is invisible once made, because nothing downstream can
    detect a uniformly shifted origin.

    The span requirement exists because rare-disease codes appear in claims as
    *hypotheses* long before they appear as diagnoses. A neurologist coding
    G12.21 once while ruling ALS out, four years before the real diagnosis,
    will otherwise anchor the index date on the rule-out and hand you four
    spurious years of "early diagnosis". Requiring the qualifying claims to
    cluster inside a year rejects that; a lone rule-out code cannot recruit a
    partner from years later.
    """
    claims = dataset.claims
    dx = claims.loc[
        (claims["code"] == profile.dx_code) & (claims["code_type"] == "DX"),
        ["patient_id", "day"],
    ]
    if dx.empty:
        return Phenotype(
            pd.DataFrame(columns=["patient_id", "index_day", "rule"]),
            dataset.n_patients,
        )

    confirm = claims.loc[
        claims["code"].isin(profile.confirm_codes) & (claims["code_type"] == "PX"),
        ["patient_id", "day"],
    ]
    confirm_by_pt = confirm.groupby("patient_id")["day"].apply(np.array).to_dict()

    rows = []
    for pid, grp in dx.groupby("patient_id", sort=False):
        days = np.sort(grp["day"].unique())
        index_day = _qualifying_index(days, study)
        rule = "repeat_dx"
        if index_day is None:
            c = confirm_by_pt.get(pid)
            hit = None
            if c is not None:
                for d in days:
                    if np.any(np.abs(c - d) <= study.confirm_window_days):
                        hit = int(d)
                        break
            if hit is None:
                continue
            index_day, rule = hit, "dx_plus_confirmation"
        rows.append((pid, int(index_day), rule))

    return Phenotype(
        pd.DataFrame(rows, columns=["patient_id", "index_day", "rule"]),
        dataset.n_patients,
    )


# ---------------------------------------------------------------------------
# Onset proxy
# ---------------------------------------------------------------------------


def signature_codes(profile: DiseaseProfile) -> set[str]:
    """Codes that count as 'this odyssey is underway'.

    Deliberately includes the misdiagnosis set. For hypermobile EDS the
    fibromyalgia claim *is* the odyssey -- excluding it would place the origin
    years late and hide most of the delay inside the pre-cohort period. The
    cost is contamination in the suspicion design, which is measured rather
    than assumed away.
    """
    return set(profile.symptom_codes) | set(profile.misdx_codes)


def onset_proxy(
    days: np.ndarray,
    codes: np.ndarray | None = None,
    study: StudyConfig = DEFAULT_STUDY,
) -> float:
    """First day of a *sustained, multi-code* symptom signature.

    ``days``/``codes`` are the (day-sorted) signature-set claims. We return the
    earliest day from which at least ``signature_min_claims`` such claims,
    spanning at least ``signature_min_distinct`` distinct codes, fall inside
    ``signature_window_days``.

    Both requirements matter and they guard different failure modes.
    Persistence stops the origin being anchored on a single incidental code:
    one fatigue claim in 2009 does not mean the odyssey began in 2009.
    Requiring two *distinct* codes stops the risk set from collapsing into
    "everyone who has ever had back pain" -- the signature sets for hEDS,
    Gaucher and Fabry all contain high-prevalence codes (fatigue, joint pain,
    abdominal pain), and without this the suspicion cohort recruits most of the
    control population and the estimated curve measures nothing.

    Returns ``nan`` when no qualifying signature exists.
    """
    if days.size == 0:
        return np.nan
    if codes is None:
        codes = np.arange(days.size).astype(str)
    need_n = study.signature_min_claims
    need_k = study.signature_min_distinct
    window = study.signature_window_days

    j = 0
    for i in range(days.size):
        while j < days.size and days[j] - days[i] <= window:
            j += 1
        if j - i >= need_n and len(set(codes[i:j])) >= need_k:
            return float(days[i])
    return np.nan


# ---------------------------------------------------------------------------
# Pathway features between origin and index
# ---------------------------------------------------------------------------


def _pathway_features(sub: pd.DataFrame, profile: DiseaseProfile, t0: float, t1: float):
    """Observable odyssey features in the interval ``[t0, t1)``."""
    w = sub[(sub["day"] >= t0) & (sub["day"] < t1)]
    if w.empty:
        return dict(
            n_misdx_obs=0, n_specialists_obs=0, n_encounters=0, n_er=0,
            first_specialist_day=np.nan, first_recognizing_day=np.nan,
        )
    misdx = set(profile.misdx_codes)
    dxw = w[w["code_type"] == "DX"]
    # A wrong label counts only if it recurs -- a single coded differential is
    # a clinician thinking out loud, not a label the patient carried.
    counts = dxw.loc[dxw["code"].isin(misdx), "code"].value_counts()
    n_misdx = int((counts >= 2).sum())

    specs = w["specialty"].dropna()
    specialists = set(specs.unique()) - NON_SPECIALIST
    spec_rows = w[~w["specialty"].isin(NON_SPECIALIST)]
    rec_rows = w[w["specialty"].isin(profile.recognizing_specialties)]
    return dict(
        n_misdx_obs=n_misdx,
        n_specialists_obs=len(specialists),
        n_encounters=int(w["claim_id"].nunique()),
        n_er=int(w.loc[w["setting"] == "ER", "claim_id"].nunique()),
        first_specialist_day=float(spec_rows["day"].min()) if len(spec_rows) else np.nan,
        first_recognizing_day=float(rec_rows["day"].min()) if len(rec_rows) else np.nan,
    )


# ---------------------------------------------------------------------------
# Cohort builders
# ---------------------------------------------------------------------------


def build_cohort(
    dataset: ClaimsDataset,
    profile: DiseaseProfile,
    study: StudyConfig = DEFAULT_STUDY,
    design: str = "suspicion",
) -> pd.DataFrame:
    """Build an analysis cohort.

    ``design`` is ``"suspicion"`` (prospective, censored) or ``"diagnosed"``
    (retrospective, complete cases only).

    Returned columns use a common survival contract so that the same
    estimators run on either design: ``entry`` and ``exit`` are years on the
    odyssey clock (origin = onset proxy), ``event`` is 1 for diagnosis, and
    ``cause`` is 0 censored / 1 diagnosed / 2 died undiagnosed.
    """
    if design not in {"suspicion", "diagnosed"}:
        raise ValueError("design must be 'suspicion' or 'diagnosed'")

    spans = continuous_spans(dataset.enrolment, study).set_index("patient_id")
    death = dataset.enrolment.set_index("patient_id")["death_day"]
    death = death[~death.index.duplicated(keep="first")]
    demo = dataset.enrolment.drop_duplicates("patient_id").set_index("patient_id")

    pheno = apply_phenotype(dataset, profile, study)
    index_by_pt = dict(zip(pheno.index["patient_id"], pheno.index["index_day"]))

    sig = signature_codes(profile)
    claims = dataset.claims
    sig_claims = (
        claims.loc[
            (claims["code_type"] == "DX") & (claims["code"].isin(sig)),
            ["patient_id", "day", "code"],
        ]
        .drop_duplicates()
        .sort_values(["patient_id", "day"])
    )
    sig_by_pt = {
        pid: (g["day"].to_numpy(float), g["code"].to_numpy(object))
        for pid, g in sig_claims.groupby("patient_id", sort=False)
    }

    if design == "diagnosed":
        candidates = list(index_by_pt)
    else:
        candidates = list(sig_by_pt)

    rows = []
    grouped = dict(tuple(claims[claims["patient_id"].isin(candidates)].groupby("patient_id")))

    for pid in candidates:
        if pid not in spans.index:
            continue
        obs_start = float(spans.at[pid, "obs_start"])
        obs_end = float(spans.at[pid, "obs_end"])
        entry = sig_by_pt.get(pid)
        if entry is None:
            continue
        days, codes = entry
        keep = (days >= obs_start) & (days <= obs_end)
        days, codes = days[keep], codes[keep]
        if days.size == 0:
            continue

        index_day = index_by_pt.get(pid, np.nan)
        if design == "diagnosed":
            keep = days <= index_day
            days, codes = days[keep], codes[keep]
            if days.size == 0:
                continue
        t0 = onset_proxy(days, codes, study)
        if not np.isfinite(t0):
            # Diagnosed cases with no qualifying prior signature still count --
            # they were diagnosed, we simply cannot date their onset. Dropping
            # them silently would select against short, clean odysseys.
            if design == "diagnosed":
                t0 = float(days[0])
            else:
                continue

        # Structural: coverage that started essentially at the same time as the
        # first symptom claim is a look-back ceiling, not an onset.
        left_trunc = bool(t0 - obs_start < LEFT_TRUNCATION_GRACE_DAYS)
        lookback_available = t0 - obs_start

        d_day = float(death.get(pid, np.nan))
        diagnosed = np.isfinite(index_day) and index_day >= t0
        if diagnosed:
            exit_day, cause = float(index_day), 1
        elif np.isfinite(d_day) and d_day <= obs_end:
            exit_day, cause = d_day, 2
        else:
            exit_day, cause = obs_end, 0

        if exit_day < t0:
            continue

        sub = grouped.get(pid)
        feats = _pathway_features(sub, profile, t0, exit_day + 1)

        birth = float(demo.at[pid, "birth_day"])
        rows.append(
            {
                "patient_id": pid,
                "disease": profile.key,
                "design": design,
                "onset_proxy_day": t0,
                "index_day": index_day if diagnosed else np.nan,
                "obs_start": obs_start,
                "obs_end": obs_end,
                "entry": 0.0,
                "exit": (exit_day - t0) / DAYS_PER_YEAR,
                "event": int(cause == 1),
                "cause": cause,
                "left_truncated": left_trunc,
                "lookback_years": lookback_available / DAYS_PER_YEAR,
                "age_at_proxy": (t0 - birth) / DAYS_PER_YEAR,
                "sex": demo.at[pid, "sex"],
                "time_to_specialist": (feats["first_specialist_day"] - t0) / DAYS_PER_YEAR,
                "time_to_recognizing": (feats["first_recognizing_day"] - t0) / DAYS_PER_YEAR,
                **{k: feats[k] for k in ("n_misdx_obs", "n_specialists_obs",
                                         "n_encounters", "n_er")},
            }
        )

    cohort = pd.DataFrame(rows)
    if cohort.empty:
        return cohort

    if design == "diagnosed":
        # Incident-case requirement: enough clean history before index that a
        # pre-existing diagnosis would have been visible. Prevalent cases carry
        # an index date that is an artefact of when their record starts.
        keep = (cohort["index_day"] - cohort["obs_start"]) >= study.lookback_days
        cohort = cohort.loc[keep].reset_index(drop=True)

    return cohort.reset_index(drop=True)


def summarise_cohort(cohort: pd.DataFrame) -> dict:
    if cohort.empty:
        return {"n": 0}
    return {
        "n": int(len(cohort)),
        "n_diagnosed": int(cohort["event"].sum()),
        "n_died_undiagnosed": int((cohort["cause"] == 2).sum()),
        "n_censored": int((cohort["cause"] == 0).sum()),
        "pct_left_truncated": float(100 * cohort["left_truncated"].mean()),
        "median_followup_years": float(cohort["exit"].median()),
        "median_lookback_years": float(cohort["lookback_years"].median()),
    }
