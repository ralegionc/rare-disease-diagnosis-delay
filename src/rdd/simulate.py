"""Generative simulator for the diagnostic odyssey.

The simulator exists because the quantity of interest -- time from symptom
*onset* to correct diagnosis -- is never observed in claims. Claims record
billing, not biology: the earliest thing they can possibly show is the first
time the patient sought care and someone coded something. So every estimator in
this package is estimating a latent quantity from a proxy, and the only way to
know whether it works is to run it against data where the latent quantity is
known by construction.

Generative structure
--------------------
Per patient::

    birth -> onset -> [patient-side delay] -> first care contact
          -> {primary care <-> specialist} loop, punctuated by
             misdiagnosis labels that suppress further workup for a dwell time
          -> reaching a recognising specialty with enough specific evidence
          -> confirmatory test -> diagnosis
    with a competing exponential hazard of death throughout.

Two things make this more than a delay distribution with extra steps:

1. **Evidence accrues on the disease's own clock.** Each symptom has an
   ``onset_lag_years`` and a ``specificity``. Fabry's pathognomonic findings
   (angiokeratoma, corneal verticillata, LVH) arrive 3-15 years after the
   burning limb pain that first brings the patient in, so the model produces a
   long delay *without* anyone being told to make Fabry slow.

2. **Wrong labels are absorbing-ish, not just wasted visits.** A misdiagnosis
   explains the symptoms, so it suppresses diagnostic effort for its dwell
   time. This is what separates hEDS (fibromyalgia, 3.4-year dwell, assigned to
   half of patients) from ALS (cervical spondylosis, 0.55-year dwell, disproved
   by its own failed surgery).

Claims are then emitted from the pathway and **clipped to enrolment spans**,
which is what generates the left truncation and right censoring that the
analysis side has to cope with. The truth table keeps the unclipped timeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import profiles as prof_mod
from .config import DEFAULT_SIM, DEFAULT_STUDY, DAYS_PER_YEAR, SimConfig, StudyConfig
from .profiles import DiseaseProfile
from .schema import ClaimsDataset

# Per-visit probabilities governing pathway movement. These are *care-system*
# parameters, deliberately shared across diseases: variation between diseases
# should come from the profiles, not from bespoke routing rules.
P_REFERRAL_FROM_PRIMARY = 0.38
P_REFERRAL_FROM_SPECIALIST = 0.26
P_MISLABEL_ON_ENCOUNTER = 0.55
P_RETURN_TO_PRIMARY = 0.45
POST_DX_FOLLOWUP_YEARS = 3.0

#: Per-visit chance a recognising specialist codes the rare disease as a
#: hypothesis without diagnosing it, and the chance a control picks up a
#: spurious rare-disease code from a rule-out workup. Without these the
#: phenotype algorithm is being graded on an exam with no wrong answers.
P_RULE_OUT_CODING = 0.09
P_CONTROL_RULE_OUT = 0.010

EM_CODES_BY_SETTING = {
    "OFFICE": ("99213", "99214", "99204"),
    "ER": ("99283", "99284", "99285"),
    "INPATIENT": ("99222", "99232"),
    "OUTPATIENT": ("99215", "99244"),
    "LAB": ("80053", "85025"),
}


@dataclass
class Friction:
    """Per-disease calibration scalar.

    One number per disease, fitted by :func:`calibrate` so that the simulated
    median delay matches the literature anchor in the profile.

    It represents one construct -- *how readily the care system reaches for a
    wrong explanation instead of escalating* -- expressed through the three
    levers that construct actually operates on:

    ``mislabel``
        propensity to attach a wrong label at an encounter;
    ``dwell``
        how long that label survives before it stops explaining the symptoms;
    ``recognition``
        how readily the right specialist makes the call.

    Biology (symptom lags, specificity, mortality) is untouched. Keeping
    calibration to a single monotone scalar per disease is deliberate: a richer
    calibration would hit the anchors trivially and prove nothing.
    """

    value: float = 1.0

    def dwell(self, years: float) -> float:
        return years * self.value

    def recognition(self, p: float) -> float:
        # friction > 1 -> harder to recognise; friction < 1 -> easier.
        return float(np.clip(p ** self.value, 1e-4, 0.99))

    def mislabel(self, p: float = P_MISLABEL_ON_ENCOUNTER) -> float:
        return float(np.clip(p * self.value, 0.0, 0.90))


# ---------------------------------------------------------------------------
# Enrolment and demography
# ---------------------------------------------------------------------------


def _draw_enrolment(rng: np.random.Generator, n: int, study: StudyConfig):
    """Coverage spans. Plan churn is the origin of most missing follow-up."""
    span = study.db_end - study.db_start
    prevalent = rng.random(n) < DEFAULT_SIM.prevalent_enrolment
    start = np.where(
        prevalent,
        study.db_start,
        study.db_start + rng.integers(0, max(span - 365, 1), size=n),
    ).astype(np.int64)
    mean_years = 1.0 / DEFAULT_SIM.annual_disenrolment
    duration = rng.exponential(mean_years * DAYS_PER_YEAR, size=n)
    duration = np.clip(duration, 180.0, None)
    end = np.minimum(start + duration, study.db_end).astype(np.int64)
    return start, end


# ---------------------------------------------------------------------------
# One patient's odyssey
# ---------------------------------------------------------------------------


def _manifest_days(rng, profile: DiseaseProfile, onset_day: float):
    """Which symptoms this patient gets, and when each becomes billable."""
    out = []
    for s in profile.symptoms:
        if rng.random() >= s.prob:
            continue
        lag = s.onset_lag_years * float(np.exp(rng.normal(0.0, 0.40)))
        lag = max(lag, 0.0) + rng.exponential(0.05)
        out.append((onset_day + lag * DAYS_PER_YEAR, s))
    if not out:  # every case must have at least one manifestation
        s = max(profile.symptoms, key=lambda x: x.prob)
        out.append((onset_day + rng.exponential(20.0), s))
    out.sort(key=lambda t: t[0])
    return out


def _evidence(manifest, now: float) -> float:
    """Combined specificity of everything visible so far, as a noisy-OR."""
    q = 1.0
    for day, s in manifest:
        if day <= now:
            q *= 1.0 - s.specificity
    return 1.0 - q


def _visible_symptoms(manifest, now: float):
    return [s for day, s in manifest if day <= now]


def _pick_misdx(rng, profile: DiseaseProfile, specialty: str, used: set[str]):
    """Prefer a wrong label the current specialty would plausibly reach for."""
    pool = [m for m in profile.misdx if m.code not in used]
    if not pool:
        return None
    weights = np.array(
        [m.prob * (2.5 if m.specialty == specialty else 1.0) for m in pool], dtype=float
    )
    if weights.sum() <= 0:
        return None
    weights /= weights.sum()
    return pool[int(rng.choice(len(pool), p=weights))]


def simulate_patient(
    rng: np.random.Generator,
    patient_id: int,
    profile: DiseaseProfile,
    study: StudyConfig,
    friction: Friction,
    sim_horizon_day: float,
):
    """Simulate one case. Returns ``(encounters, truth)``.

    Encounters are emitted on the *true* timeline with no reference to
    enrolment; clipping happens later so that the truth table stays honest.
    """
    # --- demography and onset ------------------------------------------
    sex = "F" if rng.random() < profile.female_fraction else "M"
    onset_age = max(0.5, rng.normal(profile.onset_age_mean, profile.onset_age_sd))
    # Onset is spread well before the database window opens so that the cohort
    # contains genuinely left-truncated odysseys, not just tidy ones.
    onset_day = float(
        rng.uniform(study.db_start - 20 * DAYS_PER_YEAR, study.db_end)
    )
    birth_day = onset_day - onset_age * DAYS_PER_YEAR

    death_day = onset_day + rng.exponential(1.0 / profile.death_hazard) * DAYS_PER_YEAR
    horizon = min(sim_horizon_day, death_day)

    manifest = _manifest_days(rng, profile, onset_day)
    first_symptom_day = manifest[0][0]

    # --- patient-side delay --------------------------------------------
    care_delay = rng.lognormal(
        np.log(profile.care_seeking_median_years), profile.care_seeking_logsd
    )
    now = first_symptom_day + care_delay * DAYS_PER_YEAR
    first_care_day = now

    encounters: list[tuple] = []
    specialty = "PRIMARY_CARE"
    label = None  # active misdiagnosis
    label_expiry = -np.inf
    used_labels: set[str] = set()
    specialties_seen: set[str] = {"PRIMARY_CARE"}
    first_specialist_day = np.nan
    first_correct_specialty_day = np.nan
    dx_day = np.nan
    n_misdx = 0

    inter_visit_days = DAYS_PER_YEAR / max(profile.visit_rate, 0.1)
    rec_p = friction.recognition(profile.recognition_prob)
    mislabel_p = friction.mislabel()

    while now < horizon:
        visible = _visible_symptoms(manifest, now)
        if not visible:
            now += rng.exponential(inter_visit_days)
            continue

        # Which complaint drives this encounter, and hence where it happens.
        sym = visible[int(rng.integers(len(visible)))]
        setting = sym.settings[int(rng.integers(len(sym.settings)))]
        if setting == "ER":
            enc_specialty = "EMERGENCY"
        elif setting == "LAB":
            enc_specialty = specialty
        else:
            enc_specialty = specialty

        codes = [(s.code, "DX") for s in rng.choice(
            np.array(visible, dtype=object),
            size=min(len(visible), 1 + int(rng.integers(0, 3))),
            replace=False,
        )]
        if label is not None and now < label_expiry:
            codes.insert(0, (label.code, "DX"))
        em = EM_CODES_BY_SETTING[setting]
        codes.append((em[int(rng.integers(len(em)))], "PX"))
        encounters.append((now, enc_specialty, setting, tuple(codes)))

        if enc_specialty not in specialties_seen:
            specialties_seen.add(enc_specialty)
        if enc_specialty not in ("PRIMARY_CARE", "EMERGENCY") and np.isnan(
            first_specialist_day
        ):
            first_specialist_day = now
        if enc_specialty in profile.recognizing_specialties and np.isnan(
            first_correct_specialty_day
        ):
            first_correct_specialty_day = now

        # --- an active wrong label suppresses further diagnostic effort ---
        if label is not None and now < label_expiry:
            now += rng.exponential(inter_visit_days)
            continue
        if label is not None:
            label = None  # expired: the explanation stopped working

        # --- recognition ---------------------------------------------------
        if specialty in profile.recognizing_specialties:
            ev = _evidence(manifest, now)
            if rng.random() < P_RULE_OUT_CODING:
                # The specialist considered it and coded it, then moved on.
                # These "hypothesis" claims are ubiquitous in real data and are
                # precisely what the repeat-claim phenotype has to reject.
                encounters.append(
                    (now, specialty, "OFFICE", ((profile.dx_code, "DX"), ("99244", "PX")))
                )
            if rng.random() < np.clip(rec_p * (0.25 + 0.75 * ev), 0.0, 1.0):
                confirm = profile.confirm_codes[
                    int(rng.integers(len(profile.confirm_codes)))
                ]
                encounters.append((now, specialty, "LAB", ((confirm, "PX"),)))
                dx_day = now + rng.uniform(7.0, 90.0)
                break

        # --- otherwise: acquire a wrong label, or get referred onward -------
        if rng.random() < mislabel_p:
            cand = _pick_misdx(rng, profile, specialty, used_labels)
            if cand is not None:
                label = cand
                used_labels.add(cand.code)
                n_misdx += 1
                dwell = friction.dwell(cand.dwell_years) * float(
                    np.exp(rng.normal(0.0, 0.30))
                )
                label_expiry = now + dwell * DAYS_PER_YEAR
                px = [(p, "PX") for p in cand.procedures]
                if px:
                    encounters.append(
                        (
                            now + rng.uniform(1, 30),
                            cand.specialty,
                            "OUTPATIENT" if len(px) > 1 else "OFFICE",
                            tuple([(cand.code, "DX")] + px),
                        )
                    )
                    specialties_seen.add(cand.specialty)
        else:
            p_ref = (
                P_REFERRAL_FROM_PRIMARY
                if specialty == "PRIMARY_CARE"
                else P_REFERRAL_FROM_SPECIALIST
            )
            if rng.random() < p_ref:
                keys = list(profile.referral_mix)
                p = np.array([profile.referral_mix[k] for k in keys])
                specialty = keys[int(rng.choice(len(keys), p=p / p.sum()))]
            elif specialty != "PRIMARY_CARE" and rng.random() < P_RETURN_TO_PRIMARY:
                specialty = "PRIMARY_CARE"

        now += rng.exponential(inter_visit_days)

    # --- post-diagnosis follow-up (this is what makes the ">=2 claims"
    #     phenotype work; without it every case looks like a single stray code)
    if not np.isnan(dx_day) and dx_day < sim_horizon_day:
        t = dx_day
        stop = min(dx_day + POST_DX_FOLLOWUP_YEARS * DAYS_PER_YEAR, horizon)
        spec = profile.recognizing_specialties[0]
        while t < stop:
            encounters.append(
                (t, spec, "OFFICE", ((profile.dx_code, "DX"), ("99214", "PX")))
            )
            t += rng.exponential(2 * inter_visit_days)

    truth = {
        "patient_id": patient_id,
        "disease": profile.key,
        "onset_day": onset_day,
        "first_care_day": first_care_day,
        "first_specialist_day": first_specialist_day,
        "first_correct_specialty_day": first_correct_specialty_day,
        "dx_day": dx_day,
        "death_day": death_day if death_day < sim_horizon_day else np.nan,
        "n_misdx": n_misdx,
        "n_specialties": len(specialties_seen),
        "sex": sex,
        "birth_day": birth_day,
    }
    return encounters, truth


def simulate_control(
    rng: np.random.Generator, patient_id: int, study: StudyConfig
):
    """A patient with no rare disease.

    Controls are not padding. They carry the same non-specific labels the cases
    accumulate -- fibromyalgia, IBS, anxiety -- so the phenotype algorithm's
    specificity is measurable rather than assumed.
    """
    sex = "F" if rng.random() < 0.52 else "M"
    age = rng.uniform(0, 85)
    birth_day = study.db_start - age * DAYS_PER_YEAR
    encounters: list[tuple] = []

    codes, _, weights = zip(*prof_mod.BACKGROUND_CODES)
    w = np.array(weights, dtype=float)
    w /= w.sum()

    now = float(study.db_start)
    while now < study.db_end:
        cur_age = (now - birth_day) / DAYS_PER_YEAR
        rate = prof_mod.background_rate_for_age(cur_age)
        now += rng.exponential(DAYS_PER_YEAR / rate)
        if now >= study.db_end:
            break
        dx = codes[int(rng.choice(len(codes), p=w))]
        px = prof_mod.BACKGROUND_PROCEDURES[
            int(rng.integers(len(prof_mod.BACKGROUND_PROCEDURES)))
        ]
        setting = "ER" if rng.random() < 0.08 else "OFFICE"
        encounters.append(
            (now, "PRIMARY_CARE", setting, ((dx, "DX"), (px, "PX")))
        )

    # A realistic slice of the healthy population also carries the mimics.
    mimics = [
        ("M79.7", 0.030), ("F41.1", 0.055), ("K58.9", 0.035),
        ("R53.82", 0.012), ("M47.12", 0.020), ("D69.3", 0.004),
    ]
    for code, p in mimics:
        if rng.random() < p:
            t0 = rng.uniform(study.db_start, study.db_end)
            for _ in range(int(rng.integers(2, 7))):
                t0 += rng.exponential(150.0)
                if t0 > study.db_end:
                    break
                encounters.append(
                    (t0, "PRIMARY_CARE", "OFFICE", ((code, "DX"), ("99213", "PX")))
                )

    # Rule-out coding in the healthy population: a specialist entertains a rare
    # diagnosis, codes it once or twice, and it is never confirmed.
    if rng.random() < P_CONTROL_RULE_OUT:
        target = prof_mod.PANEL[int(rng.integers(len(prof_mod.PANEL)))]
        t0 = rng.uniform(study.db_start, study.db_end - 200)
        for _ in range(int(rng.integers(1, 3))):
            encounters.append(
                (
                    t0,
                    target.recognizing_specialties[0],
                    "OFFICE",
                    ((target.dx_code, "DX"), ("99244", "PX")),
                )
            )
            t0 += rng.uniform(10, 120)

    truth = {
        "patient_id": patient_id,
        "disease": "NONE",
        "onset_day": np.nan,
        "first_care_day": np.nan,
        "first_specialist_day": np.nan,
        "first_correct_specialty_day": np.nan,
        "dx_day": np.nan,
        "death_day": np.nan,
        "n_misdx": 0,
        "n_specialties": 1,
        "sex": sex,
        "birth_day": birth_day,
    }
    return encounters, truth


# ---------------------------------------------------------------------------
# Background morbidity for cases
# ---------------------------------------------------------------------------


def _background_encounters(rng, birth_day: float, lo: float, hi: float):
    codes, _, weights = zip(*prof_mod.BACKGROUND_CODES)
    w = np.array(weights, dtype=float)
    w /= w.sum()
    out = []
    now = lo
    while now < hi:
        age = (now - birth_day) / DAYS_PER_YEAR
        rate = prof_mod.background_rate_for_age(age)
        now += rng.exponential(DAYS_PER_YEAR / rate)
        if now >= hi:
            break
        dx = codes[int(rng.choice(len(codes), p=w))]
        px = prof_mod.BACKGROUND_PROCEDURES[
            int(rng.integers(len(prof_mod.BACKGROUND_PROCEDURES)))
        ]
        out.append((now, "PRIMARY_CARE", "OFFICE", ((dx, "DX"), (px, "PX"))))
    return out


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------


def simulate_population(
    sim: SimConfig = DEFAULT_SIM,
    study: StudyConfig = DEFAULT_STUDY,
    calibration: dict[str, float] | None = None,
    diseases: tuple[str, ...] | None = None,
) -> ClaimsDataset:
    """Build a full synthetic claims database."""
    rng = np.random.default_rng(sim.seed)
    calibration = calibration or {}
    keys = diseases or tuple(p.key for p in prof_mod.PANEL)
    sim_horizon = study.db_end + 15 * DAYS_PER_YEAR

    rows: list[tuple] = []
    truths: list[dict] = []
    pid = 0
    claim_id = 0

    for key in keys:
        profile = prof_mod.get(key)
        fr = Friction(calibration.get(key, 1.0))
        for _ in range(sim.n_cases_per_disease):
            enc, truth = simulate_patient(rng, pid, profile, study, fr, sim_horizon)
            enc = enc + _background_encounters(
                rng, truth["birth_day"], float(study.db_start), float(study.db_end)
            )
            for day, spec, setting, codes in enc:
                claim_id += 1
                for pos, (code, ctype) in enumerate(codes, start=1):
                    rows.append((pid, claim_id, day, code, ctype, pos, spec, setting))
            truths.append(truth)
            pid += 1

    for _ in range(sim.n_controls):
        enc, truth = simulate_control(rng, pid, study)
        for day, spec, setting, codes in enc:
            claim_id += 1
            for pos, (code, ctype) in enumerate(codes, start=1):
                rows.append((pid, claim_id, day, code, ctype, pos, spec, setting))
        truths.append(truth)
        pid += 1

    claims = pd.DataFrame(
        rows,
        columns=[
            "patient_id", "claim_id", "day", "code", "code_type",
            "position", "specialty", "setting",
        ],
    )
    truth_df = pd.DataFrame(truths)

    # --- enrolment, then clip claims to it -----------------------------
    n = len(truth_df)
    start, end = _draw_enrolment(rng, n, study)
    enrolment = pd.DataFrame(
        {
            "patient_id": truth_df["patient_id"].to_numpy(),
            "enrol_start": start,
            "enrol_end": end,
            "birth_day": np.floor(truth_df["birth_day"].to_numpy()).astype(np.int64),
            "sex": truth_df["sex"].to_numpy(),
            "death_day": np.nan,
        }
    )
    # Death truncates coverage, and is recorded only when it falls inside the
    # observation window -- a patient who dies after disenrolling is invisible.
    dead = truth_df["death_day"].to_numpy()
    has_death = ~np.isnan(dead)
    # Compare on whole days throughout: enrol_end is an integer column, so a
    # float death day of 8123.7 truncates to 8123 and would then fail a
    # `death <= enrol_end` test against itself, silently discarding every
    # death record and turning the competing risk into ordinary censoring.
    dead_day = np.floor(dead)
    enrolment.loc[has_death, "enrol_end"] = np.minimum(
        enrolment.loc[has_death, "enrol_end"].to_numpy(), dead_day[has_death]
    ).astype(np.int64)
    enrolment["enrol_end"] = enrolment["enrol_end"].astype(np.int64)
    observed_death = (
        has_death
        & (dead_day <= enrolment["enrol_end"].to_numpy())
        & (dead_day >= enrolment["enrol_start"].to_numpy())
    )
    enrolment.loc[observed_death, "death_day"] = dead_day[observed_death]
    enrolment = enrolment[enrolment["enrol_end"] > enrolment["enrol_start"]]

    claims["day"] = np.floor(claims["day"]).astype(np.int64)
    span = enrolment.set_index("patient_id")[["enrol_start", "enrol_end"]]
    claims = claims.join(span, on="patient_id", how="inner")
    keep = (claims["day"] >= claims["enrol_start"]) & (
        claims["day"] <= claims["enrol_end"]
    )
    claims = claims.loc[keep].drop(columns=["enrol_start", "enrol_end"])

    truth_df = truth_df.drop(columns=["sex", "birth_day"])
    return ClaimsDataset(
        claims=claims,
        enrolment=enrolment.reset_index(drop=True),
        truth=truth_df,
        source=f"simulator v0.1 seed={sim.seed} n={n}",
    )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def _median_true_delay(key: str, friction: float, study: StudyConfig, n: int, seed: int):
    """True (unclipped, uncensored) median delay at a given friction."""
    rng = np.random.default_rng(seed)
    profile = prof_mod.get(key)
    fr = Friction(friction)
    horizon = study.db_end + 40 * DAYS_PER_YEAR
    delays = []
    for i in range(n):
        _, truth = simulate_patient(rng, i, profile, study, fr, horizon)
        if not np.isnan(truth["dx_day"]):
            delays.append((truth["dx_day"] - truth["onset_day"]) / DAYS_PER_YEAR)
    if not delays:
        return np.inf
    return float(np.median(delays))


def calibrate(
    keys: tuple[str, ...] | None = None,
    study: StudyConfig = DEFAULT_STUDY,
    n: int = 500,
    seed: int = 7,
    tol: float = 0.08,
    max_iter: int = 22,
) -> dict[str, float]:
    """Fit one friction scalar per disease to its literature anchor.

    Bisection on a monotone scalar. Monotonicity is what makes this safe: more
    friction means longer-lived wrong labels and lower recognition, and both
    push delay up, so there is exactly one root.

    Note the target is the *true* median delay from onset -- the anchor from the
    literature -- not anything the pipeline estimates. Calibration therefore
    cannot flatter the estimators; it only sets the ground truth they are
    scored against.
    """
    keys = keys or tuple(p.key for p in prof_mod.PANEL)
    out: dict[str, float] = {}
    for key in keys:
        target = prof_mod.get(key).target_median_delay_years
        lo, hi = 0.05, 6.0
        best, best_err = 1.0, np.inf
        for i in range(max_iter):
            mid = 0.5 * (lo + hi)
            got = _median_true_delay(key, mid, study, n, seed + i)
            err = abs(got - target) / target
            if err < best_err:
                best, best_err = mid, err
            if err < tol:
                break
            if got < target:
                lo = mid
            else:
                hi = mid
        out[key] = round(best, 4)
    return out


def load_calibration(path: str | Path) -> dict[str, float]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_calibration(cal: dict[str, float], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cal, indent=2, sort_keys=True), encoding="utf-8")


def with_seed(sim: SimConfig, seed: int) -> SimConfig:
    return replace(sim, seed=seed)
