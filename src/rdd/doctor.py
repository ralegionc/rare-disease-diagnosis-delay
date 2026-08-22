"""Data-readiness diagnostics: can this database answer the question at all?

Run this *before* estimating anything on a new source. Every check here exists
because it is a failure mode that produces a plausible-looking number rather
than an error.

The four that actually bite:

**Vocabulary mismatch.** ``rdd.profiles`` is written in ICD-10-CM. Point the
pipeline at an ICD-9 source and the phenotype matches nothing, ``build_cohort``
returns an empty frame, and the report renders with blank tables. Nothing
raises. :func:`check_vocabulary` detects the mismatch and names the codes that
never appear.

**The look-back ceiling, checked in advance.** A database holding *W* years of
continuous enrolment cannot exhibit a delay longer than *W* -- and the estimate
does not degrade gracefully, it saturates just below *W* (see
``rdd.bias.lookback_sweep``). :func:`check_enrolment` measures the coverage
actually available and reports the ceiling, so you learn the answer is
unobtainable before you compute an answer.

**Specialty collapse.** The multi-state model distinguishes "at a specialist
who cannot diagnose this" from "at one who can". A source with no
rendering-provider specialty (SynPUF) or an unmapped taxonomy maps everything
to ``OTHER``, and the model silently degenerates while still producing a
decomposition chart.

**Cohort too small to mean anything.** Agreement with truth is strongly
sample-size dependent at the sizes rare-disease studies operate at
(``rdd.validate.SAMPLE_SIZE_NOTE``). Below roughly 60 analysable patients the
per-disease median is sampling noise wearing a confidence interval.

Nothing here reads the truth table, so it runs identically on real and
simulated data. It *does* read ``target_median_delay_years`` from the profiles,
which the estimation path must never do -- here it is used only to say "this
database is too short to see a delay of that size", which is a statement about
the database, not an estimate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import profiles as prof_mod
from .cohort import apply_phenotype, build_cohort, continuous_spans, signature_codes
from .config import DAYS_PER_YEAR, DEFAULT_STUDY, StudyConfig
from .schema import ClaimsDataset

#: Below this many phenotype-positive patients, a per-disease median delay is
#: not worth computing. See rdd.validate.SAMPLE_SIZE_NOTE.
MIN_USABLE_COHORT = 60

#: Above this share of encounters mapped to OTHER, the multi-state model cannot
#: distinguish the specialist states and its decomposition is meaningless.
MAX_OTHER_SPECIALTY = 0.40

ICD10_RE = re.compile(r"^[A-TV-Z][0-9][0-9AB]")
ICD9_RE = re.compile(r"^(\d{3}|[VE]\d{2})")


@dataclass
class DoctorReport:
    dataset: dict
    vocabulary: dict
    specialty: pd.DataFrame
    enrolment: dict
    per_disease: pd.DataFrame
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return not self.blockers

    def text(self) -> str:
        L = []
        a = L.append
        a("=" * 72)
        a("rdd doctor -- data readiness report")
        a("=" * 72)
        a("")
        a(f"source           {self.dataset.get('source')}")
        a(f"patients         {self.dataset.get('patients', 0):,}")
        a(f"claim lines      {self.dataset.get('claim_lines', 0):,}")
        a(f"distinct codes   {self.dataset.get('distinct_codes', 0):,}")
        a("")

        v = self.vocabulary
        a("-- vocabulary " + "-" * 58)
        a(f"looks like       {v['detected']}")
        a(f"profile codes present   {v['n_present']}/{v['n_expected']} "
          f"({v['coverage']:.0%})")
        if v["absent_dx_codes"]:
            a(f"diagnosis codes never seen: {', '.join(v['absent_dx_codes'])}")
        a("")

        a("-- provider specialty " + "-" * 50)
        if len(self.specialty):
            top = self.specialty.head(8)
            for _, r in top.iterrows():
                a(f"  {r['specialty']:<22} {r['share']:>6.1%}")
        a("")

        e = self.enrolment
        a("-- observable window " + "-" * 51)
        a(f"median continuous enrolment   {e['median_years']:.1f} yr")
        a(f"90th percentile               {e['p90_years']:.1f} yr")
        a(f"death date recorded for       {e['death_share']:.1%} of patients")
        a(f"=> delays beyond ~{e['median_years']:.0f} yr are structurally "
          "unobservable for a typical patient here")
        a("")

        a("-- per disease " + "-" * 57)
        if len(self.per_disease):
            a(f"  {'disease':<8}{'dx code':<10}{'w/ code':>9}{'phenotype':>11}"
              f"{'cohort':>8}  verdict")
            for _, r in self.per_disease.iterrows():
                a(f"  {r['disease']:<8}{r['dx_code']:<10}{r['n_with_code']:>9,}"
                  f"{r['n_phenotype']:>11,}{r['n_cohort']:>8,}  {r['verdict']}")
        a("")

        if self.blockers:
            a("BLOCKERS -- the pipeline cannot produce a valid estimate until "
              "these are fixed:")
            for b in self.blockers:
                a(f"  ! {b}")
            a("")
        if self.warnings:
            a("WARNINGS -- results will render but should be read with these "
              "in mind:")
            for w in self.warnings:
                a(f"  ~ {w}")
            a("")
        if not self.blockers and not self.warnings:
            a("No blockers or warnings. Proceed to `rdd analyse`.")
            a("")
        return "\n".join(L)


# ---------------------------------------------------------------------------


def check_vocabulary(ds: ClaimsDataset) -> dict:
    """Are the profiles' codes actually in this source's vocabulary?"""
    codes = set(ds.claims.loc[ds.claims["code_type"] == "DX", "code"].unique())
    sample = list(codes)[:4000]
    n10 = sum(1 for c in sample if ICD10_RE.match(str(c)))
    n9 = sum(1 for c in sample if ICD9_RE.match(str(c)))
    if not sample:
        detected = "no diagnosis codes at all"
    elif n10 > n9:
        detected = "ICD-10-CM"
    elif n9 > n10:
        detected = "ICD-9-CM (profiles are ICD-10 -- a crosswalk is required)"
    else:
        detected = "unrecognised diagnosis vocabulary"

    expected: set[str] = set()
    for p in prof_mod.PANEL:
        expected |= {p.dx_code} | signature_codes(p)
    present = expected & codes
    absent_dx = [p.dx_code for p in prof_mod.PANEL if p.dx_code not in codes]
    return {
        "detected": detected,
        "n_expected": len(expected),
        "n_present": len(present),
        "coverage": len(present) / len(expected) if expected else 0.0,
        "absent_dx_codes": absent_dx,
        "absent_codes": sorted(expected - codes),
    }


def check_specialty(ds: ClaimsDataset) -> pd.DataFrame:
    s = ds.claims.groupby("claim_id")["specialty"].first()
    tab = s.value_counts(normalize=True).rename("share").reset_index()
    tab.columns = ["specialty", "share"]
    return tab


def check_enrolment(ds: ClaimsDataset, study: StudyConfig = DEFAULT_STUDY) -> dict:
    spans = continuous_spans(ds.enrolment, study)
    yrs = (spans["obs_end"] - spans["obs_start"]) / DAYS_PER_YEAR
    death = ds.enrolment.drop_duplicates("patient_id")["death_day"]
    return {
        "median_years": float(yrs.median()) if len(yrs) else 0.0,
        "p90_years": float(yrs.quantile(0.90)) if len(yrs) else 0.0,
        "max_years": float(yrs.max()) if len(yrs) else 0.0,
        "death_share": float(death.notna().mean()) if len(death) else 0.0,
        "n_patients": int(len(spans)),
    }


def check_disease(
    ds: ClaimsDataset, profile, study: StudyConfig, enrol: dict, other_share: float
) -> dict:
    claims = ds.claims
    n_with_code = int(
        claims.loc[
            (claims["code"] == profile.dx_code) & (claims["code_type"] == "DX"),
            "patient_id",
        ].nunique()
    )
    pheno = apply_phenotype(ds, profile, study)
    n_pheno = int(len(pheno.index))
    try:
        cohort = build_cohort(ds, profile, study, design="diagnosed")
        n_cohort = int(len(cohort))
    except Exception:  # pragma: no cover - defensive on unfamiliar sources
        n_cohort = 0

    sig = signature_codes(profile)
    seen = set(claims.loc[claims["code_type"] == "DX", "code"].unique())
    sig_missing = sorted(sig - seen)

    anchor = profile.target_median_delay_years
    if n_with_code == 0:
        verdict = "BLOCKED: diagnosis code never appears"
    elif n_pheno == 0:
        verdict = "BLOCKED: no patient meets the case definition"
    elif n_cohort < MIN_USABLE_COHORT:
        verdict = f"BLOCKED: cohort {n_cohort} < {MIN_USABLE_COHORT} usable"
    elif enrol["median_years"] < anchor:
        verdict = (
            f"LIMITED: history {enrol['median_years']:.0f}yr < published "
            f"{anchor:.0f}yr delay"
        )
    elif other_share > MAX_OTHER_SPECIALTY:
        verdict = "LIMITED: specialty collapsed to OTHER"
    else:
        verdict = "OK"

    return {
        "disease": profile.key,
        "dx_code": profile.dx_code,
        "n_with_code": n_with_code,
        "n_phenotype": n_pheno,
        "n_cohort": n_cohort,
        "n_signature_codes_missing": len(sig_missing),
        "signature_codes_missing": ", ".join(sig_missing[:8]),
        "published_delay_years": anchor,
        "verdict": verdict,
    }


def diagnose(ds: ClaimsDataset, study: StudyConfig = DEFAULT_STUDY) -> DoctorReport:
    vocab = check_vocabulary(ds)
    spec = check_specialty(ds)
    enrol = check_enrolment(ds, study)

    other = float(spec.loc[spec["specialty"] == "OTHER", "share"].sum())
    rows = [check_disease(ds, p, study, enrol, other) for p in prof_mod.PANEL]
    per_disease = pd.DataFrame(rows)

    blockers, warnings = [], []

    if "crosswalk" in vocab["detected"]:
        blockers.append(
            "source appears to be ICD-9 while rdd.profiles is ICD-10-CM; map the "
            "profile code sets before anything downstream will match"
        )
    if vocab["coverage"] < 0.25:
        blockers.append(
            f"only {vocab['coverage']:.0%} of the profiles' codes appear in this "
            "source; the phenotype and the onset proxy will both mostly miss"
        )
    if all("BLOCKED" in v for v in per_disease["verdict"]):
        blockers.append("no disease in the panel is estimable in this source")

    if other > MAX_OTHER_SPECIALTY:
        warnings.append(
            f"{other:.0%} of encounters have specialty OTHER (limit "
            f"{MAX_OTHER_SPECIALTY:.0%}); the multi-state model cannot separate "
            "the specialist states -- extend adapters.generic.SPECIALTY_MAP"
        )
    if enrol["death_share"] == 0:
        warnings.append(
            "no death dates: competing-risk estimates degrade to plain "
            "censoring and 1-KM will overstate the probability of diagnosis"
        )
    short = per_disease[per_disease["verdict"].str.startswith("LIMITED: history")]
    if len(short):
        warnings.append(
            f"{len(short)} disease(s) have a published delay longer than the "
            f"median {enrol['median_years']:.0f}yr of continuous enrolment here; "
            "their estimates are bounded by the data, not by the disease"
        )

    return DoctorReport(
        dataset=ds.summary(),
        vocabulary=vocab,
        specialty=spec,
        enrolment=enrol,
        per_disease=per_disease,
        blockers=blockers,
        warnings=warnings,
    )
