"""Run configuration and canonical paths.

Everything downstream reads paths from here so that a run against real claims
only has to change ``RDD_DATA`` / ``RDD_OUT`` rather than any analysis code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(os.environ.get("RDD_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("RDD_DATA", ROOT / "data"))
RAW = DATA / "raw"
INTERIM = DATA / "interim"
OUT = Path(os.environ.get("RDD_OUT", ROOT / "outputs"))
FIGDIR = OUT / "figures"
TABDIR = OUT / "tables"

for _p in (RAW, INTERIM, OUT, FIGDIR, TABDIR):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------
# Internally every date is an integer count of days since EPOCH. Claims spans
# are decades long and we do a lot of interval arithmetic; integers are both
# faster and immune to the timezone/dtype accidents that come with datetime64.

EPOCH = date(2000, 1, 1)
DAYS_PER_YEAR = 365.25


def to_day(d: date) -> int:
    return (d - EPOCH).days


def to_date(day: int | float) -> date:
    return EPOCH + timedelta(days=int(day))


def years(days: float):
    """Days -> years. Works on scalars and arrays."""
    return days / DAYS_PER_YEAR


def days_of(y: float):
    return y * DAYS_PER_YEAR


# ---------------------------------------------------------------------------
# Study design
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StudyConfig:
    """Design parameters of the *observational study*, not of the simulator.

    These are the knobs a claims analyst actually controls. Keeping them in one
    frozen object is what makes the lookback-window bias experiment in
    :mod:`rdd.bias` a one-line sweep instead of a rewrite.
    """

    #: First and last day for which the data source has any claims at all.
    db_start: int = to_day(date(2008, 1, 1))
    db_end: int = to_day(date(2023, 12, 31))

    #: Required days of continuous enrolment *before* the index diagnosis for a
    #: case to be called incident rather than prevalent. This is the single
    #: most consequential design choice in the whole study.
    lookback_days: int = int(5 * DAYS_PER_YEAR)

    #: Allowed gap in enrolment before we treat coverage as broken.
    max_enrolment_gap_days: int = 45

    #: Claims phenotype: how many qualifying diagnosis claims, how far apart,
    #: and within what span. The span requirement is what stops a single
    #: rule-out code years earlier from being adopted as the index date.
    min_dx_claims: int = 2
    min_dx_claim_spacing_days: int = 30
    max_dx_claim_span_days: int = 365

    #: Suspicion-cohort entry: how many signature claims, over how many
    #: distinct codes, within what window. Requiring two distinct codes is what
    #: keeps the risk set from collapsing into "everyone who ever had back
    #: pain" for the diseases whose signature overlaps common morbidity.
    signature_min_claims: int = 3
    signature_min_distinct: int = 2
    signature_window_days: int = 730

    #: A single diagnosis claim also qualifies if paired with a confirmatory
    #: procedure (enzyme assay, genetic panel) within this window.
    confirm_window_days: int = 180

    #: Horizon for restricted mean sojourn times, in years.
    rmst_horizon_years: float = 15.0

    def __post_init__(self):
        if self.db_end <= self.db_start:
            raise ValueError("db_end must follow db_start")
        if self.min_dx_claims < 1:
            raise ValueError("min_dx_claims must be >= 1")


DEFAULT_STUDY = StudyConfig()


@dataclass(frozen=True)
class SimConfig:
    """Simulator size and sampling knobs."""

    n_cases_per_disease: int = 1200
    n_controls: int = 6000
    seed: int = 20260813
    #: Fraction of the population that churns out of the plan each year.
    annual_disenrolment: float = 0.14
    #: Fraction enrolled already at db_start (rest enter later).
    prevalent_enrolment: float = 0.55
    diseases: tuple[str, ...] = field(default=())


DEFAULT_SIM = SimConfig()
