"""Generic two-file adapter for a licensed extract.

The contract is deliberately small, because the point of friction with a
licensed extract (MarketScan, Optum, a state APCD) is never the exotic columns
-- it is getting agreement on patient id, date, code and specialty. Everything
else in this package is downstream of those four.

Expected input
--------------
``claims.csv``
    ``patient_id, claim_id, service_date, code, code_type, position,
    specialty, setting``
``enrolment.csv``
    ``patient_id, enrol_start, enrol_end, birth_date, sex[, death_date]``

Dates may be ISO strings or integers. Specialty values are mapped through
:data:`SPECIALTY_MAP`; anything unrecognised becomes ``OTHER`` and is counted
in the returned notes, because a mapping that silently dumps 40% of encounters
into ``OTHER`` will quietly destroy the multi-state model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import EPOCH
from ..schema import SETTINGS, SPECIALTIES, ClaimsDataset

#: Common source spellings -> canonical specialty buckets. Extend for your
#: source's taxonomy; this covers CMS provider specialty descriptions and the
#: usual commercial-extract spellings.
SPECIALTY_MAP: dict[str, str] = {
    "family practice": "PRIMARY_CARE",
    "general practice": "PRIMARY_CARE",
    "internal medicine": "PRIMARY_CARE",
    "pediatric medicine": "PRIMARY_CARE",
    "pediatrics": "PRIMARY_CARE",
    "geriatric medicine": "PRIMARY_CARE",
    "nurse practitioner": "PRIMARY_CARE",
    "physician assistant": "PRIMARY_CARE",
    "emergency medicine": "EMERGENCY",
    "allergy/immunology": "ALLERGY_IMMUNOLOGY",
    "allergy and immunology": "ALLERGY_IMMUNOLOGY",
    "cardiology": "CARDIOLOGY",
    "cardiovascular disease": "CARDIOLOGY",
    "dermatology": "DERMATOLOGY",
    "gastroenterology": "GASTROENTEROLOGY",
    "medical genetics": "GENETICS",
    "clinical genetics": "GENETICS",
    "genetic counseling": "GENETICS",
    "hematology": "HEMATOLOGY",
    "hematology/oncology": "HEMATOLOGY",
    "medical oncology": "HEMATOLOGY",
    "nephrology": "NEPHROLOGY",
    "neurology": "NEUROLOGY",
    "orthopedic surgery": "ORTHOPEDICS",
    "orthopaedics": "ORTHOPEDICS",
    "physical medicine and rehabilitation": "ORTHOPEDICS",
    "psychiatry": "PSYCHIATRY",
    "psychology": "PSYCHIATRY",
    "pulmonary disease": "PULMONOLOGY",
    "pulmonology": "PULMONOLOGY",
    "rheumatology": "RHEUMATOLOGY",
    "general surgery": "SURGERY",
    "surgery": "SURGERY",
}

SETTING_MAP: dict[str, str] = {
    "11": "OFFICE", "office": "OFFICE",
    "23": "ER", "emergency": "ER", "emergency room": "ER",
    "21": "INPATIENT", "inpatient": "INPATIENT",
    "22": "OUTPATIENT", "outpatient": "OUTPATIENT",
    "81": "LAB", "lab": "LAB", "laboratory": "LAB",
}


def _to_day(series: pd.Series) -> pd.Series:
    """Dates -> integer days since EPOCH, accepting strings or integers."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("int64")
    dt = pd.to_datetime(series, errors="coerce")
    return (dt - pd.Timestamp(EPOCH)).dt.days.astype("Int64").astype("int64")


def map_specialty(values: pd.Series) -> tuple[pd.Series, float]:
    v = values.fillna("").astype(str).str.strip().str.lower()
    mapped = v.map(SPECIALTY_MAP)
    upper = v.str.upper()
    mapped = mapped.fillna(upper.where(upper.isin(SPECIALTIES)))
    unmapped = mapped.isna().mean()
    return mapped.fillna("OTHER"), float(unmapped)


def map_setting(values: pd.Series) -> pd.Series:
    v = values.fillna("").astype(str).str.strip().str.lower()
    mapped = v.map(SETTING_MAP)
    upper = v.str.upper()
    mapped = mapped.fillna(upper.where(upper.isin(SETTINGS)))
    return mapped.fillna("OFFICE")


def load_generic(directory: str | Path) -> tuple[ClaimsDataset, list[str]]:
    directory = Path(directory)
    notes: list[str] = []
    claims = pd.read_csv(directory / "claims.csv")
    enrol = pd.read_csv(directory / "enrolment.csv")

    claims["day"] = _to_day(claims["service_date"])
    claims["specialty"], unmapped = map_specialty(claims["specialty"])
    if unmapped > 0.10:
        notes.append(
            f"{unmapped:.1%} of encounters had an unmapped specialty and became "
            "OTHER; extend SPECIALTY_MAP before trusting the multi-state fit"
        )
    claims["setting"] = map_setting(claims.get("setting", pd.Series("OFFICE", index=claims.index)))
    claims["code"] = claims["code"].astype(str).str.strip().str.upper()
    if "position" not in claims:
        claims["position"] = 1
        notes.append("no diagnosis position column; all codes treated as principal")

    enrol["enrol_start"] = _to_day(enrol["enrol_start"])
    enrol["enrol_end"] = _to_day(enrol["enrol_end"])
    enrol["birth_day"] = _to_day(enrol["birth_date"])
    if "death_date" in enrol:
        enrol["death_day"] = _to_day(enrol["death_date"]).astype(float)
        enrol.loc[enrol["death_day"] < -30000, "death_day"] = np.nan
    else:
        enrol["death_day"] = np.nan
        notes.append(
            "no death date in source; death cannot be separated from plan "
            "churn, so competing-risk estimates fall back to plain censoring"
        )

    ds = ClaimsDataset(claims=claims, enrolment=enrol, source=f"generic:{directory}")
    return ds, notes
