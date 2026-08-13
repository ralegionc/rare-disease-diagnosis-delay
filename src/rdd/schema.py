"""Canonical claims schema.

Every data source -- the simulator, CMS DE-SynPUF, an OMOP CDM instance, a
licensed extract -- is normalised into these three frames before any analysis
code touches it. Analysis modules import *only* from here, which is why the
same pipeline runs on synthetic and real data without branching.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# ---------------------------------------------------------------------------
# Column contracts
# ---------------------------------------------------------------------------

CLAIMS_COLUMNS: dict[str, str] = {
    "patient_id": "int64",
    "claim_id": "int64",
    "day": "int32",  # days since rdd.config.EPOCH
    "code": "string",  # ICD-10-CM or CPT/HCPCS
    "code_type": "string",  # DX | PX
    "position": "int8",  # 1 = principal diagnosis
    "specialty": "string",  # rendering-provider specialty bucket
    "setting": "string",  # OFFICE | ER | INPATIENT | OUTPATIENT | LAB
}

ENROLMENT_COLUMNS: dict[str, str] = {
    "patient_id": "int64",
    "enrol_start": "int32",
    "enrol_end": "int32",
    "birth_day": "int32",
    "sex": "string",  # F | M
    # Date of death where the source records one (Medicare and most licensed
    # extracts do; NaN where it does not). Without this, death is
    # indistinguishable from plan churn and the competing-risks analysis
    # silently degrades into ordinary censoring.
    "death_day": "float64",
}

# Simulator-only. Never available for real data; used purely to score the
# estimators. Anything that reads this frame lives under rdd.validate.
TRUTH_COLUMNS: dict[str, str] = {
    "patient_id": "int64",
    "disease": "string",
    "onset_day": "float64",
    "first_care_day": "float64",
    "first_specialist_day": "float64",
    "first_correct_specialty_day": "float64",
    "dx_day": "float64",
    "death_day": "float64",
    "n_misdx": "int16",
    "n_specialties": "int16",
}

SETTINGS = ("OFFICE", "ER", "INPATIENT", "OUTPATIENT", "LAB")

SPECIALTIES = (
    "PRIMARY_CARE",
    "EMERGENCY",
    "ALLERGY_IMMUNOLOGY",
    "CARDIOLOGY",
    "DERMATOLOGY",
    "GASTROENTEROLOGY",
    "GENETICS",
    "HEMATOLOGY",
    "NEPHROLOGY",
    "NEUROLOGY",
    "ORTHOPEDICS",
    "PSYCHIATRY",
    "PULMONOLOGY",
    "RHEUMATOLOGY",
    "SURGERY",
    "OTHER",
)


class SchemaError(ValueError):
    pass


def _validate(df: pd.DataFrame, spec: dict[str, str], name: str) -> pd.DataFrame:
    missing = [c for c in spec if c not in df.columns]
    if missing:
        raise SchemaError(f"{name} is missing required columns: {missing}")
    out = df.loc[:, list(spec)].copy()
    for col, dtype in spec.items():
        try:
            out[col] = out[col].astype(dtype)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise SchemaError(f"{name}.{col} cannot be cast to {dtype}: {exc}") from exc
    return out


def validate_claims(df: pd.DataFrame) -> pd.DataFrame:
    out = _validate(df, CLAIMS_COLUMNS, "claims")
    bad_type = set(out["code_type"].dropna().unique()) - {"DX", "PX"}
    if bad_type:
        raise SchemaError(f"claims.code_type has unexpected values: {sorted(bad_type)}")
    bad_setting = set(out["setting"].dropna().unique()) - set(SETTINGS)
    if bad_setting:
        raise SchemaError(f"claims.setting has unexpected values: {sorted(bad_setting)}")
    return out.sort_values(["patient_id", "day", "position"]).reset_index(drop=True)


def validate_enrolment(df: pd.DataFrame) -> pd.DataFrame:
    out = _validate(df, ENROLMENT_COLUMNS, "enrolment")
    if (out["enrol_end"] < out["enrol_start"]).any():
        raise SchemaError("enrolment has spans ending before they start")
    return out.sort_values(["patient_id", "enrol_start"]).reset_index(drop=True)


def validate_truth(df: pd.DataFrame) -> pd.DataFrame:
    return _validate(df, TRUTH_COLUMNS, "truth")


@dataclass
class ClaimsDataset:
    """The unit every adapter returns and every analysis stage consumes."""

    claims: pd.DataFrame
    enrolment: pd.DataFrame
    truth: pd.DataFrame | None = None
    source: str = "unknown"

    def __post_init__(self):
        self.claims = validate_claims(self.claims)
        self.enrolment = validate_enrolment(self.enrolment)
        if self.truth is not None:
            self.truth = validate_truth(self.truth)

    @property
    def has_truth(self) -> bool:
        return self.truth is not None

    @property
    def n_patients(self) -> int:
        return int(self.enrolment["patient_id"].nunique())

    def write(self, directory) -> None:
        from pathlib import Path

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.claims.to_parquet(directory / "claims.parquet", index=False)
        self.enrolment.to_parquet(directory / "enrolment.parquet", index=False)
        if self.truth is not None:
            self.truth.to_parquet(directory / "truth.parquet", index=False)
        (directory / "SOURCE.txt").write_text(self.source, encoding="utf-8")

    @classmethod
    def read(cls, directory) -> "ClaimsDataset":
        from pathlib import Path

        directory = Path(directory)
        truth_path = directory / "truth.parquet"
        src_path = directory / "SOURCE.txt"
        return cls(
            claims=pd.read_parquet(directory / "claims.parquet"),
            enrolment=pd.read_parquet(directory / "enrolment.parquet"),
            truth=pd.read_parquet(truth_path) if truth_path.exists() else None,
            source=src_path.read_text(encoding="utf-8") if src_path.exists() else "unknown",
        )

    def summary(self) -> dict:
        return {
            "source": self.source,
            "patients": self.n_patients,
            "claim_lines": int(len(self.claims)),
            "distinct_codes": int(self.claims["code"].nunique()),
            "first_day": int(self.claims["day"].min()),
            "last_day": int(self.claims["day"].max()),
            "has_truth": self.has_truth,
        }
