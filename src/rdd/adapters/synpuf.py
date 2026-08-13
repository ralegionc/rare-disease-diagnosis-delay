"""CMS DE-SynPUF adapter.

The Data Entrepreneurs' Synthetic Public Use File is the only genuinely public
dataset with real Medicare claims *structure*. It is free, needs no DUA, and is
the right thing to point this pipeline at first when you want to prove the
ingest path works on something you did not write yourself.

It is also, for this specific question, close to useless as evidence, and the
adapter says so up front rather than letting a reader assume otherwise:

* **Three years of history (2008-2010).** Section 1 of :mod:`rdd.bias` shows
  what a three-year look-back does to a delay estimate. Any delay past three
  years is unobservable by construction.
* **Medicare, so ~65+.** Five of the six panel diseases have onset decades
  earlier. Their odysseys happened before the data starts.
* **Synthesised per-variable to protect privacy**, which breaks longitudinal
  coherence -- the property this entire analysis depends on. A patient's code
  sequence in SynPUF is not a trajectory.
* **No rendering-provider specialty** in the beneficiary-summary-linked files,
  so ``SPECIALIST`` versus ``CORRECT_SPECIALTY`` cannot be distinguished and
  the multi-state model collapses to three states.

Use it to smoke-test ingest at real volume. Do not use it to estimate a delay.

Expected layout (as distributed by CMS)::

    DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv
    DE1_0_2008_to_2010_Carrier_Claims_Sample_1A.csv
    DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv
    DE1_0_2008_to_2010_Outpatient_Claims_Sample_1.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import EPOCH
from ..schema import ClaimsDataset

DX_PREFIXES = ("ICD9_DGNS_CD", "ICD9_PRCDR_CD", "LINE_ICD9_DGNS_CD")
HCPCS_PREFIX = "HCPCS_CD"


def _yyyymmdd(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    return (dt - pd.Timestamp(EPOCH)).dt.days


def _melt_codes(df: pd.DataFrame, id_cols, prefixes, code_type, setting, specialty):
    cols = [c for c in df.columns if any(c.startswith(p) for p in prefixes)]
    if not cols:
        return pd.DataFrame()
    long = df.melt(id_vars=id_cols, value_vars=cols, var_name="slot", value_name="code")
    long = long[long["code"].notna() & (long["code"].astype(str).str.strip() != "")]
    long["code"] = long["code"].astype(str).str.strip().str.upper()
    long["code_type"] = code_type
    long["setting"] = setting
    long["specialty"] = specialty
    long["position"] = (
        long["slot"].str.extract(r"(\d+)$").fillna(1).astype(int).clip(1, 127)
    )
    return long.drop(columns=["slot"])


def load_synpuf(directory: str | Path, sample: str = "1") -> tuple[ClaimsDataset, list[str]]:
    directory = Path(directory)
    notes = [
        "SynPUF covers 2008-2010 only: delays beyond ~3 years are structurally "
        "unobservable (see rdd.bias.lookback_sweep)",
        "SynPUF is a Medicare sample (~65+); onset for most panel diseases "
        "predates the observation window",
        "SynPUF is synthesised per-variable, so within-patient code sequences "
        "are not real trajectories",
        "no rendering-provider specialty available: all encounters mapped to "
        "OTHER, which collapses the multi-state model to three states",
        "codes are ICD-9; profile code sets in rdd.profiles are ICD-10 and need "
        "a crosswalk before the phenotype will fire",
    ]

    bene_files = sorted(directory.glob(f"*Beneficiary_Summary_File_Sample_{sample}*.csv"))
    if not bene_files:
        raise FileNotFoundError(f"no beneficiary summary file in {directory}")
    bene = pd.concat([pd.read_csv(f, low_memory=False) for f in bene_files])
    bene = bene.drop_duplicates("DESYNPUF_ID", keep="first")

    ids = pd.Index(bene["DESYNPUF_ID"].unique())
    id_map = pd.Series(np.arange(len(ids), dtype="int64"), index=ids)

    frames = []
    for pattern, setting in (
        (f"*Carrier_Claims_Sample_{sample}*.csv", "OFFICE"),
        (f"*Inpatient_Claims_Sample_{sample}*.csv", "INPATIENT"),
        (f"*Outpatient_Claims_Sample_{sample}*.csv", "OUTPATIENT"),
    ):
        for f in sorted(directory.glob(pattern)):
            df = pd.read_csv(f, low_memory=False)
            date_col = "CLM_FROM_DT" if "CLM_FROM_DT" in df else "CLM_THRU_DT"
            df["day"] = _yyyymmdd(df[date_col])
            df = df[df["day"].notna()]
            ids_cols = ["DESYNPUF_ID", "CLM_ID", "day"]
            dx = _melt_codes(df, ids_cols, DX_PREFIXES, "DX", setting, "OTHER")
            px = _melt_codes(df, ids_cols, (HCPCS_PREFIX,), "PX", setting, "OTHER")
            frames.extend([x for x in (dx, px) if len(x)])

    if not frames:
        raise FileNotFoundError(f"no claim files found in {directory}")
    claims = pd.concat(frames, ignore_index=True)
    claims["patient_id"] = claims["DESYNPUF_ID"].map(id_map)
    claims = claims[claims["patient_id"].notna()]
    claims["claim_id"] = pd.factorize(claims["CLM_ID"])[0].astype("int64")
    claims["day"] = claims["day"].astype("int32")

    birth = _yyyymmdd(bene["BENE_BIRTH_DT"])
    death = _yyyymmdd(bene["BENE_DEATH_DT"]) if "BENE_DEATH_DT" in bene else pd.Series(np.nan)
    enrol = pd.DataFrame(
        {
            "patient_id": bene["DESYNPUF_ID"].map(id_map).to_numpy(),
            "enrol_start": (pd.Timestamp("2008-01-01") - pd.Timestamp(EPOCH)).days,
            "enrol_end": (pd.Timestamp("2010-12-31") - pd.Timestamp(EPOCH)).days,
            "birth_day": birth.to_numpy(),
            "sex": np.where(bene["BENE_SEX_IDENT_CD"].to_numpy() == 2, "F", "M"),
            "death_day": death.to_numpy().astype(float),
        }
    ).dropna(subset=["patient_id", "birth_day"])
    enrol.loc[enrol["death_day"].notna(), "enrol_end"] = np.minimum(
        enrol.loc[enrol["death_day"].notna(), "enrol_end"],
        enrol.loc[enrol["death_day"].notna(), "death_day"],
    ).astype(int)

    ds = ClaimsDataset(
        claims=claims, enrolment=enrol, source=f"CMS DE-SynPUF sample {sample}"
    )
    return ds, notes
