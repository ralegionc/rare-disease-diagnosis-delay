"""OMOP CDM v5.4 adapter.

If your institution has a CDM instance, this is the path that runs on real
longitudinal data with real provider specialties -- the thing SynPUF cannot
give you. The queries below are plain SQL against the standard tables and take
any DBAPI connection (psycopg, duckdb, databricks-sql, snowflake-connector).

One design note that matters. OMOP stores everything as standard concept ids,
so ``rdd.profiles``' ICD-10 strings must be mapped through ``CONCEPT_RELATIONSHIP``
before the phenotype will match anything. :func:`map_source_codes` does that
lookup and, importantly, **reports codes that mapped to nothing**. A silently
unmapped diagnosis code does not produce an error -- it produces a cohort of
size zero, or worse, a cohort that is quietly missing one manifestation of the
disease and therefore has a systematically late onset proxy.
"""

from __future__ import annotations

import pandas as pd

from ..schema import ClaimsDataset

OMOP_QUERIES = {
    "claims": """
        SELECT
            co.person_id                                  AS patient_id,
            co.visit_occurrence_id                        AS claim_id,
            co.condition_start_date                       AS service_date,
            c.concept_code                                AS code,
            'DX'                                          AS code_type,
            COALESCE(co.condition_type_concept_id, 1)     AS position,
            COALESCE(spec.concept_name, 'OTHER')          AS specialty,
            COALESCE(vis.concept_name, 'OFFICE')          AS setting
        FROM condition_occurrence co
        JOIN concept c            ON c.concept_id = co.condition_concept_id
        LEFT JOIN visit_occurrence v  ON v.visit_occurrence_id = co.visit_occurrence_id
        LEFT JOIN concept vis     ON vis.concept_id = v.visit_concept_id
        LEFT JOIN provider p      ON p.provider_id = co.provider_id
        LEFT JOIN concept spec    ON spec.concept_id = p.specialty_concept_id
        WHERE co.condition_start_date IS NOT NULL

        UNION ALL

        SELECT
            po.person_id, po.visit_occurrence_id, po.procedure_date,
            c.concept_code, 'PX', 1,
            COALESCE(spec.concept_name, 'OTHER'),
            COALESCE(vis.concept_name, 'OFFICE')
        FROM procedure_occurrence po
        JOIN concept c            ON c.concept_id = po.procedure_concept_id
        LEFT JOIN visit_occurrence v  ON v.visit_occurrence_id = po.visit_occurrence_id
        LEFT JOIN concept vis     ON vis.concept_id = v.visit_concept_id
        LEFT JOIN provider p      ON p.provider_id = po.provider_id
        LEFT JOIN concept spec    ON spec.concept_id = p.specialty_concept_id
        WHERE po.procedure_date IS NOT NULL
    """,
    "enrolment": """
        SELECT
            op.person_id                       AS patient_id,
            MIN(op.observation_period_start_date) AS enrol_start,
            MAX(op.observation_period_end_date)   AS enrol_end,
            MAX(pr.birth_datetime)             AS birth_date,
            MAX(CASE WHEN pr.gender_concept_id = 8532 THEN 'F' ELSE 'M' END) AS sex,
            MAX(d.death_date)                  AS death_date
        FROM observation_period op
        JOIN person pr ON pr.person_id = op.person_id
        LEFT JOIN death d ON d.person_id = op.person_id
        GROUP BY op.person_id
    """,
    "map_codes": """
        SELECT src.concept_code AS source_code, std.concept_id AS standard_concept_id
        FROM concept src
        JOIN concept_relationship cr
          ON cr.concept_id_1 = src.concept_id AND cr.relationship_id = 'Maps to'
        JOIN concept std ON std.concept_id = cr.concept_id_2
        WHERE src.vocabulary_id IN ('ICD10CM', 'CPT4', 'HCPCS')
          AND src.concept_code IN ({codes})
    """,
}


def map_source_codes(conn, codes) -> tuple[pd.DataFrame, list[str]]:
    """Map ICD-10/CPT strings to standard concept ids; report what failed."""
    codes = list(dict.fromkeys(codes))
    placeholders = ", ".join(f"'{c}'" for c in codes)
    df = pd.read_sql(OMOP_QUERIES["map_codes"].format(codes=placeholders), conn)
    missing = sorted(set(codes) - set(df["source_code"]))
    notes = []
    if missing:
        notes.append(
            "codes with no standard mapping (the phenotype cannot see these): "
            + ", ".join(missing)
        )
    return df, notes


def load_omop(conn, limit_persons: int | None = None) -> tuple[ClaimsDataset, list[str]]:
    from .generic import _to_day, map_setting, map_specialty

    notes: list[str] = []
    claims = pd.read_sql(OMOP_QUERIES["claims"], conn)
    enrol = pd.read_sql(OMOP_QUERIES["enrolment"], conn)

    if limit_persons:
        keep = enrol["patient_id"].drop_duplicates().head(limit_persons)
        enrol = enrol[enrol["patient_id"].isin(keep)]
        claims = claims[claims["patient_id"].isin(keep)]

    claims["day"] = _to_day(claims["service_date"])
    claims["specialty"], unmapped = map_specialty(claims["specialty"])
    if unmapped > 0.10:
        notes.append(
            f"{unmapped:.1%} of encounters had a provider specialty outside "
            "SPECIALTY_MAP and became OTHER"
        )
    claims["setting"] = map_setting(claims["setting"])
    claims["position"] = 1
    claims["claim_id"] = pd.factorize(claims["claim_id"])[0].astype("int64")

    enrol["enrol_start"] = _to_day(enrol["enrol_start"])
    enrol["enrol_end"] = _to_day(enrol["enrol_end"])
    enrol["birth_day"] = _to_day(enrol["birth_date"])
    enrol["death_day"] = _to_day(enrol["death_date"]).astype(float)

    ds = ClaimsDataset(claims=claims, enrolment=enrol, source="OMOP CDM v5.4")
    return ds, notes
