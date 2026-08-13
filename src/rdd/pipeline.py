"""Stage orchestration.

Stages write artefacts to disk and later stages read them back rather than
recomputing. That is slightly more code than passing objects around, and it
buys the property that matters for a study: the figures, the tables and the
report are all rendered from the same stored numbers, so they cannot disagree.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import decompose as dec_mod
from . import figures as fig_mod
from . import multistate, states, validate
from . import profiles as prof_mod
from .bias import competing_risk_comparison, lookback_sweep
from .cohort import build_cohort, summarise_cohort
from .config import DEFAULT_SIM, DEFAULT_STUDY, FIGDIR, INTERIM, TABDIR, SimConfig, StudyConfig
from .schema import ClaimsDataset
from .simulate import calibrate, load_calibration, save_calibration, simulate_population

DATASET_DIR = INTERIM / "dataset"
CALIBRATION_PATH = INTERIM.parent / "calibration.json"
SUMMARY_PATH = TABDIR / "summary.json"


def _write(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TABDIR / f"{name}.csv", index=False)


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(TABDIR / f"{name}.csv")


# ---------------------------------------------------------------------------


def stage_calibrate(n: int = 1500, tol: float = 0.025) -> dict:
    """Fit and persist the friction scalars.

    ``n`` is deliberately large. At n=400 the fitted medians carry enough
    sampling noise that hypermobile EDS and Fabry -- two years apart by
    design -- came out within 0.06 years of each other, which is not a
    calibration that can support a rank-correlation check. The committed
    ``calibration.json`` is fitted at this precision and `rdd all` reuses it
    rather than silently refitting a worse one.
    """
    cal = calibrate(n=n, tol=tol)
    save_calibration(cal, CALIBRATION_PATH)
    return cal


def stage_simulate(sim: SimConfig = DEFAULT_SIM, study: StudyConfig = DEFAULT_STUDY,
                   n_cases: int | None = None, n_controls: int | None = None) -> ClaimsDataset:
    cal = load_calibration(CALIBRATION_PATH)
    if not cal:
        cal = stage_calibrate()
    if n_cases is not None:
        sim = replace(sim, n_cases_per_disease=n_cases)
    if n_controls is not None:
        sim = replace(sim, n_controls=n_controls)
    ds = simulate_population(sim, study, cal)
    ds.write(DATASET_DIR)
    return ds


def load_dataset() -> ClaimsDataset:
    if not (DATASET_DIR / "claims.parquet").exists():
        raise FileNotFoundError(
            f"no dataset at {DATASET_DIR}; run `rdd simulate` or an adapter first"
        )
    return ClaimsDataset.read(DATASET_DIR)


# ---------------------------------------------------------------------------


def stage_analyse(ds: ClaimsDataset | None = None,
                  study: StudyConfig = DEFAULT_STUDY) -> dict:
    ds = ds or load_dataset()
    horizon = study.rmst_horizon_years

    cohort_rows, trans_by, soj_rows, occ_frames, sweep_frames = [], {}, [], [], []
    for p in prof_mod.PANEL:
        susp = build_cohort(ds, p, study, design="suspicion")
        diag = build_cohort(ds, p, study, design="diagnosed")
        for design, co in (("suspicion", susp), ("diagnosed", diag)):
            s = summarise_cohort(co)
            s.update(disease=p.key, design=design)
            cohort_rows.append(s)
        if susp.empty:
            continue

        susp.to_csv(TABDIR / f"cohort_{p.key}_suspicion.csv", index=False)
        tr = states.transitions(ds, susp, p)
        trans_by[p.key] = tr
        if tr.empty:
            continue

        fitted = multistate.fit(tr, p.key)
        st = fitted.sojourn_table()
        st.insert(0, "disease", p.key)
        soj_rows.append(st)

        grid = multistate.occupancy_grid(tr, horizon)
        occ = multistate.occupancy(tr, grid)
        occ.insert(0, "disease", p.key)
        occ_frames.append(occ)

        sw = lookback_sweep(ds, p, study=study)
        sw.insert(0, "disease", p.key)
        sweep_frames.append(sw)

    _write(pd.DataFrame(cohort_rows), "cohort_summary")
    if soj_rows:
        _write(pd.concat(soj_rows, ignore_index=True), "sojourn_fits")
    if occ_frames:
        _write(pd.concat(occ_frames, ignore_index=True), "occupancy")
    if sweep_frames:
        _write(pd.concat(sweep_frames, ignore_index=True), "lookback_sweep")

    decomposition = dec_mod.decompose_panel(trans_by, horizon)
    if not decomposition.empty:
        _write(decomposition, "decomposition")

    # Both the high-mortality case and the moderate one: the contrast between
    # them is the point (see rdd.bias section 3).
    for key in ("ALS", "LOPD"):
        cr = competing_risk_comparison(ds, prof_mod.get(key), study)
        if not cr.empty:
            cr.insert(0, "disease", key)
            _write(cr, f"competing_risk_{key}")

    summary: dict = {"has_truth": ds.has_truth, "dataset": ds.summary(),
                     "study": {k: getattr(study, k) for k in
                               ("db_start", "db_end", "lookback_days",
                                "min_dx_claims", "min_dx_claim_spacing_days",
                                "max_dx_claim_span_days", "signature_min_claims",
                                "signature_min_distinct", "rmst_horizon_years")}}

    if ds.has_truth:
        phen = validate.phenotype_accuracy(ds, study)
        _write(phen, "phenotype_accuracy")
        proxy = validate.onset_proxy_accuracy(ds, study)
        if not proxy.empty:
            _write(proxy, "onset_proxy_error")
        panel = validate.delay_recovery(ds, study)
        _write(panel, "delay_recovery")
        inv = dec_mod.invisible_phase(ds.truth)
        _write(inv, "invisible_phase")
        scores = validate.recovery_scores(panel)
        summary["recovery_scores"] = scores
        summary["gate_failures"] = validate.check_gates(scores, phen, proxy)

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------


def stage_figures(ds: ClaimsDataset | None = None) -> list[str]:
    ds = ds or load_dataset()
    made = []

    cohorts = {}
    for p in prof_mod.PANEL:
        f = TABDIR / f"cohort_{p.key}_suspicion.csv"
        if f.exists():
            cohorts[p.key] = pd.read_csv(f)
    if cohorts:
        fig_mod.fig_km_panel(cohorts, FIGDIR / "01_time_to_diagnosis.png")
        made.append("01_time_to_diagnosis.png")

    if (TABDIR / "decomposition.csv").exists():
        fig_mod.fig_decomposition(_read("decomposition"), FIGDIR / "02_delay_decomposition.png")
        made.append("02_delay_decomposition.png")

    if (TABDIR / "lookback_sweep.csv").exists():
        sw = _read("lookback_sweep")
        fig_mod.fig_lookback(
            {k: g for k, g in sw.groupby("disease")}, FIGDIR / "03_lookback_ceiling.png"
        )
        made.append("03_lookback_ceiling.png")

    if (TABDIR / "delay_recovery.csv").exists():
        panel = _read("delay_recovery")
        fig_mod.fig_bias_dumbbell(panel, FIGDIR / "04_estimate_vs_truth.png")
        fig_mod.fig_bias_components(panel, FIGDIR / "05_bias_components.png")
        made += ["04_estimate_vs_truth.png", "05_bias_components.png"]

    for key in ("ALS", "LOPD"):
        if (TABDIR / f"competing_risk_{key}.csv").exists():
            name = f"06_competing_risk_{key.lower()}.png"
            fig_mod.fig_competing_risk(_read(f"competing_risk_{key}"), key, FIGDIR / name)
            made.append(name)

    if (TABDIR / "onset_proxy_error.csv").exists():
        fig_mod.fig_proxy_error(_read("onset_proxy_error"), FIGDIR / "07_onset_proxy_error.png")
        made.append("07_onset_proxy_error.png")

    if (TABDIR / "occupancy.csv").exists():
        occ = _read("occupancy")
        occ.columns = [c if not c.isdigit() else int(c) for c in occ.columns]
        for key in ("hEDS", "ALS"):
            sub = occ[occ["disease"] == key]
            if len(sub) > 3:
                fig_mod.fig_occupancy(sub, key, FIGDIR / f"08_occupancy_{key}.png")
                made.append(f"08_occupancy_{key}.png")

    if (TABDIR / "sojourn_fits.csv").exists():
        fig_mod.fig_sojourn_shapes(_read("sojourn_fits"), FIGDIR / "09_sojourn_shapes.png")
        made.append("09_sojourn_shapes.png")

    return made


def stage_validate(seeds=(1, 2, 3), n_cases: int = 400, n_controls: int = 2000) -> pd.DataFrame:
    cal = load_calibration(CALIBRATION_PATH)
    panels, scores = validate.multi_seed_recovery(
        seeds=seeds, calibration=cal, n_cases=n_cases, n_controls=n_controls
    )
    _write(panels, "recovery_by_seed")
    _write(scores, "recovery_scores")
    return scores
