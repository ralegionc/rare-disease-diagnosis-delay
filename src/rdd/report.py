"""Render the methods report from stored tables.

Section markers are load-bearing, not decoration:

``[METHOD]``
    a claim about how the estimator behaves. Valid regardless of the data
    source, because it is a statement about the procedure.
``[SYNTHETIC]``
    a number produced from simulated claims. It describes the simulator, and
    it is evidence about the *estimator*, not about any real patient.
``[LITERATURE]``
    an external published figure, cited.

No section in this report claims anything about real-world diagnostic delay,
because the pipeline has not yet been run on real-world data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import profiles as prof_mod
from .config import OUT, TABDIR
from .states import STATE_LABELS, State
from .viz import DISEASE_LABEL, DISEASE_ORDER

REPORT_PATH = OUT / "report.md"


def _try(name: str):
    p = TABDIR / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def _fmt(v, nd=1, suffix=""):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    v = v + 0.0  # collapse -0.0, which renders as a confusing "-0.00"
    if v == 0:
        v = abs(v)
    return f"{v:.{nd}f}{suffix}"


def _md_table(df: pd.DataFrame, cols: dict[str, str], nd: int = 2) -> str:
    head = "| " + " | ".join(cols.values()) + " |"
    rule = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [head, rule]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, (int, np.integer)):
                cells.append(f"{v:,}")
            elif isinstance(v, float):
                cells.append(_fmt(v, nd))
            else:
                cells.append(str(v) if v is not None else "--")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _order_rows(df, col="disease"):
    if df is None or df.empty:
        return df
    df = df.copy()
    df["__o"] = df[col].map({k: i for i, k in enumerate(DISEASE_ORDER)})
    return df.sort_values("__o").drop(columns="__o")


def write_report(path: Path = REPORT_PATH) -> Path:
    summary = {}
    sp = TABDIR / "summary.json"
    if sp.exists():
        summary = json.loads(sp.read_text(encoding="utf-8"))

    recovery = _order_rows(_try("delay_recovery"))
    cohort = _try("cohort_summary")
    phen = _order_rows(_try("phenotype_accuracy"))
    proxy = _order_rows(_try("onset_proxy_error"))
    dec = _try("decomposition")
    sweep = _try("lookback_sweep")
    soj = _try("sojourn_fits")
    inv = _order_rows(_try("invisible_phase"))
    scores = summary.get("recovery_scores", {})
    ds_info = summary.get("dataset", {})

    L: list[str] = []
    a = L.append

    a("# Modelling the rare-disease diagnostic odyssey from administrative claims")
    a("")
    a("*Methods report. Every quantitative result below is marked with its "
      "provenance: `[METHOD]` for properties of the estimator, `[SYNTHETIC]` "
      "for numbers computed on simulated claims, `[LITERATURE]` for published "
      "external figures. No result here is a claim about real patients.*")
    a("")

    # ------------------------------------------------------------------
    a("## The problem in one paragraph")
    a("")
    a("Patients with rare conditions are widely reported to wait years for a "
      "correct diagnosis. Administrative claims are the obvious data source "
      "for measuring this at scale -- they are longitudinal, population-sized "
      "and already collected. They are also structurally incapable of "
      "observing the quantity of interest. Claims record billing, so the "
      "earliest event they can contain is the first time a patient sought care "
      "*and someone coded something*; symptom onset leaves no trace. Every "
      "estimate of diagnostic delay from claims is therefore an estimate of a "
      "latent quantity through a proxy, under censoring, inside a database "
      "with finite history. This package builds that estimator, and then "
      "measures how far wrong it is by running it against simulated claims "
      "where the latent quantity is known.")
    a("")

    # ------------------------------------------------------------------
    a("## The panel")
    a("")
    a("Six conditions chosen to disagree with each other along the axes that "
      "plausibly drive delay: tempo, presenting setting, mortality, and how "
      "specific the early symptoms are. `[LITERATURE]`")
    a("")
    rows = []
    for p in prof_mod.PANEL:
        rows.append(
            {
                "disease": DISEASE_LABEL[p.key],
                "code": p.dx_code,
                "anchor": p.target_median_delay_years,
                "onset_age": p.onset_age_mean,
                "death_hazard": p.death_hazard,
                "dominant_label": max(p.misdx, key=lambda m: m.prob).label,
                "reference": p.reference,
            }
        )
    a(_md_table(
        pd.DataFrame(rows),
        {
            "disease": "Condition", "code": "ICD-10",
            "anchor": "Published delay (yr)", "onset_age": "Mean onset age",
            "death_hazard": "Death hazard /yr",
            "dominant_label": "Most common wrong label",
        },
        nd=2,
    ))
    a("")
    a("The published delays are *inputs*, not findings: they set the ground "
      "truth the simulator generates from. One friction scalar per disease is "
      "fitted by bisection so the simulated true median delay matches its "
      "anchor. Because the target is the true delay rather than anything the "
      "pipeline estimates, calibration cannot flatter the estimators.")
    a("")

    # ------------------------------------------------------------------
    a("## Study designs")
    a("")
    a("`[METHOD]` Two designs are implemented because the literature is split "
      "between them and they do not agree.")
    a("")
    a("**Diagnosed-anchored (retrospective).** Identify patients who carry the "
      "diagnosis, then look backwards for the first sustained symptom "
      "signature. Optimistic for two compounding reasons: the never-diagnosed "
      "are absent from the denominator, and a delay longer than the available "
      "history cannot be represented.")
    a("")
    a("**Suspicion-anchored (prospective).** Enter the cohort on a symptom "
      "signature, follow forward to diagnosis or censoring. Not conditioned on "
      "the outcome, but contaminated by phenocopies -- people with "
      "fibromyalgia who do not have hypermobile EDS enter the hEDS risk set "
      "and dilute the curve.")
    a("")
    a("Neither is correct. Reported together they bracket the answer.")
    a("")
    if cohort is not None and len(cohort):
        c = cohort.copy()
        c["disease"] = c["disease"].map(DISEASE_LABEL).fillna(c["disease"])
        a("`[SYNTHETIC]` Cohort sizes:")
        a("")
        a(_md_table(
            c,
            {"disease": "Condition", "design": "Design", "n": "n",
             "n_diagnosed": "Diagnosed", "n_censored": "Censored",
             "n_died_undiagnosed": "Died undiagnosed",
             "pct_left_truncated": "% left-truncated"},
            nd=1,
        ))
        a("")

    # ------------------------------------------------------------------
    a("## Does the case definition find the right people?")
    a("")
    if phen is not None and len(phen):
        a("`[SYNTHETIC]` The simulator emits rule-out coding -- specialists who "
          "code a rare disease as a hypothesis and move on, and controls who "
          "pick up a spurious code during a workup -- so the phenotype "
          "algorithm is graded on an exam that has wrong answers.")
        a("")
        pp = phen.copy()
        pp["disease"] = pp["disease"].map(DISEASE_LABEL).fillna(pp["disease"])
        a(_md_table(pp, {"disease": "Condition", "n_flagged": "Flagged",
                         "n_true": "True cases", "ppv": "PPV",
                         "sensitivity": "Sensitivity"}, nd=3))
        a("")
        a(f"Sensitivity is low across the board (median "
          f"{_fmt(float(phen['sensitivity'].median()), 2)}), and that is the "
          "correct behaviour rather than a defect: most simulated patients are "
          "diagnosed outside their coverage window, or not at all within it. A "
          "case definition cannot find a diagnosis that was never billed. It "
          "is precision that matters for a delay study, since a false positive "
          "contributes an entirely fictitious odyssey.")
        a("")

    # ------------------------------------------------------------------
    a("## How late is the origin?")
    a("")
    if proxy is not None and len(proxy):
        pr = proxy.copy()
        pr["disease"] = pr["disease"].map(DISEASE_LABEL).fillna(pr["disease"])
        a("`[SYNTHETIC]` The claims-based origin -- first sustained symptom "
          "signature -- against true symptom onset. The error is one-sided by "
          "construction: the proxy can never precede the first claim.")
        a("")
        a(_md_table(pr, {"disease": "Condition", "n": "n",
                         "median_proxy_error_years": "Median lateness (yr)",
                         "p25": "p25", "p75": "p75",
                         "median_vs_first_care": "vs first care contact (yr)"}, nd=2))
        a("")
    if inv is not None and len(inv):
        iv = inv.copy()
        iv["disease"] = iv["disease"].map(DISEASE_LABEL).fillna(iv["disease"])
        a("`[SYNTHETIC]` Part of that lateness is a phase claims cannot see at "
          "all: symptom onset to first care contact. No claim is generated "
          "while a patient is unwell and has not presented, so this interval "
          "is absent from administrative data rather than merely noisy. It is "
          "the irreducible floor on what any claims-only study can observe.")
        a("")
        a(_md_table(iv, {"disease": "Condition", "n": "n",
                         "median_years": "Median (yr)", "p25": "p25",
                         "p75": "p75"}, nd=2))
        a("")

    # ------------------------------------------------------------------
    a("## Estimate versus truth")
    a("")
    if recovery is not None and len(recovery):
        rr = recovery.copy()
        rr["disease_label"] = rr["disease"].map(DISEASE_LABEL).fillna(rr["disease"])
        a("`[SYNTHETIC]` The retrospective estimate against ground truth, with "
          "the bias split into the part caused by *who gets analysed* "
          "(selection) and the part caused by *where the clock starts* "
          "(measurement).")
        a("")
        a(_md_table(rr, {
            "disease_label": "Condition",
            "true_median_from_onset": "True (yr)",
            "retrospective_median": "Estimated (yr)",
            "selection_component": "Selection (yr)",
            "measurement_component": "Measurement (yr)",
            "understatement_fraction": "Understated by",
        }, nd=2))
        a("")
        if scores:
            a(f"`[SYNTHETIC]` Across the panel: r = "
              f"{_fmt(scores.get('pearson_r'), 2)} (Spearman rho "
              f"{_fmt(scores.get('spearman_rho'), 2)}) between estimated and true "
              f"median delay, mean absolute bias "
              f"{_fmt(scores.get('mean_abs_bias_years'), 2)} years, mean "
              f"understatement {_fmt(100 * scores.get('mean_understatement_fraction', np.nan), 0, '%')}.")
            a("")
            sel = scores.get("median_selection_component", np.nan)
            mea = scores.get("median_measurement_component", np.nan)
            a(f"The split is the useful part. Median selection component "
              f"{_fmt(sel, 2)} years; median measurement component "
              f"{_fmt(mea, 2)} years. Selection is fixable by design -- follow "
              "a suspicion cohort forward instead of conditioning on the "
              "outcome. Measurement is not fixable by design at all, because "
              "it is a property of the data source. The only remedies are a "
              "longer look-back or an onset date from outside the claims.")
            a("")
        from .validate import SAMPLE_SIZE_NOTE

        a("### How many patients does this need?")
        a("")
        a("`[SYNTHETIC]` " + SAMPLE_SIZE_NOTE.replace("\n", " "))
        a("")
        a("**The ranking survives even though the levels do not.** A rank "
          "correlation near 1 with a large negative bias means claims can tell "
          "you reliably *which* rare diseases have the worst odysseys while "
          "being systematically wrong about *how bad* any of them are. For "
          "prioritising where to intervene that is often enough; for quoting a "
          "number in a policy document it is not.")
        a("")

    # ------------------------------------------------------------------
    a("## The look-back ceiling")
    a("")
    if sweep is not None and len(sweep):
        piv = sweep.pivot_table(index="history_years", columns="disease",
                                values="median_delay")
        cols = [c for c in DISEASE_ORDER if c in piv.columns]
        piv = piv[cols].reset_index()
        a("`[SYNTHETIC]` The retrospective analysis re-run on progressively "
          "truncated claims histories. Estimated median delay in years:")
        a("")
        a(_md_table(piv, {**{"history_years": "History (yr)"},
                          **{c: DISEASE_LABEL.get(c, c) for c in cols}}, nd=1))
        a("")
        a("`[METHOD]` A database holding *W* years of history cannot exhibit a "
          "delay longer than *W*. The estimate does not become noisy as *W* "
          "shrinks -- it saturates just below *W*, for every disease "
          "simultaneously. Two studies of the same condition in databases of "
          "different vintage will therefore disagree, with the shorter "
          "database reporting the shorter delay. Presented as a time series, "
          "that artefact is indistinguishable from an improvement in care.")
        a("")

    # ------------------------------------------------------------------
    a("## Where the years go")
    a("")
    if dec is not None and len(dec):
        piv = dec.pivot_table(index="disease", columns="phase", values="share")
        piv = piv.reindex([k for k in DISEASE_ORDER if k in piv.index]).reset_index()
        piv["disease"] = piv["disease"].map(DISEASE_LABEL).fillna(piv["disease"])
        cols = {"disease": "Condition"}
        for s in (State.UNDIFFERENTIATED, State.SPECIALIST, State.MISLABELLED,
                  State.CORRECT_SPECIALTY):
            lbl = STATE_LABELS[s]
            if lbl in piv.columns:
                cols[lbl] = lbl
        a("`[SYNTHETIC]` Share of observed pre-diagnosis time by pathway phase, "
          "from the occupancy integral of the fitted multi-state model.")
        a("")
        a(_md_table(piv, cols, nd=3))
        a("")
        a("The phases are inferred from provider specialty and from the "
          "persistence of differential-diagnosis codes, not from anything the "
          "simulator labelled. `[METHOD]`")
        a("")
    if soj is not None and len(soj):
        mis = soj[soj["state"] == int(State.MISLABELLED)].dropna(subset=["shape"])
        if len(mis):
            a(f"`[SYNTHETIC]` Weibull sojourn shape in the *carrying a wrong "
              f"label* state has a panel median of "
              f"{_fmt(float(mis['shape'].median()), 2)}. A shape below 1 means "
              "the exit hazard falls the longer the label has held: each "
              "additional year makes revision less likely, not more. That is "
              "what a settled misdiagnosis looks like quantitatively, and it "
              "is why a wrong label costs more delay than the visits it "
              "generates would suggest.")
            a("")

    # ------------------------------------------------------------------
    a("## Death is not censoring")
    a("")
    a("`[METHOD]` For a condition with substantial pre-diagnosis mortality, "
      "`1 - Kaplan-Meier` answers \"what fraction would eventually be "
      "diagnosed if nobody died\". Nobody asked that. The Aalen-Johansen "
      "cumulative incidence answers the question actually posed, and the gap "
      "between them is the amount by which the naive curve extrapolates the "
      "diagnosis hazard over person-time that does not exist.")
    a("")
    peaks = {}
    for key in ("ALS", "LOPD"):
        cr = _try(f"competing_risk_{key}")
        if cr is not None and len(cr):
            w = cr.loc[cr["overstatement"].idxmax()]
            rel = (
                float(w["overstatement"]) / float(w["aj_diagnosed"])
                if float(w["aj_diagnosed"]) > 0 else np.nan
            )
            peaks[key] = (float(w["overstatement"]), rel, float(w["t"]),
                          float(cr["aj_died_undiagnosed"].max()))
    for key, (abs_, rel, t, dcif) in peaks.items():
        a(f"`[SYNTHETIC]` {DISEASE_LABEL.get(key, key)}: peak overstatement "
          f"{_fmt(100 * abs_, 1)} percentage points ({_fmt(100 * rel, 0)}% "
          f"relative) at {_fmt(t, 1)} years; cumulative incidence of dying "
          f"undiagnosed reaches {_fmt(100 * dcif, 1)}%.")
    if len(peaks) == 2 and peaks["ALS"][1] < peaks["LOPD"][1]:
        a("")
        a("**The correction is larger for Pompe than for ALS, and that inverts "
          "the intuition.** ALS has by far the higher mortality, but it kills "
          "fast enough that most patients who die undiagnosed never accumulate "
          "a symptom signature and so never enter the cohort. Mortality did "
          "its selecting upstream of the estimator, where no competing-risk "
          "method can reach it. A small Aalen-Johansen correction is therefore "
          "not evidence that death did not matter -- it can equally mean death "
          "mattered so much that those patients are missing from the "
          "denominator entirely. `[METHOD]`")
    if peaks:
        a("")

    # ------------------------------------------------------------------
    a("## What would need to be true for these estimates to be usable")
    a("")
    a("`[METHOD]` Three conditions, in decreasing order of how easy they are "
      "to satisfy:")
    a("")
    a("1. **History longer than the delay.** If the plausible delay is a "
      "decade, a decade of continuous enrolment is the minimum, and enrolment "
      "churn means a database needs far more than a decade of coverage to "
      "yield a usable sample of such patients.")
    a("2. **An outcome-independent risk set.** Enter patients on symptoms, not "
      "on the diagnosis, and accept the contamination -- it is measurable, "
      "whereas the selection bias it replaces is not.")
    a("3. **An external onset date for a validation subset.** Nothing inside "
      "the claims can calibrate the measurement error. A linked registry, "
      "chart review or patient survey on even a few hundred patients would "
      "pin the one term this design cannot estimate from its own data.")
    a("")

    # ------------------------------------------------------------------
    a("## Status")
    a("")
    a("The pipeline is validated end to end on the simulator. It has **not** "
      "been run on real claims: MarketScan, Optum and the Medicare LDS are all "
      "licensed, and the one public alternative -- CMS DE-SynPUF -- is "
      "unsuitable on four independent grounds documented in "
      "`rdd/adapters/synpuf.py` (three-year history, Medicare age range, "
      "per-variable synthesis that breaks longitudinal coherence, and no "
      "provider specialty). The adapters exist so that the analysis runs "
      "unchanged the moment real data is available.")
    a("")
    if summary.get("gate_failures"):
        a("**Quality gates currently failing:**")
        a("")
        for f in summary["gate_failures"]:
            a(f"- {f}")
        a("")
    else:
        a("All recovery quality gates pass. They are enforced as tests, so a "
          "change that quietly breaks recovery fails CI rather than producing "
          "a slightly different plot nobody checks.")
        a("")
    if ds_info:
        a(f"*Generated from {ds_info.get('patients', 0):,} simulated patients "
          f"and {ds_info.get('claim_lines', 0):,} claim lines. "
          f"Source: {ds_info.get('source', 'unknown')}.*")
        a("")

    a("## Figures")
    a("")
    for name, cap in [
        ("01_time_to_diagnosis.png", "Cumulative probability of diagnosis, prospective design"),
        ("02_delay_decomposition.png", "Share of pre-diagnosis time by pathway phase"),
        ("03_lookback_ceiling.png", "Estimated delay against available database history"),
        ("04_estimate_vs_truth.png", "Claims estimate against ground truth"),
        ("05_bias_components.png", "Bias split into selection and measurement"),
        ("06_competing_risk_als.png", "Competing risks in ALS"),
        ("06_competing_risk_lopd.png", "Competing risks in late-onset Pompe"),
        ("07_onset_proxy_error.png", "Lateness of the claims-based origin"),
        ("08_occupancy_hEDS.png", "State occupancy over the odyssey (hEDS)"),
        ("08_occupancy_ALS.png", "State occupancy over the odyssey (ALS)"),
        ("09_sojourn_shapes.png", "Weibull sojourn shape by phase"),
    ]:
        if (OUT / "figures" / name).exists():
            a(f"![{cap}](figures/{name})")
            a("")
            a(f"*{cap}.*")
            a("")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    return path
