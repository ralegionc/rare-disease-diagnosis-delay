"""Figure generation.

Every figure is built from an artefact written by the ``analyse`` stage, never
from a live recomputation, so the numbers in a chart and the numbers in the
report cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import viz
from .config import FIGDIR
from .states import STATE_LABELS, State
from .survival import kaplan_meier
from .viz import DISEASE_COLOR, DISEASE_LABEL, DISEASE_ORDER


def _order(keys):
    return [k for k in DISEASE_ORDER if k in set(keys)]


# ---------------------------------------------------------------------------
# 1. Time to diagnosis, prospective design
# ---------------------------------------------------------------------------


def fig_km_panel(cohorts: dict, path: Path):
    viz.apply_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for key in _order(cohorts):
        co = cohorts[key]
        if co.empty or co["event"].sum() < 5:
            continue
        km = kaplan_meier(co["exit"].to_numpy(), co["event"].to_numpy())
        y = 1.0 - km.survival
        ax.step(km.time, y, where="post", color=DISEASE_COLOR[key],
                label=DISEASE_LABEL[key])
        # Direct label at the right-hand end of each curve.
        ax.annotate(
            DISEASE_LABEL[key], xy=(km.time[-1], y[-1]), xytext=(4, 0),
            textcoords="offset points", va="center", fontsize=8.5,
            color=viz.INK_2,
        )
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    viz.finish(
        ax,
        "Cumulative probability of diagnosis after a symptom signature appears",
        "Suspicion-anchored cohort; patients never diagnosed contribute censored time",
        "Years since first sustained symptom signature",
        "Diagnosed",
    )
    ax.legend(loc="upper left", ncols=2)
    viz.caption(
        fig,
        "Curves that flatten far below 100% are diseases where most of the "
        "suspected population is never diagnosed inside the observation window.",
    )
    viz.save(fig, path)


# ---------------------------------------------------------------------------
# 2. Delay decomposition
# ---------------------------------------------------------------------------


def fig_decomposition(dec: pd.DataFrame, path: Path):
    viz.apply_style()
    keys = _order(dec["disease"].unique())
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    states = [int(s) for s in (State.UNDIFFERENTIATED, State.SPECIALIST,
                               State.MISLABELLED, State.CORRECT_SPECIALTY)]
    ypos = np.arange(len(keys))
    left = np.zeros(len(keys))
    for s in states:
        vals = np.array(
            [
                float(dec[(dec["disease"] == k) & (dec["state"] == s)]["share"].sum())
                for k in keys
            ]
        )
        ax.barh(ypos, vals, left=left, height=0.62, color=viz.PHASE_COLOR[s],
                label=STATE_LABELS[State(s)], edgecolor=viz.SURFACE, linewidth=1.6)
        for i, (v, l0) in enumerate(zip(vals, left)):
            if v > 0.09:
                ax.text(l0 + v / 2, ypos[i], f"{v:.0%}", ha="center", va="center",
                        fontsize=8.5, color="white", fontweight="bold")
        left += vals
    ax.set_yticks(ypos, [DISEASE_LABEL[k] for k in keys])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    viz.finish(
        ax,
        "Where the pre-diagnosis years are spent",
        "Share of restricted mean time before diagnosis, by pathway phase",
        "Share of observed pre-diagnosis time",
        grid_axis="x",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncols=2)
    viz.caption(
        fig,
        "Phases are inferred from provider specialty and the persistence of "
        "differential diagnosis codes. Time before the patient first presents "
        "is not included -- no claim exists for it.",
        y=-0.30,
    )
    viz.save(fig, path)


# ---------------------------------------------------------------------------
# 3. The look-back ceiling
# ---------------------------------------------------------------------------


def fig_lookback(sweeps: dict, path: Path):
    viz.apply_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    xs = None
    for key in _order(sweeps):
        s = sweeps[key].dropna(subset=["median_delay"])
        if s.empty:
            continue
        xs = s["history_years"].to_numpy()
        ax.plot(xs, s["median_delay"], color=DISEASE_COLOR[key], marker="o",
                markersize=5, label=DISEASE_LABEL[key], zorder=3)
    if xs is not None:
        lim = np.linspace(0, max(xs), 50)
        ax.plot(lim, lim, color=viz.MUTED, linestyle=(0, (4, 3)), linewidth=1.4,
                zorder=1)
        ax.annotate("structural ceiling\n(delay cannot exceed history)",
                    xy=(lim[-1] * 0.55, lim[-1] * 0.55), xytext=(-6, 10),
                    textcoords="offset points", fontsize=8.5, color=viz.MUTED,
                    ha="right")
    viz.finish(
        ax,
        "Estimated median delay is bounded by how much history the database holds",
        "Retrospective design, re-run on progressively truncated claims histories",
        "Years of claims history available",
        "Estimated median delay (years)",
    )
    ax.legend(loc="upper left", ncols=2)
    viz.caption(
        fig,
        "Curves track the diagonal while history is the binding constraint and "
        "flatten once it is not. A study run on a short database reports a "
        "short delay for every disease, regardless of the disease.",
    )
    viz.save(fig, path)


# ---------------------------------------------------------------------------
# 4. Bias decomposition
# ---------------------------------------------------------------------------


def fig_bias_dumbbell(panel: pd.DataFrame, path: Path):
    viz.apply_style()
    p = panel.set_index("disease")
    keys = _order(p.index)
    y = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(7.6, 4.4))

    true_v = np.array([p.at[k, "true_median_from_onset"] for k in keys])
    est_v = np.array([p.at[k, "retrospective_median"] for k in keys])

    for i, k in enumerate(keys):
        ax.plot([est_v[i], true_v[i]], [y[i], y[i]], color=viz.AXIS, linewidth=3,
                solid_capstyle="round", zorder=1)
    ax.scatter(est_v, y, s=70, color=viz.SERIES[7], zorder=3,
               label="Claims estimate (retrospective design)",
               edgecolor=viz.SURFACE, linewidth=1.5)
    ax.scatter(true_v, y, s=70, color=viz.SERIES[0], zorder=3,
               label="True median delay from onset",
               edgecolor=viz.SURFACE, linewidth=1.5)
    for i, k in enumerate(keys):
        gap = true_v[i] - est_v[i]
        ax.annotate(f"-{gap:.1f} yr", xy=(max(true_v[i], est_v[i]), y[i]),
                    xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color=viz.CRITICAL)

    ax.set_yticks(y, [DISEASE_LABEL[k] for k in keys])
    ax.invert_yaxis()
    ax.set_xlim(0, max(np.nanmax(true_v) * 1.25, 1))
    viz.finish(
        ax,
        "Every disease's delay is understated, and the longer the odyssey the worse it gets",
        "Retrospective claims estimate against simulator ground truth",
        "Median years from symptom onset to diagnosis",
        grid_axis="x",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=2)
    viz.caption(
        fig,
        "The gap is predominantly measurement rather than selection: "
        "restricting ground truth to exactly the analysed patients moves it "
        "far less than starting the clock late does.",
        y=-0.28,
    )
    viz.save(fig, path)


def fig_bias_components(panel: pd.DataFrame, path: Path):
    viz.apply_style()
    p = panel.set_index("disease")
    keys = _order(p.index)
    y = np.arange(len(keys))
    sel = np.array([p.at[k, "selection_component"] for k in keys])
    mea = np.array([p.at[k, "measurement_component"] for k in keys])

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.barh(y - 0.19, sel, height=0.34, color=viz.SERIES[2],
            label="Selection (who is analysed)")
    ax.barh(y + 0.19, mea, height=0.34, color=viz.SERIES[1],
            label="Measurement (how late the origin is)")
    ax.axvline(0, color=viz.AXIS, linewidth=1)
    ax.set_yticks(y, [DISEASE_LABEL[k] for k in keys])
    ax.invert_yaxis()
    viz.finish(
        ax,
        "The bias is measurement, not selection",
        "Decomposition of retrospective bias, in years (negative = understates delay)",
        "Years contributed to the total bias",
        grid_axis="x",
    )
    ax.legend(loc="lower left")
    viz.caption(
        fig,
        "Selection bias is fixable by design. Measurement bias is a property "
        "of the data source: claims begin when billing begins.",
    )
    viz.save(fig, path)


# ---------------------------------------------------------------------------
# 5. Competing risks
# ---------------------------------------------------------------------------


def fig_competing_risk(cr: pd.DataFrame, disease: str, path: Path):
    viz.apply_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(cr["t"], cr["naive_1_minus_km"], color=viz.SERIES[7],
            linestyle=(0, (5, 3)), label="1 - Kaplan-Meier (death as censoring)")
    ax.plot(cr["t"], cr["aj_diagnosed"], color=viz.SERIES[0],
            label="Aalen-Johansen: diagnosed")
    ax.plot(cr["t"], cr["aj_died_undiagnosed"], color=viz.SERIES[1],
            label="Aalen-Johansen: died undiagnosed")
    ax.fill_between(cr["t"], cr["aj_diagnosed"], cr["naive_1_minus_km"],
                    color=viz.SERIES[7], alpha=0.10, linewidth=0)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    viz.finish(
        ax,
        f"Treating death as censoring overstates diagnosis probability ({DISEASE_LABEL.get(disease, disease)})",
        "Shaded area is the overstatement",
        "Years since first sustained symptom signature",
        "Cumulative probability",
    )
    ax.legend(loc="upper left")
    viz.caption(
        fig,
        "A patient who dies undiagnosed is not still waiting for a diagnosis. "
        "1 - KM answers 'what if nobody died', which is not the question.",
    )
    viz.save(fig, path)


# ---------------------------------------------------------------------------
# 6. Onset proxy error
# ---------------------------------------------------------------------------


def fig_proxy_error(proxy: pd.DataFrame, path: Path):
    viz.apply_style()
    keys = _order(proxy["disease"].unique())
    p = proxy.set_index("disease")
    y = np.arange(len(keys))
    med = np.array([p.at[k, "median_proxy_error_years"] for k in keys])
    lo = np.array([p.at[k, "p25"] for k in keys])
    hi = np.array([p.at[k, "p75"] for k in keys])

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, k in enumerate(keys):
        ax.plot([lo[i], hi[i]], [y[i], y[i]], color=viz.AXIS, linewidth=6,
                solid_capstyle="round", zorder=1)
        ax.scatter([med[i]], [y[i]], s=64, color=DISEASE_COLOR[k], zorder=3,
                   edgecolor=viz.SURFACE, linewidth=1.5)
        ax.annotate(f"{med[i]:+.1f} yr", xy=(hi[i], y[i]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=8.5,
                    color=viz.INK_2)
    ax.axvline(0, color=viz.AXIS, linewidth=1)
    ax.set_yticks(y, [DISEASE_LABEL[k] for k in keys])
    ax.invert_yaxis()
    viz.finish(
        ax,
        "How late the claims-based onset proxy is",
        "Median and interquartile range of (first symptom claim - true onset)",
        "Years the origin is placed after true symptom onset",
        grid_axis="x",
    )
    viz.caption(
        fig,
        "The proxy can never precede the first claim, so this error is "
        "one-sided. It transfers directly into every delay estimate.",
    )
    viz.save(fig, path)


# ---------------------------------------------------------------------------
# 7. State occupancy
# ---------------------------------------------------------------------------


def fig_occupancy(occ: pd.DataFrame, disease: str, path: Path):
    viz.apply_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    states = [int(State.UNDIFFERENTIATED), int(State.SPECIALIST),
              int(State.MISLABELLED), int(State.CORRECT_SPECIALTY),
              int(State.DIAGNOSED), int(State.DEAD)]
    colors = [viz.SERIES[0], viz.SERIES[1], viz.SERIES[2], viz.SERIES[3],
              viz.GOOD, viz.MUTED]
    ys = [occ[s].to_numpy() if s in occ else np.zeros(len(occ)) for s in states]
    ax.stackplot(occ["t"], *ys, colors=colors,
                 labels=[STATE_LABELS[State(s)] for s in states],
                 edgecolor=viz.SURFACE, linewidth=0.6)
    ax.set_ylim(0, 1)
    ax.set_xlim(occ["t"].min(), occ["t"].max())
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    viz.finish(
        ax,
        f"Odyssey state occupancy over time ({DISEASE_LABEL.get(disease, disease)})",
        "Fraction of the still-observed cohort in each pathway phase",
        "Years since first sustained symptom signature",
        "Share of cohort under observation",
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    viz.caption(
        fig,
        f"Denominator falls with follow-up; minimum observed n = "
        f"{int(occ['n_observed'].min())}.",
    )
    viz.save(fig, path)


# ---------------------------------------------------------------------------
# 8. Sojourn hazard shape
# ---------------------------------------------------------------------------


def fig_sojourn_shapes(soj: pd.DataFrame, path: Path):
    viz.apply_style()
    d = soj[soj["best_family"].notna() & soj["shape"].notna()].copy()
    if d.empty:
        return
    keys = _order(d["disease"].unique())
    states = [int(State.UNDIFFERENTIATED), int(State.SPECIALIST),
              int(State.MISLABELLED), int(State.CORRECT_SPECIALTY)]
    width = 0.19
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    x = np.arange(len(keys))
    for j, s in enumerate(states):
        vals = [
            float(d[(d["disease"] == k) & (d["state"] == s)]["shape"].mean())
            if len(d[(d["disease"] == k) & (d["state"] == s)]) else np.nan
            for k in keys
        ]
        ax.bar(x + (j - 1.5) * width, vals, width=width * 0.92,
               color=viz.PHASE_COLOR[s], label=STATE_LABELS[State(s)],
               edgecolor=viz.SURFACE, linewidth=1.2)
    ax.axhline(1.0, color=viz.CRITICAL, linewidth=1.4, linestyle=(0, (4, 3)))
    ax.annotate("shape = 1: memoryless exit", xy=(len(keys) - 0.5, 1.0),
                xytext=(0, 6), textcoords="offset points", ha="right",
                fontsize=8.5, color=viz.CRITICAL)
    ax.set_xticks(x, [DISEASE_LABEL[k] for k in keys], rotation=18, ha="right")
    viz.finish(
        ax,
        "Weibull sojourn shape by pathway phase",
        "Shape below 1 means the exit hazard falls the longer the patient stays",
        ylabel="Weibull shape parameter",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncols=2)
    viz.caption(
        fig,
        "A shape below 1 in 'carrying a wrong label' is what a settled "
        "misdiagnosis looks like quantitatively: each additional year makes "
        "revision less likely, not more.",
        y=-0.34,
    )
    viz.save(fig, path)


def figure_index(directory: Path = FIGDIR) -> list[str]:
    return sorted(p.name for p in Path(directory).glob("*.png"))
