"""Command line interface.

::

    rdd calibrate                 fit friction scalars to the literature anchors
    rdd simulate                  build the synthetic claims database
    rdd ingest <adapter> <path>   load a real source into the canonical schema
    rdd doctor                    can this database answer the question at all?
    rdd analyse                   cohorts, multi-state fits, decomposition, bias
    rdd validate                  parameter recovery across seeds
    rdd figures                   render figures from stored tables
    rdd report                    write the methods report
    rdd all                       everything, in order
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace

import pandas as pd

from . import pipeline
from .config import DEFAULT_STUDY, OUT, TABDIR


def _print_df(df: pd.DataFrame, floatfmt: str = "%.2f") -> None:
    with pd.option_context("display.width", 160, "display.max_columns", 40,
                           "display.float_format", lambda v: floatfmt % v):
        print(df.to_string(index=False))


def cmd_calibrate(args):
    cal = pipeline.stage_calibrate(n=args.n, tol=args.tol)
    print(json.dumps(cal, indent=2))
    print("\nFriction < 1 means the disease needed *less* diagnostic friction "
          "than the model's default to match its published delay.")


def cmd_simulate(args):
    ds = pipeline.stage_simulate(n_cases=args.n_cases, n_controls=args.n_controls)
    print(json.dumps(ds.summary(), indent=2, default=str))


def cmd_ingest(args):
    if args.adapter == "synpuf":
        from .adapters import load_synpuf

        ds, notes = load_synpuf(args.path)
    elif args.adapter == "generic":
        from .adapters import load_generic

        ds, notes = load_generic(args.path)
    else:
        print("omop ingest needs a live connection; call rdd.adapters.load_omop(conn)",
              file=sys.stderr)
        return 2
    ds.write(pipeline.DATASET_DIR)
    print(json.dumps(ds.summary(), indent=2, default=str))
    if notes:
        print("\nAdapter caveats -- read these before believing any result:")
        for n in notes:
            print(f"  * {n}")
    return 0


def cmd_doctor(args):
    """Always run this first on a new source. It answers 'is this database
    capable of supporting the estimate' before anything computes an estimate."""
    from .doctor import diagnose

    ds = pipeline.load_dataset()
    report = diagnose(ds, DEFAULT_STUDY)
    print(report.text())
    report.per_disease.to_csv(TABDIR / "doctor_per_disease.csv", index=False)
    print(f"Per-disease detail written to {TABDIR / 'doctor_per_disease.csv'}")
    return 0 if report.usable else 2


def cmd_analyse(args):
    study = DEFAULT_STUDY
    if args.lookback_years is not None:
        study = replace(study, lookback_days=int(args.lookback_years * 365.25))
    summary = pipeline.stage_analyse(study=study)
    print(json.dumps(summary.get("recovery_scores", {}), indent=2, default=float))
    fails = summary.get("gate_failures", [])
    if fails:
        print("\nQUALITY GATE FAILURES:")
        for f in fails:
            print(f"  ! {f}")
    else:
        print("\nAll quality gates passed.")
    print(f"\nTables written to {TABDIR}")


def cmd_validate(args):
    scores = pipeline.stage_validate(
        seeds=tuple(args.seeds), n_cases=args.n_cases, n_controls=args.n_controls
    )
    _print_df(scores, "%.3f")


def cmd_figures(args):
    made = pipeline.stage_figures()
    for m in made:
        print(f"  {m}")
    print(f"\n{len(made)} figures written.")


def cmd_report(args):
    from .report import write_report

    path = write_report()
    print(f"Report written to {path}")


def cmd_all(args):
    if args.recalibrate or not pipeline.CALIBRATION_PATH.exists():
        pipeline.stage_calibrate(n=args.cal_n)
    else:
        print(f"Reusing calibration at {pipeline.CALIBRATION_PATH} "
              "(pass --recalibrate to refit).")
    pipeline.stage_simulate(n_cases=args.n_cases, n_controls=args.n_controls)
    summary = pipeline.stage_analyse()
    pipeline.stage_validate(seeds=tuple(args.seeds), n_cases=max(args.n_cases // 2, 200),
                            n_controls=max(args.n_controls // 2, 1000))
    pipeline.stage_figures()
    from .report import write_report

    path = write_report()
    fails = summary.get("gate_failures", [])
    print(f"\nReport: {path}")
    print("Quality gates: " + ("PASSED" if not fails else f"{len(fails)} FAILED"))
    for f in fails:
        print(f"  ! {f}")
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rdd", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("calibrate", help="fit friction scalars to literature anchors")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--tol", type=float, default=0.05)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("simulate", help="build the synthetic claims database")
    p.add_argument("--n-cases", type=int, default=None)
    p.add_argument("--n-controls", type=int, default=None)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("ingest", help="load a real claims source")
    p.add_argument("adapter", choices=["synpuf", "generic", "omop"])
    p.add_argument("path")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("doctor", help="check whether a source can support the estimate")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("analyse", help="run the analysis stages")
    p.add_argument("--lookback-years", type=float, default=None)
    p.set_defaults(func=cmd_analyse)

    p = sub.add_parser("validate", help="parameter recovery across seeds")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--n-cases", type=int, default=400)
    p.add_argument("--n-controls", type=int, default=2000)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("figures", help="render figures")
    p.set_defaults(func=cmd_figures)

    p = sub.add_parser("report", help="write the methods report")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("all", help="run everything")
    p.add_argument("--n-cases", type=int, default=1200)
    p.add_argument("--n-controls", type=int, default=6000)
    p.add_argument("--cal-n", type=int, default=1500)
    p.add_argument("--recalibrate", action="store_true",
                   help="refit friction scalars instead of reusing the committed ones")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.set_defaults(func=cmd_all)

    args = ap.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
