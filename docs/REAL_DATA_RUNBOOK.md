# Getting this onto real data

*Written August 2026. Access terms and prices change — re-check before acting
on any row.*

The code is finished. **Access is the blocker**, and it is worth being blunt
about the shape of it: longitudinal claims with provider specialty, a decade of
continuous enrolment, and enough rare-disease patients to matter is a
commercial product. There is no free version of exactly that.

But "no real claims" is not the same as "nothing to do next", and the single
highest-value next step costs nothing and needs no permission — see rung 1.

## First, always

```bash
rdd ingest <adapter> <path>
rdd doctor                       # <- before anything else
```

`rdd doctor` answers *can this database support the estimate at all* before
anything computes one. It catches the four failure modes that produce a
plausible-looking number rather than an error: wrong code vocabulary, a
look-back shorter than the delay, specialty collapsed to `OTHER`, and a cohort
too small to mean anything. It exits non-zero on a blocker.

Do not skip it. An ICD-9 source against ICD-10 profiles produces an empty
cohort and a report full of blank tables, and **nothing raises**.

---

## The ladder

### Rung 1 — Synthea. Free, today, no gatekeeper. **Do this first.**

[Synthea](https://synthetichealth.github.io/synthea/) is an open-source
synthetic patient generator from MITRE. Unlike CMS DE-SynPUF it produces
**longitudinally coherent** records — real trajectories, not per-variable
synthesis — which is the property this whole analysis depends on.

It is not real data. It is something arguably more useful right now:

> The estimators are currently validated only against `rdd.simulate` — a
> simulator that shares their own assumptions. That is circular, and it is the
> first hole a reviewer will poke. Synthea is an **independently built
> generative model** with different authors, different mechanics and different
> biases. If the estimators recover sensible structure there too, the finding
> stops being about my simulator.

Work involved:

1. Generate a population with the standard modules (`./run_synthea -p 50000`).
2. Write custom Generic Module Framework modules (JSON state machines) for as
   many panel diseases as you can — Synthea ships none of the six. This is the
   real work, and it is genuinely portfolio-worthy on its own.
3. ETL Synthea CSV output → the canonical schema via
   `adapters.generic.load_generic`. Synthea emits `conditions.csv`,
   `encounters.csv`, `patients.csv`, `providers.csv` — the mapping is close to
   direct. Provider specialty is available, which SynPUF lacks.
4. `rdd doctor`, then `rdd analyse`.

The honest write-up line: *"cross-validated against an independently authored
generative model."* That is a substantially stronger claim than the repo makes
today.

Also worth checking: [SYNRARE](https://arxiv.org/html/2607.09404), a 2026
synthetic rare-disease EHR benchmark. Newer and less proven than Synthea, but
purpose-built for this problem.

### Rung 2 — CMS DE-SynPUF. Free, today, but it cannot answer the question.

Already has an adapter (`adapters/synpuf.py`). Use it to prove the ingest path
at real volume and real messiness. **Do not use it to estimate a delay** — the
adapter documents four independent reasons, and `rdd doctor` will flag two of
them automatically:

- three years of history (2008–2010) — see finding 4, the look-back ceiling
- Medicare age range; onset for five of six panel diseases predates it
- per-variable synthesis breaks longitudinal coherence
- no rendering-provider specialty → multi-state model collapses

Requires the ICD-9 crosswalk (open work item 3).

### Rung 3 — All of Us Researcher Workbench. Free, but needs an institution.

Real EHR data, already in **OMOP CDM** — the `adapters/omop.py` path runs
against it nearly as-is. $300 initial compute credits, no data fee.

The catch: **your institution must have signed a Data Use and Registration
Agreement.** There is no independent-researcher pathway as of August 2026. If
you are enrolled anywhere, or can get an affiliation through a collaborator,
check whether a DURA already exists — many universities have one and most
students don't know.

Registered Tier gives de-identified EHR + surveys. Note this is EHR, not
claims: encounter coverage is a health-system's view rather than a payer's, so
enrolment spans mean something different and `check_enrolment` needs
re-reading before you trust it.

### Rung 4 — State All-Payer Claims Databases. Weeks to months, hundreds to
low thousands of dollars.

Genuine longitudinal multi-payer claims.
[Massachusetts CHIA](https://www.chiamass.gov/ma-apcd/) is the best-documented;
Colorado, Minnesota, Maine, Virginia and others run comparable programs.
Application + data-use agreement + IRB or an equivalent review. Fees vary from
nominal for limited files to low thousands for the full set.

Best realistic route to *actual* claims for an unaffiliated researcher. Budget
months for the paperwork, not days.

### Rung 5 — Commercial. Not realistic for a portfolio project.

Merative MarketScan, Optum CDM, Komodo, IQVIA. Tens of thousands of dollars
upward, usually institutional. Listed for completeness. If you land somewhere
with a licence, `adapters/generic.py` is the contract to map onto and rung 4's
notes apply.

### Deliberately not on the ladder

- **MIMIC-IV** — free and credentialing accepts independent researchers, but it
  is ICU-only. No longitudinal outpatient trajectory, which is the entire
  object of study. Wrong shape, not merely small.
- **N3C** — OMOP and rich, but institutional DUA required, same gate as rung 3.
- **SEER-Medicare** — cancer registry linkage. Wrong diseases.

---

## What "real data" changes about the analysis

Do not just re-run and publish the numbers. Three things need rethinking:

1. **The `[SYNTHETIC]` markers become real claims.** Every marked sentence in
   `report.py` needs revisiting — several are statements about the simulator
   that would be *unsupported* as statements about patients.

2. **The invisible phase stops being measurable.** `decompose.invisible_phase`
   reads the truth table. On real data there is no truth table, so onset→first
   care is simply unknown. Report the decomposition as a share of *observed*
   time and say plainly that a phase of unknown size sits before it.

3. **The validation story inverts.** Right now: "here is how wrong the
   estimator is." On real data you cannot know that — there is no ground truth.
   What you can do is carry the *simulator-derived* bias estimates across as a
   sensitivity analysis: "the retrospective design understated delay by 34% on
   simulated data with comparable censoring; the real estimate should be read
   as a lower bound of similar magnitude." That framing is the point of having
   built the harness, and it is what makes this project worth more than a
   Kaplan–Meier curve.

## Cheapest experiment that would materially strengthen the repo

Rung 1, restricted to two diseases. Write Synthea modules for hEDS (long,
indolent, heavy mislabelling) and ALS (fast, fatal). They sit at opposite ends
of the panel, so if the estimator orders them correctly and reproduces the
mislabelled-state hazard shape on someone else's generative model, findings 1,
3 and 6 stop being circular. That is a weekend or two, and it costs nothing.
