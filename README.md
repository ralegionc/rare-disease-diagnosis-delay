# Rare disease diagnosis delay

Modelling the diagnostic odyssey — symptom onset to correct diagnosis — from
administrative claims, for six rare conditions whose odysseys differ by more
than a decade.

The headline number everyone quotes ("rare disease patients wait 5–7 years")
is a **latent** quantity. Claims record billing, not biology: the earliest
event they can contain is the first time a patient sought care *and someone
coded something*. Symptom onset leaves no trace at all. So every claims-based
estimate of diagnostic delay is an estimate of an unobservable quantity through
a proxy, under censoring, inside a database with finite history.

This repo builds that estimator properly — and then measures how wrong it is,
by running the whole pipeline against a simulator where the latent quantity is
known by construction.

## The three headline results

All from simulated claims (`[SYNTHETIC]` in the report), so they are evidence
about the **estimator**, not about real patients.

1. **The retrospective design understates delay for every disease, and the
   longer the true odyssey the worse it gets.** Hypermobile EDS: 12 years true,
   ~6 years estimated.
2. **That bias is measurement, not selection.** Restricting ground truth to
   exactly the patients the analysis kept barely moves it — the damage is done
   by where the clock starts, not by who is in the sample. Selection bias is
   fixable by study design; this one is not.
3. **The ranking survives even though the levels do not.** Rank correlation
   between estimated and true median delay stays high while absolute bias is
   large. Claims can tell you reliably *which* rare diseases have the worst
   odysseys while being systematically wrong about *how bad* any of them are.

Plus two that came out of building the validation harness rather than being
looked for:

4. **The disease with the highest mortality shows the *smallest*
   competing-risk correction.** ALS kills fast enough that most patients who
   die undiagnosed never accumulate a symptom signature and so never enter the
   cohort. Mortality selected upstream of the estimator, where no
   competing-risk method can reach. A small Aalen–Johansen correction is not
   reassurance that death did not matter.
5. **Agreement with truth is strongly sample-size dependent, at the sizes real
   studies operate at.** Correlation rises from r=0.81 at 400 simulated cases
   per disease to r=0.98 at 1,000 — because only 50–90 of each thousand ever
   reach the analysable cohort. A delay estimate for one rare disease from a
   few dozen patients is not precise enough to compare against another disease
   or another era, however carefully it was computed.

## What's in it

```
src/rdd/
  profiles.py     6 disease profiles, literature-anchored
  simulate.py     generative model of the odyssey + calibration by bisection
  schema.py       canonical claims schema every source normalises into
  adapters/       CMS DE-SynPUF, OMOP CDM v5.4, generic CSV
  cohort.py       claims phenotype, onset proxy, two competing study designs
  states.py       claims -> odyssey state process
  survival.py     KM, log-rank, Nelson-Aalen, Aalen-Johansen, parametric MLE
  multistate.py   semi-Markov sojourn fits + state occupancy
  decompose.py    where the years go
  bias.py         look-back ceiling, design comparison, competing risks
  validate.py     parameter recovery + quality gates
  figures.py      publication figures
  report.py       methods report, provenance-marked
```

Dependencies: numpy, scipy, pandas, matplotlib, pyarrow. **No survival library**
— Kaplan–Meier, log-rank, Nelson–Aalen, Aalen–Johansen and the censored
parametric MLEs are ~300 auditable lines of numpy, because the combination this
problem needs (left truncation + competing risks + semi-Markov sojourns) is
exactly where the wrappers stop agreeing with each other. Nothing touches the
network; a test asserts it.

## Run it

```bash
pip install -e ".[dev]"
rdd all                    # calibrate -> simulate -> analyse -> validate -> figures -> report
pytest                     # 99 tests
```

Individual stages:

```bash
rdd calibrate              # fit friction scalars to the published delays
rdd simulate               # build the synthetic claims database
rdd doctor                 # can this source support the estimate at all?
rdd analyse                # cohorts, multi-state fits, decomposition, bias
rdd validate --seeds 1 2 3 # recovery across independent replicates
rdd figures && rdd report
```

`docs/REAL_DATA_RUNBOOK.md` covers how to get from here to real data.

Output lands in `outputs/`: `report.md`, `figures/*.png`, `tables/*.csv`.

## Running it on real claims

```bash
rdd ingest generic /path/to/extract     # two CSVs, see adapters/generic.py
rdd ingest synpuf  /path/to/synpuf      # CMS public file
rdd doctor                              # run this BEFORE analyse
rdd analyse && rdd figures && rdd report
```

Always run `rdd doctor` before `rdd analyse` on a new source. It answers
*can this database support the estimate at all* before anything computes one,
catching the four failure modes that produce a plausible-looking number rather
than an error: wrong code vocabulary, a look-back shorter than the delay,
provider specialty collapsed to `OTHER`, and a cohort too small to mean
anything. It exits non-zero on a blocker.

Or against an OMOP CDM instance:

```python
from rdd.adapters import load_omop
ds, notes = load_omop(connection)       # any DBAPI connection
ds.write("data/interim/dataset")
```

Every adapter returns a `notes` list naming what it had to substitute, and the
CLI prints it. A missing death date or an unmapped provider specialty is a
result-changing problem, and an adapter that filled those in quietly would push
a data problem somewhere nobody can see it.

**A warning about DE-SynPUF specifically.** It is the only public dataset with
real Medicare claims *structure*, and it is the right thing to point the ingest
path at first. It is also unusable as evidence here, on four independent
grounds documented in `adapters/synpuf.py`: three years of history (see result
1 above), a Medicare age range that postdates onset for five of the six
conditions, per-variable synthesis that breaks the longitudinal coherence this
entire analysis depends on, and no rendering-provider specialty. Use it to
smoke-test ingest at real volume. Do not use it to estimate a delay.

## Design decisions worth arguing about

**The simulator is the point, not a stand-in for data.** The whole project rests
on being able to score estimators against a known answer. `rdd validate` does
that across independent seeds; four quality gates are asserted as tests, and
two of them assert that the *bias is still present* — if the retrospective
design ever stopped understating delay on this simulator, the simulator would
have stopped reproducing the problem, and that is a failure even though it
looks like an improvement.

**Calibration cannot flatter the estimators.** One friction scalar per disease
is fitted by bisection so the simulated *true* median delay matches its
published anchor. The target is ground truth, not anything the pipeline
estimates. A single monotone scalar is deliberate: a richer calibration would
hit the anchors trivially and prove nothing.

**Wrong labels are modelled as suppressors, not as wasted visits.** A
misdiagnosis *explains* the symptoms, so it stops diagnostic effort for a dwell
time. This is what separates hEDS (fibromyalgia, 3.4-year dwell, assigned to
roughly half of patients) from ALS (cervical spondylosis, 0.55-year dwell,
disproved by its own failed surgery). The fitted Weibull sojourn shape in the
mislabelled state comes out below 1 — the exit hazard *falls* the longer the
label has held, which is what a settled misdiagnosis looks like quantitatively.

**Two study designs, reported as bounds.** Diagnosed-anchored (what nearly all
published claims studies do) is optimistic; suspicion-anchored is contaminated
by phenocopies. Neither is correct. The pair brackets the answer, and the
contamination is at least measurable, whereas the selection bias it replaces is
not.

**One phase is not modelled, because it cannot be.** Symptom onset to first
care contact generates no claim. It is absent from administrative data rather
than noisy, and its size is the honest ceiling on what any claims-only study
can see. `decompose.invisible_phase` measures it from ground truth and the
report flags it as unobservable.

## Status

Validated end to end on the simulator. **Not** run on real claims — MarketScan,
Optum and the Medicare LDS are licensed, and DE-SynPUF is unsuitable for the
reasons above. The adapters exist so the analysis runs unchanged the moment a
real source is available.

What a real run would newly establish: whether the phase decomposition holds
outside the simulator's assumptions; whether the mislabelled-state hazard shape
really is below 1 in practice; how large the measurement bias is against an
external onset date; and whether the look-back ceiling explains the
disagreement between published delay estimates for the same condition.

## Sources for the profile anchors

- ALS: [Time to Diagnosis and Factors Affecting Diagnostic Delay in ALS](https://www.ncbi.nlm.nih.gov/books/NBK573423/) — 10–16 months onset-to-diagnosis; ~49% see a non-neurology specialist first; initial misdiagnosis 13–68%
- Hereditary angioedema: [Improvement in diagnostic delays over time in HAE (Icatibant Outcome Survey)](https://link.springer.com/article/10.1186/s13601-018-0229-4)
- Fabry / Gaucher / MPS II patient journeys: [PLOS ONE 2020 German-wide telephone survey](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0244279)
- hEDS: [How long does it take to get diagnosed with EDS?](https://www.eds.clinic/articles/hard-to-get-eds-diagnosis)

Anchors are literature-informed *priors* that set the simulator's ground truth,
not fitted values, and nothing in the analysis pipeline reads them.

## Licence

MIT.
