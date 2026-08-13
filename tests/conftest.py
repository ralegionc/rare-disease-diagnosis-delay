from __future__ import annotations

from dataclasses import replace

import pytest

from rdd.config import DEFAULT_SIM, DEFAULT_STUDY
from rdd.simulate import simulate_population

# Frictions fitted by `rdd calibrate`. Pinned here so the test suite never
# spends a minute re-fitting them, and so a change in the simulator that moves
# the calibration shows up as a failing anchor test rather than silently
# re-fitting itself back into agreement.
CALIBRATION = {
    "ALS": 0.05,
    "GD1": 0.5148,
    "LOPD": 0.6078,
    "HAE": 0.7937,
    "hEDS": 0.5148,
    "FABRY": 0.5613,
}


@pytest.fixture(scope="session")
def small_dataset():
    """A small simulated database, shared across the suite."""
    sim = replace(DEFAULT_SIM, n_cases_per_disease=150, n_controls=700, seed=11)
    return simulate_population(sim, DEFAULT_STUDY, CALIBRATION)


@pytest.fixture(scope="session")
def medium_dataset():
    sim = replace(DEFAULT_SIM, n_cases_per_disease=400, n_controls=2000, seed=12)
    return simulate_population(sim, DEFAULT_STUDY, CALIBRATION)


@pytest.fixture(scope="session")
def recovery_dataset():
    """Large enough for the recovery gates to be about the estimator.

    The diagnosed cohort is a small fraction of simulated cases -- most
    patients are diagnosed outside their coverage window or not at all -- so
    1,000 cases per disease yields only 50-90 analysable odysseys each. Below
    roughly that, agreement with ground truth is dominated by sampling noise in
    the per-disease medians rather than by any property of the method (r rises
    from 0.81 at 400 cases to 0.98 at 1,000 on the same seed). That instability
    is itself a finding about real rare-disease claims studies, recorded in
    `rdd.validate.SAMPLE_SIZE_NOTE`; here it just means the gates need a
    fixture that clears it.
    """
    sim = replace(DEFAULT_SIM, n_cases_per_disease=1000, n_controls=5000, seed=12)
    return simulate_population(sim, DEFAULT_STUDY, CALIBRATION)


@pytest.fixture(scope="session")
def study():
    return DEFAULT_STUDY
