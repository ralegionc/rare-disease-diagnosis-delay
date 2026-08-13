"""rdd -- modelling the rare-disease diagnostic odyssey from administrative claims.

The quantity everyone quotes -- "five to seven years to diagnosis" -- is a
latent one. Claims record billing, not biology, so symptom onset leaves no
trace and the earliest observable event is the first coded encounter. This
package builds the estimator anyway, and then measures how wrong it is against
a simulator where the latent quantity is known by construction.

Typical use::

    from rdd import pipeline
    pipeline.stage_calibrate()
    ds = pipeline.stage_simulate()
    pipeline.stage_analyse(ds)

or from a shell::

    rdd all
"""

from .config import DEFAULT_SIM, DEFAULT_STUDY, SimConfig, StudyConfig
from .profiles import PANEL, PROFILES, DiseaseProfile
from .schema import ClaimsDataset

__version__ = "0.1.0"

__all__ = [
    "ClaimsDataset",
    "DiseaseProfile",
    "PANEL",
    "PROFILES",
    "SimConfig",
    "StudyConfig",
    "DEFAULT_SIM",
    "DEFAULT_STUDY",
    "__version__",
]
