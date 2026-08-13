"""Disease profiles for the simulated panel.

Six conditions were chosen because they *disagree* with each other along the
axes that plausibly drive diagnostic delay. A panel that is all lysosomal
storage disorders would let a model look good for the wrong reason.

===============  ========  ==========  ==================  ====================
profile          anchor    tempo       presenting setting  dominant obstacle
===============  ========  ==========  ==================  ====================
ALS              ~1 yr     fast        neurology office    competing mortality
GD1              ~4 yr     moderate    haematology         looks like cancer
LOPD             ~8 yr     slow        neurology           looks like LGMD
HAE              ~10 yr    episodic    emergency dept      episodes resolve
hEDS             ~12 yr    indolent    primary/ortho       symptoms unspecific
FABRY            ~14 yr    multiphase  scattered           no single owner
===============  ========  ==========  ==================  ====================

Provenance of the anchors
-------------------------
``target_median_delay_years`` and the misdiagnosis sets are *literature-anchored
priors*, not fitted values. They set the ground truth the simulator generates
from; nothing in the analysis pipeline reads them. Sources:

* ALS: 10-16 months onset-to-diagnosis; ~49% see a non-neurology specialist
  first; reported initial-misdiagnosis rates 13-68%, commonly cervical
  myelopathy, radiculopathy and neuropathy.
  Richards et al., in *Amyotrophic Lateral Sclerosis* (NCBI Bookshelf NBK573423).
* Gaucher type 1: multi-year delay with haematological workup for
  thrombocytopenia/splenomegaly preceding the correct diagnosis; bone-marrow
  biopsy is a characteristic wasted-procedure marker.
* Late-onset Pompe: multi-year delay, characteristically mislabelled as
  limb-girdle muscular dystrophy or polymyositis before enzyme assay.
* Hereditary angioedema: historically ~8-13 years; abdominal attacks drive
  emergency presentations and unnecessary laparotomy/appendicectomy.
  Zanichelli et al., *Clin Transl Allergy* (Icatibant Outcome Survey).
* Hypermobile EDS: ~10-14 years, strongly female-skewed, with fibromyalgia,
  chronic fatigue and anxiety as the dominant intermediate labels.
* Fabry: ~13-16 years, longer in females; childhood acroparaesthesia
  misread as growing pains/rheumatic complaints, organ involvement arriving
  a decade later. Mehta et al., FOS registry.

The point of the panel is contrast, not epidemiological precision. Where the
literature reports a range we take a round number inside it, and
``tests/test_profiles.py`` asserts the panel spans at least a decade of delay
so the contrast cannot silently collapse.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Symptom:
    """A billable manifestation of the disease.

    ``specificity`` is the probability that a clinician seeing *this code alone*
    would put the target disease in the differential. It is the quantity that
    makes rare-disease delay hard: high-prevalence symptoms with specificity
    near zero (fatigue, joint pain) generate claims without generating suspicion.
    """

    code: str
    label: str
    prob: float  # probability the patient ever develops it
    onset_lag_years: float  # mean years after true onset before it appears
    specificity: float  # 0 = generic, 1 = pathognomonic
    owning_specialty: str
    settings: tuple[str, ...] = ("OFFICE",)

    def __post_init__(self):
        for name in ("prob", "specificity"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{self.code}.{name} must be in [0, 1], got {v}")
        if self.onset_lag_years < 0:
            raise ValueError(f"{self.code}.onset_lag_years must be >= 0")


@dataclass(frozen=True)
class Misdiagnosis:
    """An incorrect label the patient may acquire on the way.

    ``dwell_years`` is the mechanism that actually generates delay in this
    model. A wrong label is not merely a wasted visit: it *explains* the
    symptoms, so it suppresses further diagnostic effort for as long as it
    holds. Labels that are cheap to sustain and hard to falsify (fibromyalgia,
    anxiety) carry long dwells; labels that get disproved by their own
    treatment failure (appendicitis, ITP) carry short ones.
    """

    code: str
    label: str
    prob: float
    dwell_years: float
    specialty: str
    #: Procedures the wrong label tends to trigger -- the iatrogenic footprint.
    procedures: tuple[str, ...] = ()

    def __post_init__(self):
        if not 0.0 <= self.prob <= 1.0:
            raise ValueError(f"{self.code}.prob must be in [0, 1]")
        if self.dwell_years <= 0:
            raise ValueError(f"{self.code}.dwell_years must be > 0")


@dataclass(frozen=True)
class DiseaseProfile:
    key: str
    name: str
    dx_code: str
    onset_age_mean: float
    onset_age_sd: float
    female_fraction: float
    symptoms: tuple[Symptom, ...]
    misdx: tuple[Misdiagnosis, ...]
    recognizing_specialties: tuple[str, ...]
    #: Per-visit probability that a recognizing specialist actually makes the
    #: call, given the patient has accumulated enough specific findings.
    recognition_prob: float
    #: Referral propensity: where a primary-care physician sends this patient.
    referral_mix: dict[str, float]
    confirm_codes: tuple[str, ...]
    #: Patient-side delay (onset -> first care contact), lognormal in years.
    care_seeking_median_years: float
    care_seeking_logsd: float
    #: Encounters per year while symptomatic and undiagnosed.
    visit_rate: float
    #: Constant hazard of death per year from onset (competing risk).
    death_hazard: float
    target_median_delay_years: float
    reference: str
    prevalence_per_100k: float = 1.0
    inheritance: str = ""

    def __post_init__(self):
        if not self.symptoms:
            raise ValueError(f"{self.key}: needs at least one symptom")
        total = sum(self.referral_mix.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"{self.key}: referral_mix sums to {total}, not 1")
        if not self.recognizing_specialties:
            raise ValueError(f"{self.key}: needs a recognizing specialty")

    # -- convenience -------------------------------------------------------

    @property
    def symptom_codes(self) -> tuple[str, ...]:
        return tuple(s.code for s in self.symptoms)

    @property
    def misdx_codes(self) -> tuple[str, ...]:
        return tuple(m.code for m in self.misdx)

    @property
    def specific_symptom_codes(self) -> tuple[str, ...]:
        """Codes a competent clinician might actually act on."""
        return tuple(s.code for s in self.symptoms if s.specificity >= 0.30)


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------

ALS = DiseaseProfile(
    key="ALS",
    name="Amyotrophic lateral sclerosis",
    dx_code="G12.21",
    onset_age_mean=58.0,
    onset_age_sd=11.0,
    female_fraction=0.44,
    prevalence_per_100k=5.2,
    inheritance="mostly sporadic",
    symptoms=(
        Symptom("M62.81", "generalised muscle weakness", 0.95, 0.0, 0.10, "PRIMARY_CARE"),
        Symptom("R29.898", "abnormal gait / frequent falls", 0.55, 0.25, 0.12, "PRIMARY_CARE"),
        Symptom("R47.81", "slurred speech", 0.40, 0.35, 0.35, "NEUROLOGY"),
        Symptom("R13.10", "dysphagia", 0.45, 0.60, 0.30, "GASTROENTEROLOGY"),
        Symptom("R25.2", "cramp and fasciculation", 0.80, 0.15, 0.55, "NEUROLOGY"),
        Symptom("R06.00", "dyspnoea", 0.35, 0.90, 0.15, "PULMONOLOGY"),
    ),
    misdx=(
        Misdiagnosis("M47.12", "cervical spondylosis with myelopathy", 0.30, 0.55,
                     "ORTHOPEDICS", procedures=("72141", "63075")),
        Misdiagnosis("G56.00", "carpal tunnel syndrome", 0.16, 0.35, "ORTHOPEDICS",
                     procedures=("64721",)),
        Misdiagnosis("M54.12", "cervical radiculopathy", 0.20, 0.40, "ORTHOPEDICS"),
        Misdiagnosis("G70.00", "myasthenia gravis", 0.08, 0.30, "NEUROLOGY"),
    ),
    recognizing_specialties=("NEUROLOGY",),
    recognition_prob=0.55,
    referral_mix={
        "NEUROLOGY": 0.51, "ORTHOPEDICS": 0.24, "SURGERY": 0.06,
        "RHEUMATOLOGY": 0.06, "PULMONOLOGY": 0.05, "OTHER": 0.08,
    },
    confirm_codes=("95886", "95861"),  # needle EMG with nerve conduction
    care_seeking_median_years=0.30,
    care_seeking_logsd=0.60,
    visit_rate=6.5,
    death_hazard=0.32,  # median survival ~3 yr from onset
    target_median_delay_years=1.0,
    reference="NCBI Bookshelf NBK573423 (10-16 months onset-to-diagnosis)",
)

GD1 = DiseaseProfile(
    key="GD1",
    name="Gaucher disease type 1",
    dx_code="E75.22",
    onset_age_mean=21.0,
    onset_age_sd=14.0,
    female_fraction=0.51,
    prevalence_per_100k=1.2,
    inheritance="autosomal recessive",
    symptoms=(
        Symptom("R16.1", "splenomegaly", 0.90, 0.0, 0.45, "HEMATOLOGY"),
        Symptom("D69.6", "thrombocytopenia, unspecified", 0.80, 0.10, 0.25, "HEMATOLOGY"),
        Symptom("D64.9", "anaemia, unspecified", 0.60, 0.30, 0.08, "PRIMARY_CARE"),
        Symptom("M25.559", "bone/hip pain", 0.55, 0.80, 0.20, "ORTHOPEDICS"),
        Symptom("R53.83", "fatigue", 0.70, 0.0, 0.02, "PRIMARY_CARE"),
        Symptom("M87.059", "avascular necrosis of femur", 0.18, 2.50, 0.55, "ORTHOPEDICS"),
    ),
    misdx=(
        Misdiagnosis("D69.3", "immune thrombocytopenic purpura", 0.34, 1.20, "HEMATOLOGY",
                     procedures=("38221", "85097")),  # bone marrow biopsy
        Misdiagnosis("C85.90", "lymphoma, unspecified (workup)", 0.14, 0.45, "HEMATOLOGY",
                     procedures=("38221", "78815")),
        Misdiagnosis("K76.9", "hepatic disease, unspecified", 0.12, 0.90, "GASTROENTEROLOGY"),
        Misdiagnosis("M25.559", "idiopathic bone pain", 0.20, 1.50, "ORTHOPEDICS"),
    ),
    recognizing_specialties=("HEMATOLOGY", "GENETICS"),
    recognition_prob=0.42,
    referral_mix={
        "HEMATOLOGY": 0.46, "GASTROENTEROLOGY": 0.16, "ORTHOPEDICS": 0.16,
        "GENETICS": 0.06, "SURGERY": 0.06, "OTHER": 0.10,
    },
    confirm_codes=("82657", "81251"),  # beta-glucosidase assay, GBA sequencing
    care_seeking_median_years=0.60,
    care_seeking_logsd=0.75,
    visit_rate=5.0,
    death_hazard=0.012,
    target_median_delay_years=4.0,
    reference="Patient-journey surveys of lysosomal storage disorders (PLOS ONE 2020)",
)

LOPD = DiseaseProfile(
    key="LOPD",
    name="Late-onset Pompe disease",
    dx_code="E74.02",
    onset_age_mean=35.0,
    onset_age_sd=14.0,
    female_fraction=0.50,
    prevalence_per_100k=0.7,
    inheritance="autosomal recessive",
    symptoms=(
        Symptom("M62.81", "proximal muscle weakness", 0.95, 0.0, 0.12, "PRIMARY_CARE"),
        Symptom("R29.898", "difficulty rising from a chair / waddling gait", 0.70, 0.40, 0.18,
                "NEUROLOGY"),
        Symptom("R06.00", "exertional dyspnoea", 0.60, 1.20, 0.10, "PULMONOLOGY"),
        Symptom("G47.30", "sleep-disordered breathing", 0.45, 2.00, 0.22, "PULMONOLOGY"),
        Symptom("R79.82", "raised serum creatine kinase", 0.85, 0.30, 0.40, "NEUROLOGY",
                settings=("LAB",)),
        Symptom("M62.5", "muscle wasting", 0.40, 2.50, 0.25, "NEUROLOGY"),
    ),
    misdx=(
        Misdiagnosis("G71.09", "limb-girdle muscular dystrophy", 0.32, 2.20, "NEUROLOGY",
                     procedures=("20200", "88305")),  # muscle biopsy
        Misdiagnosis("M33.20", "polymyositis", 0.22, 1.40, "RHEUMATOLOGY",
                     procedures=("20200", "J1100")),
        Misdiagnosis("G72.9", "myopathy, unspecified", 0.28, 1.80, "NEUROLOGY"),
        Misdiagnosis("M79.7", "fibromyalgia", 0.14, 2.00, "RHEUMATOLOGY"),
    ),
    recognizing_specialties=("NEUROLOGY", "GENETICS"),
    recognition_prob=0.30,
    referral_mix={
        "NEUROLOGY": 0.44, "RHEUMATOLOGY": 0.18, "PULMONOLOGY": 0.14,
        "ORTHOPEDICS": 0.08, "GENETICS": 0.05, "OTHER": 0.11,
    },
    confirm_codes=("82657", "81479"),  # acid alpha-glucosidase (DBS), GAA sequencing
    care_seeking_median_years=1.10,
    care_seeking_logsd=0.85,
    visit_rate=4.2,
    death_hazard=0.022,
    target_median_delay_years=8.0,
    reference="Late-onset Pompe diagnostic-delay literature (multi-year, LGMD mislabel)",
)

HAE = DiseaseProfile(
    key="HAE",
    name="Hereditary angioedema (C1-inhibitor deficiency)",
    dx_code="D84.1",
    onset_age_mean=12.0,
    onset_age_sd=6.5,
    female_fraction=0.56,
    prevalence_per_100k=2.0,
    inheritance="autosomal dominant",
    symptoms=(
        Symptom("T78.3XXA", "angioedema", 0.95, 0.0, 0.30, "ALLERGY_IMMUNOLOGY",
                settings=("ER", "OFFICE")),
        Symptom("R10.9", "recurrent severe abdominal pain", 0.75, 0.20, 0.05, "EMERGENCY",
                settings=("ER", "OFFICE")),
        Symptom("R11.2", "nausea with vomiting", 0.55, 0.25, 0.02, "EMERGENCY",
                settings=("ER",)),
        Symptom("J38.4", "laryngeal oedema", 0.25, 1.50, 0.60, "EMERGENCY",
                settings=("ER", "INPATIENT")),
        Symptom("L50.9", "urticaria (absent -- coded when misread)", 0.30, 0.10, 0.02,
                "DERMATOLOGY"),
    ),
    misdx=(
        Misdiagnosis("T78.40XA", "allergic reaction / histaminergic angioedema", 0.55, 2.60,
                     "ALLERGY_IMMUNOLOGY", procedures=("95004", "J0171")),
        Misdiagnosis("K35.80", "acute appendicitis", 0.18, 0.30, "SURGERY",
                     procedures=("44970",)),  # laparoscopic appendicectomy
        Misdiagnosis("K58.9", "irritable bowel syndrome", 0.30, 2.20, "GASTROENTEROLOGY",
                     procedures=("45378",)),
        Misdiagnosis("F45.8", "somatoform disorder", 0.14, 1.80, "PSYCHIATRY"),
    ),
    recognizing_specialties=("ALLERGY_IMMUNOLOGY", "GENETICS"),
    recognition_prob=0.38,
    referral_mix={
        "ALLERGY_IMMUNOLOGY": 0.34, "GASTROENTEROLOGY": 0.24, "SURGERY": 0.12,
        "DERMATOLOGY": 0.10, "PSYCHIATRY": 0.06, "OTHER": 0.14,
    },
    confirm_codes=("86160", "86161"),  # C4 and C1-inhibitor quantitation
    care_seeking_median_years=0.25,
    care_seeking_logsd=0.55,
    visit_rate=5.5,
    death_hazard=0.006,  # laryngeal attacks
    target_median_delay_years=10.0,
    reference="Zanichelli et al., Clin Transl Allergy 2018 (Icatibant Outcome Survey)",
)

HEDS = DiseaseProfile(
    key="hEDS",
    name="Hypermobile Ehlers-Danlos syndrome",
    dx_code="M35.7",
    onset_age_mean=14.0,
    onset_age_sd=7.0,
    female_fraction=0.85,
    prevalence_per_100k=200.0,
    inheritance="autosomal dominant (gene unknown)",
    symptoms=(
        Symptom("M25.50", "polyarticular joint pain", 0.95, 0.0, 0.04, "PRIMARY_CARE"),
        Symptom("M24.40", "recurrent joint subluxation", 0.65, 0.80, 0.45, "ORTHOPEDICS"),
        Symptom("R53.83", "fatigue", 0.85, 0.50, 0.02, "PRIMARY_CARE"),
        Symptom("I95.1", "orthostatic hypotension / presyncope", 0.50, 1.50, 0.20, "CARDIOLOGY"),
        Symptom("K59.00", "constipation / GI dysmotility", 0.45, 1.20, 0.05,
                "GASTROENTEROLOGY"),
        Symptom("L91.8", "atrophic scarring, skin hyperextensibility", 0.30, 1.00, 0.50,
                "DERMATOLOGY"),
        Symptom("F41.9", "anxiety", 0.55, 0.60, 0.01, "PSYCHIATRY"),
    ),
    misdx=(
        Misdiagnosis("M79.7", "fibromyalgia", 0.48, 3.40, "RHEUMATOLOGY"),
        Misdiagnosis("R53.82", "chronic fatigue syndrome", 0.30, 3.00, "PRIMARY_CARE"),
        Misdiagnosis("F41.1", "generalised anxiety disorder", 0.40, 2.80, "PSYCHIATRY"),
        Misdiagnosis("M54.5", "non-specific low back pain", 0.45, 1.60, "ORTHOPEDICS",
                     procedures=("72148", "97110")),
        Misdiagnosis("F45.8", "somatoform / functional disorder", 0.22, 2.40, "PSYCHIATRY"),
    ),
    recognizing_specialties=("GENETICS", "RHEUMATOLOGY"),
    recognition_prob=0.22,
    referral_mix={
        "ORTHOPEDICS": 0.30, "RHEUMATOLOGY": 0.22, "PSYCHIATRY": 0.14,
        "GASTROENTEROLOGY": 0.10, "CARDIOLOGY": 0.08, "GENETICS": 0.04,
        "DERMATOLOGY": 0.04, "OTHER": 0.08,
    },
    confirm_codes=("99244",),  # extended genetics consultation; hEDS is clinical
    care_seeking_median_years=1.40,
    care_seeking_logsd=0.90,
    visit_rate=7.5,
    death_hazard=0.002,
    target_median_delay_years=12.0,
    reference="hEDS diagnostic-delay literature (~10-14 yr, female-skewed)",
)

FABRY = DiseaseProfile(
    key="FABRY",
    name="Fabry disease",
    dx_code="E75.21",
    onset_age_mean=9.0,
    onset_age_sd=5.0,
    female_fraction=0.50,
    prevalence_per_100k=1.5,
    inheritance="X-linked",
    symptoms=(
        Symptom("G60.9", "acroparaesthesia / burning extremity pain", 0.85, 0.0, 0.25,
                "NEUROLOGY"),
        Symptom("L74.4", "anhidrosis / heat intolerance", 0.50, 0.50, 0.40, "DERMATOLOGY"),
        Symptom("Q82.8", "angiokeratoma", 0.40, 3.00, 0.75, "DERMATOLOGY"),
        Symptom("R80.9", "proteinuria", 0.65, 12.0, 0.20, "NEPHROLOGY", settings=("LAB",)),
        Symptom("I42.2", "left ventricular hypertrophy", 0.55, 15.0, 0.30, "CARDIOLOGY"),
        Symptom("H18.50", "corneal verticillata", 0.45, 4.00, 0.85, "OTHER"),
        Symptom("R10.9", "abdominal pain / diarrhoea", 0.55, 0.60, 0.02, "GASTROENTEROLOGY"),
    ),
    misdx=(
        Misdiagnosis("M79.7", "fibromyalgia", 0.26, 3.20, "RHEUMATOLOGY"),
        Misdiagnosis("M25.50", "growing pains / idiopathic limb pain", 0.42, 3.80,
                     "PRIMARY_CARE"),
        Misdiagnosis("M06.9", "juvenile idiopathic arthritis / rheumatic fever", 0.18, 2.10,
                     "RHEUMATOLOGY"),
        Misdiagnosis("K58.9", "irritable bowel syndrome", 0.28, 2.40, "GASTROENTEROLOGY"),
        Misdiagnosis("I42.9", "hypertrophic cardiomyopathy, idiopathic", 0.16, 1.60,
                     "CARDIOLOGY", procedures=("93306", "75561")),
    ),
    recognizing_specialties=("GENETICS", "NEPHROLOGY", "CARDIOLOGY"),
    recognition_prob=0.26,
    referral_mix={
        "PRIMARY_CARE": 0.16, "RHEUMATOLOGY": 0.20, "GASTROENTEROLOGY": 0.14,
        "NEUROLOGY": 0.14, "DERMATOLOGY": 0.10, "CARDIOLOGY": 0.10,
        "NEPHROLOGY": 0.08, "GENETICS": 0.03, "OTHER": 0.05,
    },
    confirm_codes=("82657", "81405"),  # alpha-galactosidase A assay, GLA sequencing
    care_seeking_median_years=0.90,
    care_seeking_logsd=0.85,
    visit_rate=4.8,
    death_hazard=0.010,
    target_median_delay_years=14.0,
    reference="Mehta et al., Fabry Outcome Survey (13-16 yr, longer in females)",
)


PANEL: tuple[DiseaseProfile, ...] = (ALS, GD1, LOPD, HAE, HEDS, FABRY)
PROFILES: dict[str, DiseaseProfile] = {p.key: p for p in PANEL}


def get(key: str) -> DiseaseProfile:
    try:
        return PROFILES[key]
    except KeyError:
        raise KeyError(f"unknown disease {key!r}; have {sorted(PROFILES)}") from None


# ---------------------------------------------------------------------------
# Background morbidity
# ---------------------------------------------------------------------------
# Undiagnosed rare-disease patients are not blank slates: they also get flu,
# hypertension and sprained ankles. Without this noise floor the phenotype
# algorithm looks far more precise than it is, because every claim in the
# record would be disease-related by construction.

BACKGROUND_CODES: tuple[tuple[str, str, float], ...] = (
    ("Z00.00", "routine adult health examination", 0.16),
    ("J06.9", "acute upper respiratory infection", 0.13),
    ("I10", "essential hypertension", 0.09),
    ("M54.5", "low back pain", 0.08),
    ("E78.5", "hyperlipidaemia", 0.07),
    ("R51.9", "headache", 0.06),
    ("K21.9", "gastro-oesophageal reflux", 0.06),
    ("F32.9", "depressive episode", 0.05),
    ("N39.0", "urinary tract infection", 0.05),
    ("S93.401A", "ankle sprain", 0.04),
    ("R05.9", "cough", 0.05),
    ("E11.9", "type 2 diabetes mellitus", 0.05),
    ("J45.909", "asthma", 0.04),
    ("R53.83", "fatigue", 0.04),
    ("M25.50", "joint pain", 0.03),
)

BACKGROUND_PROCEDURES: tuple[str, ...] = (
    "99213", "99214", "80053", "85025", "71046", "99385",
)

#: Rate of non-disease encounters per person-year, by broad age band.
BACKGROUND_VISIT_RATE: dict[str, float] = {
    "child": 2.4,
    "adult": 3.1,
    "older": 5.6,
}


def background_rate_for_age(age_years: float) -> float:
    if age_years < 18:
        return BACKGROUND_VISIT_RATE["child"]
    if age_years < 60:
        return BACKGROUND_VISIT_RATE["adult"]
    return BACKGROUND_VISIT_RATE["older"]
