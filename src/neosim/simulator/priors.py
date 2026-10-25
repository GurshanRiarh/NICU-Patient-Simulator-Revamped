"""Literature-derived clinical priors for the event and physiology layers.

The simulator distinguishes two kinds of numbers:

* **Absolute spontaneous event rates** live in :meth:`EventScheduler._rates` and
  are calibrated to the held-out PreMo cohort.  Event definitions and
  ascertainment differ substantially between neonatal studies, so a single
  literature rate cannot be transplanted into the simulator without changing
  the cohort-level burden.
* **Conditional probabilities, timing, severity distributions and intervention
  effect sizes** are taken from the neonatal literature summarised in
  ``docs/CLINICAL_EVIDENCE.md``.  Where the evidence is conflicting or only
  qualitative, the comment says so and the value is explicitly a modelling
  choice rather than a measured effect.

All functions are pure and unit-testable.  The citations use the first author,
year and PubMed identifier; full references are in the evidence document.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# ---------------------------------------------------------------------------
# Apnoea and cardiorespiratory coupling
# ---------------------------------------------------------------------------
# Apnoea duration mixture.  The clinical definition is a pause >15-20 s, but
# shorter 10-20 s events are common and have a much lower risk of bradycardia.
# Zhao 2011 (PMID 21301866) reports bradycardia in 10% of 10-14 s apnoeas, 34%
# of 15-20 s apnoeas and 75% of apnoeas >20 s.  The mixture below gives typical
# events of 10-30 s with a tail of prolonged events.
APNEA_DURATION_MIXTURE: tuple[tuple[float, float, float], ...] = (
    (0.70, 10.0, 30.0),
    (0.20, 30.0, 45.0),
    (0.10, 45.0, 90.0),
)

# Poets 1993 (PMID 8233715): median apnoea-to-desaturation delay 0.8 s (range
# -4.9 to +11.5 s) and apnoea-to-bradycardia delay 4.8 s (range -4.0 to +14.0 s).
# Zhao 2011: desaturation precedes bradycardia by a median of 4.2 s.
APNEA_DESAT_DELAY_MIN = (0.0, 0.20)
APNEA_BRADY_DELAY_MIN = (0.05, 0.25)

# Poets 1993: 86% of bradycardias were associated with a desaturation.  The
# reverse probability is much lower: Martini 2020 (PMID 32446489) found that
# isolated desaturations were the commonest event (625/1050) and isolated
# bradycardias the least common (171/1050), but the PreMo cohort's
# threshold-defined burden implies that only about 5% of desaturations are
# followed by a bradycardia.  The value below is therefore calibrated to PreMo
# rather than taken directly from the 2020 combined-event fraction.
BRADY_DESAT_ASSOC = 0.86
DESAT_BRADY_BASE = 0.03
DESAT_BRADY_SEVERITY = 0.02
DESAT_BRADY_MAX = 0.05


def sample_apnea_duration(rng: np.random.Generator) -> float:
    """Return an apnoea duration in seconds from the literature mixture."""
    draw = float(rng.random())
    cumulative = 0.0
    for weight, low, high in APNEA_DURATION_MIXTURE:
        cumulative += weight
        if draw <= cumulative:
            return float(rng.uniform(low, high))
    return float(rng.uniform(10.0, 30.0))


def bradycardia_given_apnea(duration_s: float) -> float:
    """Probability of bradycardia for an apnoea of ``duration_s`` seconds.

    Zhao 2011 gives bin values of 10% (10-14 s), 34% (15-20 s) and 75% (>20 s).
    Linear interpolation avoids a discontinuous jump at 20 s while preserving
    the reported bin estimates.
    """
    if duration_s <= 14.0:
        return 0.10
    if duration_s <= 20.0:
        return 0.10 + (duration_s - 14.0) / 6.0 * 0.24
    return min(0.75, 0.34 + (duration_s - 20.0) / 10.0 * 0.41)


def desaturation_given_apnea(duration_s: float, severity: float) -> float:
    """Probability of desaturation for an apnoea.

    Poets 1993 and the 2001 cohort (PMID 11731652) show that desaturations and
    bradycardias are almost always preceded by a respiratory pause.  The exact
    apnoea-to-desaturation probability is not reported as a single estimate;
    this function is a conservative modelling choice that rises with duration
    and severity and is capped below certainty.
    """
    duration_term = min(max((duration_s - 10.0) / 30.0, 0.0), 1.0)
    return min(0.95, 0.55 + 0.25 * severity + 0.15 * duration_term)


def desaturation_bradycardia_probability(severity: float) -> float:
    """Probability that a desaturation also produces a bradycardia.

    Most desaturations are not followed by bradycardia.  The 2020 cohort found
    that isolated desaturations were the commonest event, but its combined-event
    fraction is higher than the PreMo threshold-defined burden; this function is
    calibrated to PreMo so the simulated bradycardia burden matches the cohort.
    """
    return min(DESAT_BRADY_MAX, DESAT_BRADY_BASE + DESAT_BRADY_SEVERITY * severity)


# ---------------------------------------------------------------------------
# Feeding
# ---------------------------------------------------------------------------
# Thoyre 2003 (PMID 12706309): 22 VLBW infants averaged 10.8 desaturations
# (<90%) per bottle feed and spent 20% of feeding time below 90%; 59% of
# events were mild (85-89%), 20% moderate (81-84%) and 21% severe (<=80%).
# Poets 1997 (PMID 9174231): bottle feeding produced three times more
# desaturations than bolus gavage; the excess was greatest during the feeding
# hour and persisted for the following 2 h.  Slocum 2009 (PMID 19148108) found
# no increase in pre- versus post-feed event rates, so the simulator models
# events during and immediately after a feed rather than a blanket
# postprandial increase.
FEEDING_DESAT_BASE = 0.40
FEEDING_DESAT_IMMATURITY = 0.20
FEEDING_DESAT_SEVERITY = 0.10
FEEDING_DESAT_MAX = 0.75
FEEDING_DESAT_DURING_PROB = 0.70
FEEDING_DESAT_DURING_WINDOW = (0.0, 25.0)
FEEDING_DESAT_LATE_WINDOW = (25.0, 120.0)

# Thoyre 2003 severity split: (probability, low, high) on the simulator's
# 0-1 severity scale.  The scale maps to a target SpO2 of 93 - 25 * severity.
FEEDING_DESAT_SEVERITY_MIXTURE: tuple[tuple[float, float, float], ...] = (
    (0.59, 0.16, 0.32),
    (0.20, 0.36, 0.48),
    (0.21, 0.52, 0.75),
)

# Feeding-associated bradycardia is clinically recognised but less well
# quantified than desaturation.  Lee 2023 (PMID 36521453) recorded a median of
# 0.3 feeding-associated bradycardias/day in infants selected for feeding
# instability, while Poets 1997 found no significant effect of feeding
# technique on bradycardia.  The simulator therefore uses a low probability.
FEEDING_BRADY_BASE = 0.03
FEEDING_BRADY_IMMATURITY = 0.06
FEEDING_BRADY_SEVERITY = 0.04
FEEDING_BRADY_MAX = 0.15


def feeding_desaturation_probability(immaturity: float, severity: float) -> float:
    value = FEEDING_DESAT_BASE + FEEDING_DESAT_IMMATURITY * immaturity
    value += FEEDING_DESAT_SEVERITY * severity
    return min(FEEDING_DESAT_MAX, max(0.0, value))


def feeding_bradycardia_probability(immaturity: float, severity: float) -> float:
    value = FEEDING_BRADY_BASE + FEEDING_BRADY_IMMATURITY * immaturity
    value += FEEDING_BRADY_SEVERITY * severity
    return min(FEEDING_BRADY_MAX, max(0.0, value))


def feeding_desaturation_severity(rng: np.random.Generator) -> float:
    """Draw a severity matching the Thoyre 2003 mild/moderate/severe split."""
    draw = float(rng.random())
    cumulative = 0.0
    for weight, low, high in FEEDING_DESAT_SEVERITY_MIXTURE:
        cumulative += weight
        if draw <= cumulative:
            return float(rng.uniform(low, high))
    return float(rng.uniform(0.16, 0.32))


# ---------------------------------------------------------------------------
# Handling and suction
# ---------------------------------------------------------------------------
# Brandon 2022 (PMID 34896732): bundled care events lasted about 28 min and
# included nine activities; heart rate rose during more than 74% of diapering
# observations, while the study did not find group differences in SpO2.
# Bradycardia/desaturation are therefore uncommon during routine handling.
HANDLING_BRADY_BASE = 0.03
HANDLING_BRADY_IMMATURITY = 0.06
HANDLING_BRADY_SEVERITY = 0.05
HANDLING_BRADY_MAX = 0.15
HANDLING_DESAT_BASE = 0.02
HANDLING_DESAT_IMMATURITY = 0.05
HANDLING_DESAT_SEVERITY = 0.04
HANDLING_DESAT_MAX = 0.12

# Shah 1992 (PMID 1578315): SpO2 fell from 94% to 84% during endotracheal
# suctioning in 12 preterm infants; preoxygenation prevented the fall.  A 2025
# neonatal observational study (PMID 41267597) found desaturation to be the
# commonest disturbance during 270 suction episodes.  The incidence estimates
# from adult ICU studies (6.5% desaturation, 1.1% heart-rate change; Maggiore
# 2013, PMID 23466423) are used only as a lower bound because neonatal
# ascertainment and preoxygenation practice differ.
SUCTION_DESAT_BASE = 0.20
SUCTION_DESAT_IMMATURITY = 0.10
SUCTION_DESAT_SEVERITY = 0.10
SUCTION_DESAT_MAX = 0.40
SUCTION_BRADY_BASE = 0.08
SUCTION_BRADY_IMMATURITY = 0.04
SUCTION_BRADY_SEVERITY = 0.04
SUCTION_BRADY_MAX = 0.20
PREOXYGENATION_PROB = 0.60
PREOXYGENATION_RISK_SCALE = 0.30
PREOXYGENATION_EFFECT_SCALE = 0.25


def handling_bradycardia_probability(immaturity: float, severity: float) -> float:
    value = HANDLING_BRADY_BASE + HANDLING_BRADY_IMMATURITY * immaturity
    value += HANDLING_BRADY_SEVERITY * severity
    return min(HANDLING_BRADY_MAX, max(0.0, value))


def handling_desaturation_probability(immaturity: float, severity: float) -> float:
    value = HANDLING_DESAT_BASE + HANDLING_DESAT_IMMATURITY * immaturity
    value += HANDLING_DESAT_SEVERITY * severity
    return min(HANDLING_DESAT_MAX, max(0.0, value))


def suction_bradycardia_probability(immaturity: float, severity: float) -> float:
    value = SUCTION_BRADY_BASE + SUCTION_BRADY_IMMATURITY * immaturity
    value += SUCTION_BRADY_SEVERITY * severity
    return min(SUCTION_BRADY_MAX, max(0.0, value))


def suction_desaturation_probability(immaturity: float, severity: float) -> float:
    value = SUCTION_DESAT_BASE + SUCTION_DESAT_IMMATURITY * immaturity
    value += SUCTION_DESAT_SEVERITY * severity
    return min(SUCTION_DESAT_MAX, max(0.0, value))


def suction_risk_scale(preoxygenated: bool) -> float:
    return PREOXYGENATION_RISK_SCALE if preoxygenated else 1.0


def suction_effect_scale(preoxygenated: bool) -> float:
    return PREOXYGENATION_EFFECT_SCALE if preoxygenated else 1.0


# ---------------------------------------------------------------------------
# Medications
# ---------------------------------------------------------------------------
# Caffeine: the FDA label for caffeine citrate reports an apnoea rate of 4.9 vs
# 7.2 per 24 h on day 2 (about a 32% reduction), a >=50% reduction in 76% vs
# 57% of infants, and zero apnoeas in 26.7% vs 8.1%.  The EMA reports a >=50%
# reduction on 6/10 days and >=8 apnoea-free days in 22% vs 0%.  The neonatal
# half-life is 3-4 days (FDA label) / about 100 h (Zhao 2011), so the effect is
# modelled as long-lived within a shift.
CAFFEINE_APNEA_SEVERITY_MULT = 0.55
CAFFEINE_APNEA_DURATION_MULT = 0.85
CAFFEINE_RESP_EVENT_SEVERITY_MULT = 0.85
CAFFEINE_TAU_DECAY_MIN = 480.0

# Morphine: NEOPAIN (PMID 15158628) found no reduction in severe IVH/PVL/death
# and significant adverse effects; the PK/PD analysis (PMID 18723857) found no
# relationship between morphine concentration and the heart-rate response to
# suctioning.  The main modelled effect is therefore increased respiratory
# event severity, not a precise direct heart-rate effect.
MORPHINE_EVENT_SEVERITY_MULT = 1.25
MORPHINE_TAU_DECAY_MIN = 240.0

# Surfactant: the AAP policy statement (PMID 24379227) documents reduced
# mortality and respiratory morbidity; Dani 2026 (PMID 42158663) shows a
# significant improvement in SpO2/FiO2 after surfactant.  The direct SpO2
# kernel is a modelling choice because studies report oxygenation ratios rather
# than an isolated SpO2 delta.
SURFACTANT_DESAT_SEVERITY_MULT = 0.70
SURFACTANT_DESAT_DURATION_MULT = 0.85
SURFACTANT_TAU_DECAY_MIN = 360.0

# ---------------------------------------------------------------------------
# Direct intervention kernels
# ---------------------------------------------------------------------------
# These are full-severity amplitudes; the physiology layer multiplies by the
# scheduled event severity.  The neural generator supplies the baseline
# trajectory, so the kernels only need to represent the incremental effect.
# Where a study reports a clear direction but not an exact vital-sign delta,
# the amplitude is a documented modelling choice.
FEEDING_HR = 5.0
FEEDING_RR = -2.0
FEEDING_SPO2 = -2.0
FEEDING_TEMP = 0.08

# Brandon 2022: heart rate rose during most diapering observations; SpO2 and
# respiratory-rate changes were not consistent.  The small RR/SpO2 terms are
# modelling choices that keep handling from looking inert.
HANDLING_HR_BASE = 12.0
HANDLING_HR_SEVERITY = 12.0
HANDLING_RR_BASE = 1.0
HANDLING_RR_SEVERITY = 2.0
HANDLING_SPO2_BASE = -0.5
HANDLING_SPO2_SEVERITY = -1.5
HANDLING_TEMP = -0.05

# Shah 1992: SpO2 fell from 94% to 84% during suctioning without
# preoxygenation; preoxygenation prevented the fall.
SUCTION_HR = -18.0
SUCTION_RR = -4.0
SUCTION_SPO2 = -8.0

# Oxygen is a modelling choice: raising FiO2/flow improves SpO2 and modestly
# reduces respiratory drive.  Closed-loop oxygen is given as a longer, larger
# rescue than a routine FiO2 increase.
OXYGEN_SCHEDULED_HR = -1.5
OXYGEN_SCHEDULED_RR = -1.0
OXYGEN_SCHEDULED_SPO2 = 4.0
OXYGEN_CLOSED_HR = -2.0
OXYGEN_CLOSED_RR = -1.0
OXYGEN_CLOSED_SPO2 = 6.0

# Caffeine: FDA label/EMA reduce apnoea burden; the small direct HR/RR rise is
# a modelling choice consistent with methylxanthine physiology.
CAFFEINE_HR = 3.0
CAFFEINE_RR = 4.0
CAFFEINE_SPO2 = 0.5

# Morphine: NEOPAIN (PMID 15158628) and the PK/PD analysis (PMID 18723857)
# justify increased respiratory-event severity and respiratory depression; the
# direct vital-sign deltas are modelling choices.
MORPHINE_HR = -3.5
MORPHINE_RR = -6.0
MORPHINE_SPO2 = -1.5
MORPHINE_TEMP = -0.08

# Surfactant: AAP 2014 and Dani 2026 support improved oxygenation; the direct
# SpO2 amplitude is a modelling choice because the studies report oxygenation
# ratios rather than an isolated SpO2 delta.
SURFACTANT_HR = -2.0
SURFACTANT_RR = -5.0
SURFACTANT_SPO2 = 5.0

# Tactile stimulation arouses the infant and interrupts apnoea (Zhao 2011).
CLOSED_LOOP_TACTILE_HR = 8.0
CLOSED_LOOP_TACTILE_RR = 5.0
CLOSED_LOOP_TACTILE_SPO2 = 0.5

# ---------------------------------------------------------------------------
# Other care effects
# ---------------------------------------------------------------------------
# Kangaroo care: a 2022 meta-analysis (PMID 35010848) found a mean respiratory
# rate reduction of 3.50 breaths/min (95% CI -5.17 to -1.83); a 2023
# meta-analysis (PMID 37454580) found lower heart rate and higher SpO2, while a
# 2013 RCT (PMID 24246597) found fewer bradycardia/desaturation events during
# holding.  A 2004 study reported more events during skin-to-skin care, so the
# evidence is mixed.
KMC_HR = -3.0
KMC_RR = -3.5
KMC_SPO2 = 1.0
KMC_TEMP = 0.05

# Non-nutritive sucking: a 2022 meta-analysis (PMID 35410609) found improved
# pain scores and higher SpO2 but no significant heart-rate reduction; a 2018
# RCT (PMID 29684832) found lower heart rate and higher SpO2 when NNS was
# combined with sucrose.  The kernel is intentionally modest.
PAIN_HR = -4.0
PAIN_RR = -1.0
PAIN_SPO2 = 0.8

# Prone/head-elevated positioning reduces apnoea and desaturation (Zhao 2011;
# head-elevated tilt reduced desaturation episodes by 48.5%).  The simulator
# models a transient position change rather than a sustained posture, so the
# direct effect is small.
POSITION_HR = 1.0
POSITION_RR = -1.0
POSITION_SPO2 = 1.0
