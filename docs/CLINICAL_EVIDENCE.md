# Clinical evidence behind the simulator priors

This document records the evidence used by `src/neosim/simulator/priors.py` and
the event/physiology layers. It is deliberately separate from the code so that
clinical readers can inspect the sources, the mapping decisions, and the places
where the simulator is making a modelling choice rather than reproducing a
measured effect.

## Design principle: two kinds of numbers

The simulator uses two tiers of numbers:

1. **Absolute spontaneous event rates** are calibrated to the held-out PreMo
   cohort. Event definitions and ascertainment differ substantially between
   neonatal studies, so a published rate cannot be transplanted directly into a
   synthetic cohort without changing the cohort-level burden.
2. **Conditional probabilities, timing, severity distributions and intervention
   effects** are taken from the literature where available. Where a study
   reports a direction or a ratio but not an exact vital-sign delta, the code
   uses a documented modelling choice.

The held-out PreMo event burden used for calibration is `HR < 100` for
bradycardia and `SpO2 < 85` for desaturation, expressed as minutes per hour.
The current synthetic burden is approximately `0.085 vs 0.099` bradycardia
minutes/hour and `2.19 vs 2.07` desaturation minutes/hour (100 two-hour shifts,
seed 2026). These are cohort-level calibration targets, not claims that every
infant has the same event rate.

## Definitions differ across studies

| Source | Bradycardia definition | Desaturation definition | Apnoea definition |
| --- | --- | --- | --- |
| Poets 1993 (PMID 8233715) | fall of at least 33% from baseline for at least 4 s | SaO2 <= 80% | apnoeic pause >= 4 s |
| Zhao 2011 review (PMID 21301866) | heart rate < 2/3 of baseline for at least 4 s | SpO2 <= 80% for at least 4 s | pause > 15-20 s, or shorter if accompanied by bradycardia/desaturation |
| PreMo evaluation | HR < 100 bpm | SpO2 < 85% | RR < 5 breaths/min in the simulator |
| Lee 2023 (PMID 36521453) | HR < 100 bpm | SpO2 < 85% | not reported |

The 1993 study's "apnoeic pause" threshold of 4 s is much shorter than the
clinical apnoea definition. Its finding that 83% of bradycardias were
associated with an apnoeic pause therefore does **not** justify adding a
clinical apnoea event to 83% of bradycardias. The simulator instead models
duration-dependent apnoea-to-bradycardia coupling and the separate
bradycardia-to-desaturation association.

## Apnoea and cardiorespiratory coupling

| Finding | Source | Use in the simulator |
| --- | --- | --- |
| 193 bradycardias in 46/80 preterm infants (58%), median 3, range 1-18; 83% associated with apnoeic pauses, 86% with desaturations; 86% of bradycardias began after the desaturation fall | Poets 1993 (PMID 8233715) | `BRADY_DESAT_ASSOC = 0.86`; linked desaturation starts before the bradycardia |
| Apnoea-to-desaturation delay median 0.8 s (range -4.9 to +11.5 s); apnoea-to-bradycardia delay median 4.8 s (range -4.0 to +14.0 s) | Poets 1993 (PMID 8233715) | `APNEA_DESAT_DELAY_MIN = (0.0, 0.20)` min; `APNEA_BRADY_DELAY_MIN = (0.05, 0.25)` min |
| Desaturation precedes bradycardia by a median of 4.2 s | Zhao 2011 (PMID 21301866) | Same ordering of linked events |
| Bradycardia in 10% of 10-14 s apnoeas, 34% of 15-20 s apnoeas and 75% of apnoeas > 20 s | Zhao 2011 (PMID 21301866) | `bradycardia_given_apnea()` interpolates these bins |
| Referred infants had more desaturations (`20 +/- 6` vs `6 +/- 3` per 12 h) and bradycardias (`4.3 +/- 0.8` vs `1.1 +/- 0.3` per 12 h); events were always preceded by a respiratory pause | Di Fiore 2001 (PMID 11731652) | Supports apnoea as the upstream event; absolute rates are calibrated to PreMo instead |
| 1,050 cardiorespiratory events: 625 isolated desaturations, 171 isolated bradycardias, about 254 combined events | Martini 2020 (PMID 32446489) | Combined events are common, but the PreMo threshold-defined burden implies a lower desaturation-to-bradycardia probability than the raw 2020 fraction |
| Central apnoea is common in preterm infants and is associated with immaturity | Fairchild 2016 (PMID 26959485) | Background for the apnoea event and duration mixture |

The apnoea duration mixture is `70% 10-30 s`, `20% 30-45 s` and `10% 45-90 s`.
The short end is below the clinical 15-20 s definition because shorter events
are common and have a much lower bradycardia risk; the mixture is intended to
represent the range of events that produce monitor-level desaturations and
bradycardias, not only formally diagnosed apnoea.

The desaturation-to-bradycardia probability is deliberately low
(`0.015 + 0.01 * severity`, capped at `0.025`). The 2020 cohort's combined-event
fraction is higher, but applying it directly produced too many bradycardia
minutes against PreMo. This is a calibration decision, not a claim that the
2020 cohort is wrong: the cohorts, event definitions and ascertainment differ.

## Feeding

| Finding | Source | Use in the simulator |
| --- | --- | --- |
| 22 VLBW infants averaged 10.8 desaturations (< 90%) per bottle feed and spent 20% of feeding time below 90%; 59% mild (85-89%), 20% moderate (81-84%), 21% severe (<= 80%) | Thoyre 2003 (PMID 12706309) | Feeding desaturation probability and severity mixture |
| Bottle feeding produced about three times more desaturations than bolus gavage; the excess was greatest during the feeding hour and persisted for 2 h; no significant effect on apnoea or bradycardia | Poets 1997 (PMID 9174231) | Bottle route risk multiplier `1.0`; gavage `0.35`; feeding bradycardia is not route-scaled |
| No increase in apnoea, bradycardia or desaturation before versus after feeding | Slocum 2009 (PMID 19148108) | Conflicts with a blanket postprandial effect; the simulator places most feeding events during the feed and early post-feed window rather than across the whole inter-feed interval |
| Thickened formula reduced feeding-associated desaturations from 2.3 to 0.3/day and bradycardias from 0.3 to 0/day in a selected feeding-instability cohort | Lee 2023 (PMID 36521453) | Supports the existence of feeding-associated bradycardia; the simulator keeps the probability low because the cohort was selected and the intervention changed the feed |
| Feeding-associated desaturation was modelled as 40-75% per feed, with 70% of events during 0-25 min and 30% at 25-120 min | Modelling choice informed by Thoyre 2003 and Poets 1997 | `FEEDING_DESAT_*` constants |

The severity mixture maps to the simulator's `0-1` severity scale, which in
turn maps to an enforced nadir of `93 - 25 * severity`:

| Reported band | SpO2 range | Severity range | Mixture weight |
| --- | --- | --- | --- |
| Mild | 85-89% | 0.16-0.32 | 59% |
| Moderate | 81-84% | 0.36-0.48 | 20% |
| Severe | <= 80% | 0.52-0.75 | 21% |

## Handling, diapering and suction

| Finding | Source | Use in the simulator |
| --- | --- | --- |
| 46 preterm infants, 605 bundled care events; events averaged 28 min and contained nine activities; HR rose during about half of observations and in > 74% of diapering observations; few RR/SpO2 changes | Brandon 2022 (PMID 34896732) | Diaper changes are clustered 5-20 min after feeds; handling has a modest HR rise and small RR/SpO2 effects; bradycardia/desaturation probabilities are low |
| SpO2 fell from 94% to 84% during endotracheal suctioning in 12 preterm infants; preoxygenation prevented the fall | Shah 1992 (PMID 1578315) | Suction without preoxygenation has a larger direct SpO2 effect and higher event risk; preoxygenation scales both down |
| 270 suction episodes in 62 neonates (30-40 weeks); desaturation was the most frequent disturbance, followed by tachycardia, tachypnea, bradypnea and bradycardia | Harshitha 2025 (PMID 41267597) | Desaturation is the primary suction-associated event; bradycardia is less common |
| Less invasive suction protocol in 100 preterm infants (32-36 weeks) produced lower HR and higher SpO2 during suctioning | Temizsoy 2026 (PMID 42363180) | Supports a modifiable suction effect; the simulator's preoxygenation flag is a simplification of the full protocol |
| Adult ICU cohort: 4,506 suctionings; desaturation in 46.8% of subjects but 6.5% of suctionings; HR change in 10.1% of subjects but 1.1% of suctionings; all complications in 12.4% of procedures | Maggiore 2013 (PMID 23466423) | Used only as a lower-bound/context source because neonatal ascertainment and preoxygenation differ |

The handling probabilities are intentionally lower than the old heuristic
values: `3-15%` for bradycardia and `2-12%` for desaturation, rising with
immaturity and severity. The direct handling kernel raises HR by approximately
`12 + 12 * severity` bpm at full effect, with small RR and SpO2 changes.

Suction event probabilities are `8-20%` for bradycardia and `20-40%` for
desaturation, scaled to `30%` when preoxygenated. The direct suction kernel is
`-18 bpm`, `-4 breaths/min` and `-8% SpO2` at full severity without
preoxygenation, and `25%` of that with preoxygenation.

## Medications

### Caffeine

- The FDA label for caffeine citrate reports an apnoea rate of `4.9 vs 7.2` per
  24 h on day 2 (about a 32% reduction), a >= 50% reduction in `76% vs 57%` of
  infants, and zero apnoeas in `26.7% vs 8.1%`.
- The EMA label (Peyona) reports a >= 50% reduction on 6/10 days and >= 8
  apnoea-free days in `22% vs 0%`.
- The neonatal half-life is 3-4 days in the FDA label and about 100 h in Zhao
  2011 (PMID 21301866).
- The simulator uses `0.55` for apnoea severity, `0.85` for apnoea duration,
  `0.85` for later bradycardia/desaturation severity, and an `8 h` decay
  constant (`480 min`). The decay constant is shorter than the true half-life
  because the simulator models a single shift; the direction and relative size
  of the effect are the evidence-based parts.

### Morphine

- NEOPAIN (PMID 15158628) randomised 898 ventilated preterm infants in 16
  centres. Morphine did not reduce the composite outcome (26% vs 27%), death
  (11% vs 13%), severe IVH (11% vs 13%) or PVL (9% vs 7%), and adverse effects
  were more common. In the non-open-label subgroup the composite outcome was
  24% vs 15% (`p = 0.0338`), but the trial was not designed for that subgroup.
- The secondary PK/PD analysis (PMID 18723857) found no relationship between
  morphine concentration and the heart-rate response to suctioning.
- Soffer 2019 (PMID 31455824) is a commentary on safety, efficacy and
  equipoise, not a new trial. It should not be cited as evidence that a trial
  was stopped early.
- The simulator therefore models morphine mainly as an increase in respiratory
  event severity (`1.25`) and respiratory depression, not as a precise
  analgesic HR effect. The direct vital-sign deltas are modelling choices.

### Surfactant

- The AAP policy statement (PMID 24379227) supports surfactant replacement for
  preterm infants with RDS, with reduced mortality and respiratory morbidity.
- Dani 2026 (PMID 42158663) reports improved SpO2/FiO2 and a/APO2 after first
  and second surfactant doses in 140 very preterm infants (23+0 to 29+6 weeks).
- The simulator uses `0.70` for desaturation severity, `0.85` for duration and
  a `6 h` decay constant (`360 min`). The direct `+5% SpO2` kernel is a
  modelling choice because the studies report oxygenation ratios rather than
  an isolated SpO2 delta.

## Kangaroo care, pain and positioning

- A 2022 meta-analysis of RCTs (PMID 35010848) found a mean respiratory-rate
  reduction of `3.50 breaths/min` (95% CI `-5.17 to -1.83`) during kangaroo
  mother care.
- A 2023 meta-analysis (PMID 37454580) found lower heart rate and higher SpO2
  with kangaroo mother care or skin-to-skin contact.
- A 2013 RCT (PMID 24246597) found fewer bradycardia/desaturation events during
  holding, while a 2004 study reported more events during skin-to-skin care.
  The evidence is mixed; the simulator uses a modest beneficial kernel and
  varies the family-visit severity.
- Non-nutritive sucking: a 2022 meta-analysis in term infants (PMID 35410609)
  found improved pain scores and higher SpO2 but no significant heart-rate
  reduction; a 2018 RCT in preterm infants (PMID 29684832) found lower heart
  rate and higher SpO2 when NNS was combined with sucrose. The simulator's pain
  kernel is intentionally modest because the evidence is heterogeneous.
- Prone/head-elevated positioning is supported by Zhao 2011 (PMID 21301866),
  which reports a 48.5% reduction in desaturation episodes with head-elevated
  tilt. The simulator models a transient position change, so the direct effect
  is small.

## What is calibrated versus what is literature-derived

| Component | Source of the number | Notes |
| --- | --- | --- |
| Spontaneous bradycardia rate | PreMo held-out burden | `brady = base * 0.09`; condition-specific multiplier for `bradycardia` |
| Spontaneous desaturation rate and duration | PreMo held-out burden plus a 1-4 min episode assumption | Duration is not well standardised |
| Apnoea duration mixture | Modelling choice informed by Zhao 2011 | 10-90 s; short events included |
| Apnoea-to-bradycardia probability | Zhao 2011 bins | 10%, 34%, 75% at 10-14 s, 15-20 s, > 20 s |
| Apnoea-to-desaturation probability | Modelling choice from Poets 1993/Di Fiore 2001 | 55-95%, rising with duration and severity |
| Bradycardia-to-desaturation association | Poets 1993 | 86% |
| Desaturation-to-bradycardia probability | Calibrated to PreMo | 1.5-2.5%; lower than the raw 2020 combined fraction |
| Feeding desaturation probability and severity | Thoyre 2003; Poets 1997; calibrated | 40-75% per feed; 59/20/21 severity split |
| Handling/suction probabilities | Brandon 2022; Shah 1992; calibrated | Low for handling; higher for suction without preoxygenation |
| Caffeine/surfactant/morphine event multipliers | FDA/EMA; NEOPAIN; AAP/Dani | Directions are evidence-based; exact multipliers are modelling choices |
| Direct intervention kernels | Literature direction plus modelling choices | Full-severity amplitudes in `priors.py` |

## How the code uses these numbers

- `src/neosim/simulator/priors.py` contains the pure probability/timing
  functions and the full-severity intervention amplitudes.
- `EventScheduler._spontaneous()` draws spontaneous event times, durations and
  severities. Absolute bradycardia and desaturation rates are calibrated to
  PreMo; apnoea durations come from the literature mixture.
- `EventScheduler._couple_spontaneous()` adds linked desaturation and
  bradycardia events and records `associated_desat` / `associated_brady` in the
  event metadata. It does not add a clinical apnoea event to most
  bradycardias, because the 1993 "apnoeic pause" threshold was only 4 s.
- `EventScheduler._couple_care()` adds feeding, handling and suction events.
  Feeding route modifies desaturation risk; suction preoxygenation modifies
  both risk and direct effect.
- `EventScheduler._medication_interactions()` applies caffeine, morphine and
  surfactant multipliers to later events.
- `apply_care_effects()` in `physiology.py` applies the direct intervention
  kernels.
- `enforce_events()` guarantees that scheduled events are visible on the
  one-minute monitor grid. Apnoea enforcement changes only RR; bradycardia
  enforcement changes SpO2 only when the event is marked `associated_desat`.

## Conflicting and weak evidence

- **Postprandial events:** Poets 1997 found a persistent post-feed
  desaturation excess, whereas Slocum 2009 found no pre/post difference. The
  simulator places most feeding events during the feed and early post-feed
  window, and does not impose a blanket increase across the full inter-feed
  interval.
- **Kangaroo care:** meta-analyses suggest physiological benefit, but at least
  one study reported more events during skin-to-skin care. The simulator uses a
  modest average effect and varies visit severity.
- **Morphine:** NEOPAIN did not show the hoped-for benefit and found adverse
  effects; the PK/PD analysis did not find a clear HR relationship. The
  simulator does not claim an analgesic HR effect.
- **Non-nutritive sucking:** the 2022 meta-analysis is in term infants, while
  the 2018 RCT is in preterm infants. The evidence is not directly
  interchangeable.
- **Suction:** the strongest neonatal source (Shah 1992) is small (12 infants),
  and the 2025 study is observational. The adult ICU data are used only as a
  lower bound.
- **Event definitions:** bradycardia and desaturation thresholds differ across
  studies and across the PreMo annotation. The simulator uses `HR < 100` and
  `SpO2 < 85` for evaluation, and event severity is a separate continuous
  quantity.
- **Temporal resolution:** the simulator generates one sample per minute. A
  four-second bradycardia cannot be represented exactly; sub-minute events are
  enforced at the nearest monitor sample and documented as a resolution
  limitation.

## Evidence gaps

- There are few randomised trials of routine NICU care effects on continuous
  vital signs; most care priors are observational or physiological reasoning.
- Feeding-associated bradycardia is less well quantified than feeding-associated
  desaturation.
- The exact apnoea-to-desaturation probability is not reported as a single
  estimate; the simulator uses a bounded modelling choice.
- Dose-response relationships for caffeine, morphine and surfactant are not
  modelled; the simulator uses dose-independent multipliers.
- The interaction between events (for example, whether a desaturation that
  follows a bradycardia should be labelled separately) is simplified to avoid
  double-counting.
- The PreMo cohort is used for calibration, but it is not a NICU intervention
  trial; the absolute event burden may not generalise to other units.

## References

1. Poets CF, Stebbens VA, Samuels MP, Southall DP. The relationship between
   bradycardia, apnea, and hypoxemia in preterm infants. *Pediatric Research*.
   1993;34(2):144-147. doi:10.1203/00006450-199308000-00007. PMID 8233715.
2. Di Fiore JM, Arko MK, Miller MJ, Krauss A, Betkerur A, Zadell A, Kenney SR,
   Martin RJ. Cardiorespiratory events in preterm infants referred for apnea
   monitoring studies. *Pediatrics*. 2001;108(6):1304-1308.
   doi:10.1542/peds.108.6.1304. PMID 11731652.
3. Martini S, Frabboni G, Rucci P, Czosnyka M, Smielewski P, Galletti S, Vitali
   F, Faldella G, Austin T, Corvaglia L. Cardiorespiratory events in infants
   born preterm during the transitional period. *The Journal of Pediatrics*.
   2020;221:32-38.e2. doi:10.1016/j.jpeds.2020.02.037. PMID 32446489.
4. Zhao J, Gonzalez F, Mu D. Apnea of prematurity: from cause to treatment.
   *European Journal of Pediatrics*. 2011;170(9):1097-1105.
   doi:10.1007/s00431-011-1409-6. PMID 21301866.
5. Fairchild K, Mohr M, Paget-Brown A, Tabacaru C, Lake D, Delos J, Moorman JR,
   Kattwinkel J. Clinical associations of immature breathing in preterm
   infants: part 1-central apnea. *Pediatric Research*. 2016;80(1):21-27.
   doi:10.1038/pr.2016.43. PMID 26959485.
6. Thoyre SM, Carlson J. Occurrence of oxygen desaturation events during
   preterm infant bottle feeding near discharge. *Early Human Development*.
   2003;72(1):25-36. doi:10.1016/s0378-3782(03)00008-2. PMID 12706309.
7. Poets CF, Langner MU, Bohnhorst B. Effects of bottle feeding and two
   different methods of gavage feeding on oxygenation and breathing patterns in
   preterm infants. *Acta Paediatrica*. 1997;86(4):419-423.
   doi:10.1111/j.1651-2227.1997.tb09034.x. PMID 9174231.
8. Slocum C, Arko M, Di Fiore J, Martin RJ, Hibbs AM. Apnea, bradycardia and
   desaturation in preterm infants before and after feeding. *Journal of
   Perinatology*. 2009;29(3):209-212. doi:10.1038/jp.2008.226. PMID 19148108.
9. Lee G, Lee J, Jeon GW, Jun YH. A thickened formula reduces
   feeding-associated oxygen desaturation and bradycardia in preterm infants.
   *Clinical and Experimental Pediatrics*. 2023;66(1):32-37.
   doi:10.3345/cep.2022.00780. PMID 36521453.
10. Brandon DH, Hatch D, Barnes A, Vance AJ, Harney J, Voigtman B, Younge N.
    Impact of diaper change frequency on preterm infants' vital sign stability
    and skin health: A RCT. *Early Human Development*. 2022;164:105510.
    doi:10.1016/j.earlhumdev.2021.105510. PMID 34896732.
11. Shah AR, Kurth CD, Gwiazdowski SG, Chance B, Delivoria-Papadopoulos M.
    Fluctuations in cerebral oxygenation and blood volume during endotracheal
    suctioning in premature infants. *The Journal of Pediatrics*.
    1992;120(5):769-774. doi:10.1016/s0022-3476(05)80246-x. PMID 1578315.
12. Maggiore SM, Lellouche F, Pignataro C, Girou E, Maitre B, Richard JC,
    Lemaire F, Brun-Buisson C, Brochard L. Decreasing the adverse effects of
    endotracheal suctioning during mechanical ventilation by changing practice.
    *Respiratory Care*. 2013;58(10):1588-1597. doi:10.4187/respcare.02265.
    PMID 23466423.
13. Harshitha UK, Leslie SL, Jayashree P, Samruddha SP, Reema DS, Nayanatara AK.
    Physiological effects of endotracheal suctioning in neonates: a prospective
    observational study in the neonatal intensive care unit. *La Clinica
    Terapeutica*. 2025;176(6):785-788. doi:10.7417/ct.2025.5298. PMID 41267597.
14. Temizsoy E, Topcuoglu S, Aliustaoglu H, Ozkan E, Karatekin G. Effects of a
    less invasive suction protocol on pain and physiological stability in
    preterm infants receiving non-invasive ventilation: a randomized controlled
    trial. *BMC Pediatrics*. 2026;26(1):867.
    doi:10.1186/s12887-026-07212-8. PMID 42363180.
15. Anand KJ, Hall RW, Desai N, Shephard B, Bergqvist LL, Young TE, Boyle EM,
    Carbajal R, Bhutani VK, Moore MB, Kronsberg SS, Barton BA; NEOPAIN Trial
    Investigators Group. Effects of morphine analgesia in ventilated preterm
    neonates: primary outcomes from the NEOPAIN randomised trial. *Lancet*.
    2004;363(9422):1673-1682. doi:10.1016/s0140-6736(04)16251-x. PMID 15158628.
16. Anand KJ, Anderson BJ, Holford NH, Hall RW, Young T, Shephard B, Desai NS,
    Barton BA; NEOPAIN Trial Investigators Group. Morphine pharmacokinetics and
    pharmacodynamics in preterm and term neonates: secondary results from the
    NEOPAIN trial. *British Journal of Anaesthesia*. 2008;101(5):680-689.
    doi:10.1093/bja/aen248. PMID 18723857.
17. Soffer OD, Cornelissen L, Cummings C, Berde C. Morphine compared to placebo
    for procedural pain in preterm infants: safety, efficacy and equipoise.
    *Journal of Perinatology*. 2019;39(10):1428-1431.
    doi:10.1038/s41372-019-0476-9. PMID 31455824.
18. Polin RA, Carlo WA; Committee on Fetus and Newborn; American Academy of
    Pediatrics. Surfactant replacement therapy for preterm and term neonates
    with respiratory distress. *Pediatrics*. 2014;133(1):156-163.
    doi:10.1542/peds.2013-3443. PMID 24379227.
19. Dani C, Lo Nigro M, Poggi C, Corsini I, Pratesi S. Surfactant retreatment in
    very preterm infants with respiratory distress syndrome: an observational
    cohort study. *Translational Pediatrics*. 2026;15(4):118.
    doi:10.21037/tp-2025-aw-794. PMID 42158663.
20. Cristóbal Cañadas D, Bonillo Perales A, Galera Martínez R,
    Casado-Belmonte MDP, Parrón Carreño T. Effects of kangaroo mother care in
    the NICU on the physiological stress parameters of premature infants: a
    meta-analysis of RCTs. *International Journal of Environmental Research and
    Public Health*. 2022;19(1):583. doi:10.3390/ijerph19010583. PMID 35010848.
21. Durmaz A, Sezici E, Akkaya DD. The effect of kangaroo mother care or
    skin-to-skin contact on infant vital signs: a systematic review and
    meta-analysis. *Midwifery*. 2023;125:103771.
    doi:10.1016/j.midw.2023.103771. PMID 37454580.
22. Mitchell AJ, Yates C, Williams K, Hall RW. Effects of daily kangaroo care on
    cardiorespiratory parameters in preterm infants. *Journal of
    Neonatal-Perinatal Medicine*. 2013;6(3):243-249.
    doi:10.3233/npm-1370513. PMID 24246597.
23. Akbari N, Mutlu B, Nadali J. Effect of non-nutritive sucking during
    heel-stick procedure in pain management of term infants in the neonatal
    intensive care unit: a systematic review and meta-analysis. *Current
    Pediatric Reviews*. 2022;19(1):90-98.
    doi:10.2174/1573396318666220410225908. PMID 35410609.
24. Gao H, Li M, Gao H, Xu G, Li F, Zhou J, Zou Y, Jiang H. Effect of
    non-nutritive sucking and sucrose alone and in combination for repeated
    procedural pain in preterm infants: a randomized controlled trial.
    *International Journal of Nursing Studies*. 2018;83:25-33.
    doi:10.1016/j.ijnurstu.2018.04.006. PMID 29684832.
