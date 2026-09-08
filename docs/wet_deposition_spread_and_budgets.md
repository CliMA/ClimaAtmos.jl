# Wet Deposition: What Drives the Inter-Model Spread, and How Much Does Each Process Remove?

*Literature synthesis (2026-08-14) for the sea-salt wet-deposition build on
`zg/ssa-wetdep` (Liu et al. 2001 structure + Feng 2007 below-cloud). Companion
to `wet_deposition_schemes_audit_sea_salt.md` (scheme survey) and
`sea_salt_wet_deposition_liu2001_menu.md` (CliMA-native options). Compiled from
five parallel literature sweeps; numbers marked ✓ were read from full-text
PDFs/tables, numbers marked (a) are abstract/secondary-source level.*

---

## 1. What explains the spread between wet deposition schemes

### 1.1 It is the removal parameterizations, not the inputs

Three independent lines of evidence:

- **Harmonized-emissions experiment** ✓ (Textor et al. 2007, ACP 7, 4489):
  giving all AeroCom models identical emissions, injection heights, *and
  source size distributions* had "only a small impact" on burden diversity.
  Sea salt: emission diversity collapsed δ = 100% → 3%, but lifetime diversity
  only fell 59% → 24% and the wet-flux diversity 67% → 45%. The residual is
  pure removal/size-processing parameterization. For fine aerosols the wet/dry
  split changed < 5% per model between experiments — the split is a *property
  of the model*, insensitive to sources.
- **Identical-source tracer experiment** ✓ (Kristiansen et al. 2016, ACP 16,
  3525): 19 models given the same Fukushima ¹³⁷Cs source on their default
  accumulation-mode tracer produced e-folding lifetimes **4.8–26.7 d (factor
  ~5.6)**, median 9.4 d, vs. a purely observational 14.3 d (95% CI 13.1–15.7,
  from measured ¹³⁷Cs/¹³³Xe ratios — the co-released noble gas cancels
  transport dilution). Most models remove aged aerosol **too fast**. A NAME
  sensitivity run dividing scavenging coefficients by 10 moved its lifetime
  5.5 → 13.1 d — the coefficient magnitude *is* the diagnosed lifetime.
- **Not the precipitation fields** ✓ (Textor et al. 2006, ACP 6, 1777): no
  correlation across 16 models between global precipitation rate (2.5–3.5
  mm/d, tight) and either wet fraction or wet rate coefficient.

### 1.2 Attribution to scheme components (single-model swap experiments)

| Component varied | Effect | Source |
|---|---|---|
| In-cloud scheme structure (prescribed ratio vs diagnostic vs prognostic cloud-borne) | 20–30% global mass burdens; ~50% accumulation-mode *number* burden; order-of-magnitude concentrations in the T < 273 K mid-troposphere | Croft 2010 (a) |
| Mixed/ice-phase in-cloud treatment | 7× ice-cloud number removal between impaction kernels; 2 orders of magnitude in dust impaction mass flux; identified by Kristiansen 2016 as *the* discriminating knob for aged-aerosol lifetime (late-phase τ set by scavenging at T < 258 K) | Croft 2010 (a), Kristiansen 2016 ✓ |
| Below-cloud scheme (mode-scaled → size-resolved) | Sea-salt below-cloud flux ×8 (153 → 1250 Tg/yr), burden −15%, partly buffered by a −21% compensating drop in in-cloud removal | Croft 2009 ✓ |
| Raindrop size distribution alone | 3–5× in Λ at coarse sizes; sea-salt below-cloud flux 755–1870 Tg/yr (2.5×) between 0.4 mm and 4 mm monodisperse drops | Wang 2010 ✓, Croft 2009 ✓ |
| Collection efficiency formulation | 1–2 orders of magnitude in Λ — but only for 0.01–3 μm, where Λ is tiny anyway; **coarse-mode (> 10 μm) Λ converges to ~10⁻⁴ s⁻¹ across schemes** | Wang 2010 ✓, Jones 2022 ✓ |
| Empirical (Laakso-type) vs theoretical Λ | ~10× at 1 μm; UM coarse dust lifetime 0.9–4 d, accumulation 5.4–43.8 d across 5 schemes | Jones 2022 ✓ |
| **1-moment vs 2-moment (mode cannot shrink under scavenging)** | Coarse dust burden 13.2 vs 5.25 Tg (**2.5×**), wet fraction 23% vs 54% — *larger than any scheme choice* | Jones 2022 ✓ |
| Convective coupling (scavenging inside vs outside the transport operator; entrained-air scavenging above cloud base) | 20–35% global burdens; 5–10× Arctic BC; order of magnitude in UT concentrations; Murphy 2019 traced a >100× CESM upper-troposphere sea-salt overestimate to missing removal in convective sub-grid transport | Croft 2012 (a), H. Wang 2013 (a), Murphy 2019 ✓ |
| Re-evaporation/resuspension treatment | ~60% of formed large-scale precip evaporates before landing; naive full release: SS burden +15%; drop-size-dependent release (Gong 2006: 57% water evap → only ~20% aerosol released): SS effect ≈ −0.6 to −6% | de Bruine 2018 ✓ |
| Precipitating-fraction / cloud overlap | doubles upper-troposphere soluble-tracer lifetime (HNO₃) vs simple fractional-cover assumptions | Neu & Prather 2012 (a) |

Textor 2006's own list of named drivers ✓: interstitial-vs-scavenged fraction
treatment, size dependence of scavenging (few models resolve it), spatial
coincidence of aerosol and precipitation, ice-phase scavenging "considered
only in some models," **constant prescribed cloud water (Giorgi–Chameides
style) instead of host-model LWC**, cloud-fraction scaling, and
calculation-order artifacts (whichever removal process is computed first
claims more mass).

### 1.3 Sea-salt-specific hierarchy

For sea salt the spread ordering differs from fine aerosol because most mass
is coarse ✓ (Textor 2006, Table 10; Gliß 2021):

1. **Emitted size distribution / size cutoff** — emission diversity δ = 199%;
   AeroCom III explicitly attributes SS lifetime/burden spread to size +
   water-uptake assumptions *feeding* removal, not wet-scheme structure.
   The same model's wet fraction moves 0.37 → 0.49 just by changing emission
   scheme ✓ (Spada 2013).
2. **Dry/sedimentation split** — dry removal rate δ = 219%, sedimentation
   fraction of dry dep 59% (δ = 65%).
3. **Wet removal coefficient** — δ = 77% (SS has the *highest* wet rate
   coefficient of all species — low wet fraction is a size effect, not low
   scavenging efficiency).
4. **Convective vs stratiform attribution** — convective share of wet dep
   ranges **10–85%** across models (δ = 53%); "models do not agree on the
   rain type which is most efficient."

AeroCom SS lifetime: mean 0.48 d / median 0.41 d (δ = 58%) in phase I ✓;
median 0.56 d, diversity 92% in phase III ✓. Burden diversity (38%) is
smaller than lifetime diversity because emission and lifetime errors
anti-correlate (models emitting more coarse mass also remove it faster).

---

## 2. How much mass/number each process removes

**There is no observational measurement of this split** — every number below
is an *online model budget diagnostic* (global annual sink flux per process,
often normalized by burden into rate coefficients). Observations constrain
integrated lifetimes and concentrations; the process attribution is
model-internal. That caveat is itself a headline finding.

### 2.1 Sea-salt mass budgets

| Model / study | Wet / total removal | In-cloud | Below-cloud | Dry dep | Sedimentation | Lifetime |
|---|---|---|---|---|---|---|
| AeroCom I mean (12–15 models) ✓ | **30.5%** (δ = 65%) | — | — | — | 59% of dry | 0.48 d |
| AeroCom A/B medians ✓ | 0.21 / 0.28 | — | — | — | — | 7.2 / 14.4 h |
| ECHAM5-HAM (Croft 2010, DIAG-FULL, Table 10) ✓ | **61%** (high outlier) | 38% | 23% | 17% | 22% | 0.53–0.87 d across schemes |
| GEOS-Chem (Jaeglé 2011) ✓ | 0.36–0.40 | — | — | — | — | 7.9–8.4 h |
| NMMB/BSC sectional (Spada 2013) ✓ | 0.37–0.49 (emission-scheme dependent) | — | — | — | — | 7.3–11.3 h bulk; fine bins 25–37 h, 1–4 μm 12–17 h |

Structure within the wet term (Croft 2010, the only fully process-resolved
table found ✓):

- **In-cloud SS removal is 99% nucleation scavenging** (activation), < 1%
  in-cloud impaction. (Dust is the exception at ~51/49 — insoluble.)
- **Convective ≈ 31% of SS in-cloud removal** (34% AeroCom 8-model mean of
  wet dep, range 10–85%).
- Below-cloud washout is a genuine **~23% mass channel for SS** (25% dust) vs
  only 11–14% for fine species — but this is with the *size-resolved* scheme;
  the mode-scaled predecessor gave 3%, and Henzing 2006 quotes ~12%. The
  below-cloud share is itself scheme-dependent to a factor of several.
- Size decides the dominant sink ✓ (Bian 2019, GEOS + ATom): dry diameter
  < 3 μm — wet removal dominates; > 3 μm — **sedimentation alone exceeds 1.5×
  all other processes combined**. The bulk "wet fraction 0.2–0.5" spread is
  largely an emission-cutoff artifact; the wet fraction of the *fine* SS that
  escapes the boundary layer is near 1.

For the submicron proxy in Liu's own lineage: the GEOS CTM ²¹⁰Pb sink was
**74% convective precipitation, 12% large-scale** (a) — the Liu-2001 family is
notably convective-heavy.

### 2.2 Number vs mass

Croft 2010, Table 11 ✓ (all modes, global): in-cloud dominates number removal
(~7× dry deposition), but within stratiform in-cloud scavenging **impaction is
> 90% of the *number* flux** (nucleation-mode particles too small to
activate), 99% of it in T < 273 K clouds — the mirror image of the mass split
(94–99% nucleation). **Below-cloud washout is < 2% of number removal** (the
Greenfield gap). So: mass leaves through activation + rain; number leaves
through cold-cloud impaction; washout matters for coarse *mass* only. No
global sea-salt number-vs-mass wet budget was found (gap); the closest
observation is Murphy 2019's size-uniform (0.3–3 μm) depletion, implying
activation scavenging removes number and mass at comparable fractional rates.

### 2.3 Re-evaporation

de Bruine 2018 ✓ (EC-Earth/TM5): ~60% of formed large-scale precipitation
evaporates before reaching the surface. Release treatment matters: naive
(release ∝ evaporated fraction, the Liu 0.5·f family) gives SS burden +15%;
the Gong (2006) drop-size-dependent relation (57% of water evaporated → only
~20% of aerosol released, as fewer *whole* drops evaporate) cuts SS
resuspension ~70%, leaving −0.6 to −6% burden effects. Resuspended SS is
coarse and re-settles quickly, so re-evaporation matters much less for SS
than for fine species (−10 to −19%).

---

## 3. How these numbers are estimated (and what each method can/can't see)

**Model budget diagnostics** (Textor, Croft, Spada, Jaeglé, Gliß): online
accumulation of each process's global sink flux; rates k_i = flux/burden;
diversity δ = stdev/mean. Can attribute per process; cannot be validated per
process — only the totals meet observations. Textor notes calculation-order
artifacts bias the split itself.

**Radionuclides:**
- ²¹⁰Pb (surface ²²²Rn source) + ⁷Be (upper-troposphere cosmogenic source)
  bracket the *vertical profile* of scavenging; validated against worldwide
  surface-concentration and deposition-flux networks. They constrain
  column-integrated wet lifetime (~9 d for ²¹⁰Pb in Liu 2001), **not process
  attribution** — convection, stratosphere–troposphere exchange, and source
  errors alias directly into inferred scavenging (Liu 2001 had to retune the
  ⁷Be stratospheric source 3–4×). ¹⁰Be/⁷Be isolates transport (both attach to
  the same aerosol) and breaks that degeneracy.
- Fukushima ¹³⁷Cs/¹³³Xe ✓: the cleanest *pure-observation* lifetime — the
  ratio to the co-released noble gas cancels transport/dilution, giving
  τ_e = 14.3 d for aged accumulation-mode aerosol with no model involved.
  Limits: aged/free-troposphere aerosol only; nearly orthogonal to fresh
  boundary-layer sea-salt removal.
- Residence time depends on injection altitude ✓ (Balkanski 1993: ²¹⁰Pb
  produced in the lowest 0.5 km lives ~4× shorter than aloft) — "the" aerosol
  lifetime is not one number; surface-emitted sea salt sits at the short end.

**Activated-fraction (nucleation-scavenging) measurements**: total vs
interstitial inlets in cloud (aircraft/mountain). Observed activation
diameters D₅₀ ≈ 39–220 nm depending on supersaturation/LWC ✓ (Motos 2019,
Henning 2002, Sellegri 2003); number scavenged fraction 50% → >90% as peak
supersaturation rises 0.13% → 0.55%. Everything ≳ 0.2 μm dry activates —
i.e., essentially all sea-salt *mass* — supporting f_act ≈ 1 for SSA mass
(no direct sea-salt activated-fraction measurement was found; inference from
D₅₀). Ohata 2016: even at the surface, below-cloud impaction was ≤ 6–11% of
BC wet removal — nucleation scavenging dominates fine-mode removal.

**Field below-cloud Λ**: aerosol timeseries through rain events
(Λ = ln(N₀/N₁)/Δt) and sequential precipitation sampling (early millimeters =
washout, plateau = rainout). Measured Λ run **1–2 orders above Slinn-type
theory** for 0.1–3 μm ✓ (Wang 2010, Xu 2019) — but the wind-shielded-chamber
experiment (Sparmacher 1993) *agrees with theory*, implying the field excess
is largely contamination: advection, collocated in-cloud removal delivered by
the same rain, hygroscopic growth. Two design consequences: (i) field-fit Λ
(Laakso-type) already *contains* rainout — using it below cloud in a model
that also has a rainout term double-counts; (ii) field Λ datasets essentially
don't constrain d > 3 μm — exactly the sea-salt coarse mode. There, scheme
convergence (~10⁻⁴ s⁻¹ at drizzle) is the justification for a theoretical
Feng-type law.

**Scavenging ratios** W = C_precip/C_air: North Atlantic long-term means
~210–374 for nss-sulfate/nitrate/MSA (a); Na⁺ commonly quoted higher
(coarse). Cancels emission bias (useful model metric) but embeds cloud-base
height, precip type, and evaporation; varies orders of magnitude on short
timescales.

**Deposition networks + concentration pairs**: harmonized Na⁺ wet-deposition
fluxes (Vet 2014 global assessment) close budgets; the diagnostic signature
is instructive — EMEP ✓ (Tsyro 2011) *overestimates* airborne Na⁺ 8–46% while
*underestimating* Na⁺ in precipitation 65–70%: too-weak wet removal (and/or
missing coarse source), invisible in either metric alone.

**Trajectory/satellite methods**: accumulated-precipitation-along-trajectory
suppression works for fine aerosol ✓ (Dadashazar 2021: −53% PM2.5/ΔCO for
wet vs dry transport) but **fails for sea salt specifically** — wind-driven
emission is enhanced during the same storms that scavenge, so the signal
cancels. No CloudSat×aerosol sea-salt scavenging product exists.

---

## 4. Implications for the `zg/ssa-wetdep` build

1. **Validate size-resolved, not bulk.** The bulk wet fraction (0.2–0.5) is an
   emission-cutoff artifact. Targets: per-bin lifetimes (fine 25–37 h, 1–4 μm
   12–17 h), the ~10×/2 km vertical decay of SS ✓ (Murphy 2019), the
   "SS removal tracks water removal, log-log slope ≈ 1" relation, and
   Univ. Miami/PMEL surface concentrations + Na⁺ wet-deposition fluxes.
   Build per-process deposition diagnostics from day one — the split is only
   ever knowable from the model.
2. **The updraft term is the highest-stakes piece.** Murphy's >100× CESM
   upper-troposphere SS bug from missing convective-transport scavenging, the
   74% convective share in Liu's own budget, and the 10–85% inter-model range
   all say the EDMF-coupled scavenging (menu §1) carries the most structural
   risk/reward. Our plan (sink on `sgsʲs` tracers inside the transport, not a
   grid-mean afterthought) is on the right side of Balkanski/Giannakopoulos.
3. **Our sectional bins partially dodge the biggest below-cloud pitfall.**
   Jones 2022's 2.5× coarse-burden error is from a *modal* 1M scheme whose
   mode cannot shrink; five independent mass bins let the size distribution
   shift as coarse bins deplete faster. The residual limitation is frozen
   *sub-bin* shape — acceptable at our bin resolution.
4. **Feng coefficients: theoretical is the defensible choice for SSA.** At
   the coarse sizes carrying sea-salt mass, schemes converge (~2–4× spread)
   and the empirical alternatives (Laakso) are extrapolated and
   rainout-contaminated there. The cheap high-value sensitivity is the
   raindrop-size assumption (2.5× in SS washout flux) — worth one toggle.
   Do not calibrate (a, b) against field Λ (double counting).
5. **Cold-cloud behavior is the aged-lifetime knob.** Kristiansen: models are
   too fast on aged aerosol, and the discriminator is scavenging at
   T < 258 K. Our process-based liquid-only rainout (menu §2, replacing the
   258 K gate) is exactly the right lever — and errs on the side of *less*
   cold-cloud removal, the direction observations favor.
6. **Re-evaporation: Liu's 0.5·f is the naive end.** de Bruine shows
   drop-size-dependent release cuts the effect ~3× for SS and that resuspended
   coarse SS mostly re-settles; expected SS burden sensitivity is single-digit
   percent. Fine to defer or keep 0.5 as a ClimaParams knob; revisit with 2M
   rain (drop number enables the shrink-vs-evaporate split).
7. **Compensation is real: don't tune components in isolation.** Croft 2009's
   8× washout increase was buffered to −15% burden by a compensating rainout
   drop. Budget diagnostics per process are the only way to see whether a
   "successful" burden came from the right split.

---

## Key sources

Textor et al. 2006 (10.5194/acp-6-1777-2006); Textor et al. 2007
(10.5194/acp-7-4489-2007); Kristiansen et al. 2016 (10.5194/acp-16-3525-2016)
and 2012 (10.5194/acp-12-10759-2012); Croft et al. 2009 (10.5194/acp-9-4653-2009),
2010 (10.5194/acp-10-1511-2010), 2012 (10.5194/acp-12-10725-2012); Wang, Zhang
& Moran 2010 (10.5194/acp-10-5685-2010); Jones et al. 2022
(10.5194/acp-22-11381-2022); Jung & Shao 2006 (10.1016/j.gloplacha.2006.02.008);
Gliß et al. 2021 (10.5194/acp-21-87-2021); Spada et al. 2013
(10.5194/acp-13-11735-2013); Jaeglé et al. 2011 (10.5194/acp-11-3137-2011);
Grythe et al. 2014 (10.5194/acp-14-1277-2014); Bian et al. 2019
(10.5194/acp-19-10773-2019); Murphy et al. 2019 (10.5194/acp-19-4093-2019);
de Bruine et al. 2018 (10.5194/gmd-11-1443-2018); Luo et al. 2019/2020
(10.5194/gmd-12-3439-2019, 10.5194/gmd-13-2879-2020); Balkanski et al. 1993
(10.1029/93JD02456); Liu et al. 2001 (10.1029/2000JD900839); Motos et al. 2019
(10.5194/acp-19-3833-2019); Henning et al. 2002; Sellegri et al. 2003; Ohata
et al. 2016; Xu et al. 2019 (10.5194/acp-19-15569-2019); Sparmacher et al.
1993; Laakso et al. 2003; Tsyro et al. 2011 (10.5194/acp-11-10367-2011); Vet
et al. 2014; Dadashazar et al. 2021 (10.5194/acp-21-16121-2021); Neu & Prather
2012 (10.5194/acp-12-3289-2012); H. Wang et al. 2013 (10.5194/gmd-6-765-2013);
Holopainen/Tonttila et al. 2020 (10.5194/gmd-13-6215-2020).

*Verification flags carried over from the sweeps: Liu 2001 budget fractions
(74/12 convective/large-scale) are from secondary quotes; Gliß 2021 Table-3
medians were machine-extracted (spot-check before citing exactly); Feng 2007
coarse-mode coefficients verified separately from the paper PDF in hand;
Jung & Shao 2006, Croft 2010/2012 percentages, Savoie Na⁺ scavenging ratio,
and Vet 2014 details are abstract-level.*
