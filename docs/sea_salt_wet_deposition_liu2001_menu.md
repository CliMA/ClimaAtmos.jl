# Liu et al. (2001) Wet Deposition in ClimaAtmos — CliMA-Native Options Menu

> **Scope note:** Atmospheric/climate physics only — how rainfall removes
> natural sea-spray salt aerosol in the ClimaAtmos climate model (see the
> matching note in `sea_salt_wet_deposition_immediate_plan.md`).

*Branch context: `zg/ssa-wetdep-belowcloud`, stacked on `zg/ssa-drydep` (dry deposition
on the mass-only sea-salt scheme; derived spectrum quantities come from the
closed-form per-bin lognormal moments cached in `p.tracers.sslt_bin_moments`). Paradigm: `NonEquilibriumMicrophysics1M` +
`PrognosticEDMFX`. This doc maps each constant/assumed input of the Liu et al.
(2001, JGR 106, D11, 12109) scheme to what ClimaAtmos already diagnoses, with a
menu per input. Companion docs: `wet_deposition_schemes_audit_sea_salt.md`
(scheme survey) and `sea_salt_wet_deposition_immediate_plan.md` (older,
pre-rework plan — its architecture references are stale but §3's
activation-coupling idea still applies).*

Paper values for reference: updraft α = C₁/w = 5×10⁻⁴ m⁻¹ (C₁ = 5×10⁻³ s⁻¹,
w = 10 m/s); stratiform F₀ = 1, C₁ = 10⁻⁴ s⁻¹ + Qₖ/L, L = 1.5×10⁻³ kg m⁻³;
convective F₀ = 0.3, C₁ = 1.5×10⁻³ s⁻¹, L = 2×10⁻³ kg m⁻³; rainout suppressed
below 258 K; washout 0.1 mm⁻¹ over the max-overhead precipitating fraction;
re-evaporative release of 0.5·f of the scavenged load (f = evaporated flux
fraction), full release at f = 1.

---

## Notation and abbreviations

**Water-species suffixes** (used in state names `ρq_*`, specific humidities
`q_*`, and tendencies `dq_*_dt`):

| Code | Species |
|---|---|
| `vap` | water vapor |
| `lcl` | **l**iquid **cl**oud condensate — cloud droplets (suspended liquid) |
| `icl` | **i**ce **cl**oud condensate — cloud ice (suspended ice) |
| `rai` | rain (precipitating liquid) |
| `sno` | snow (precipitating ice) |
| `tot` | total water (vapor + all condensate + all precipitation) |

Trap worth repeating: the *precomputed* `ᶜq_liq` / `ᶜq_ice` are **totals** —
`q_liq = q_lcl + q_rai`, `q_ice = q_icl + q_sno` — they exist to feed the
thermodynamics, not to represent "cloud water."

**Microphysics process source terms** (from CloudMicrophysics'
`BulkMicrophysicsTendencies`, all kg/kg/s) follow the pattern
`S_<process>_<from>_<to>`:

| Piece | Meaning |
|---|---|
| `S_…` | a single-process source/sink rate ("S" for source term) |
| `acnv` | **a**uto**c**o**nv**ersion — cloud droplets colliding with *each other* until some grow into rain (cloud liquid → rain), or cloud ice aggregating into snow |
| `accr` | **accr**etion — falling precipitation *collecting* suspended cloud condensate as it sweeps through (e.g. rain collecting cloud droplets) |
| `phase_change` | condensation/evaporation or deposition/sublimation against vapor |
| `melt` / `freeze` | thermal phase change between the liquid and ice branches |
| `dq_*_dt` | the **net** tendency of one species — the signed sum of every `S_*` term touching it |

Examples: `S_acnv_lcl_rai` = autoconversion of cloud liquid to rain;
`S_accr_lcl_rai` = rain accreting cloud liquid; `S_phase_change_vap_rai` =
rain ↔ vapor exchange, i.e. **rain evaporation** (clamped ≤ 0 in CM — negate
it for a positive evaporation rate); `S_accr_lcl_sno_warm` = snow collecting
cloud liquid in the warm regime, where the result melts into rain.

**Model / framework abbreviations:**

| Abbrev. | Meaning |
|---|---|
| `mp` | microphysics (`ᶜmp_tendency` = cached microphysics tendencies) |
| 0M / 1M / 2M | 0-, 1-, 2-moment bulk microphysics: 0M removes supersaturation with no precipitation species; 1M carries mass (`q`) of each species; 2M adds number concentrations |
| CM / `CM1` | CloudMicrophysics.jl (its 1-moment submodule) |
| BMT | `CloudMicrophysics.BulkMicrophysicsTendencies` — the fused tendency API |
| `InstantaneousVerbose` | BMT mode returning every individual `S_*` process term at the current state |
| `LinearizedAverage` | BMT mode used in production: substeps the (linearized) microphysics over `dt` and returns the *effective average* `dq_*_dt` |
| `nsubs` | number of those substeps |
| TD / `thp` | Thermodynamics.jl / thermodynamics parameters |
| `cmp` | cloud microphysics parameters |
| CMAA | `CloudMicrophysics.AerosolActivation` (CCN activation) |
| CDNC | cloud droplet number concentration |
| EDMF(X) | eddy-diffusivity mass-flux turbulence scheme; PrognosticEDMFX carries prognostic updraft state |
| `sgsʲs` / superscript `ʲ` | sub-grid-scale updraft `j`'s state (`Y.c.sgsʲs.:(1).q_lcl` = cloud liquid *inside* updraft 1) |
| superscript `⁰` | the environment (grid mean minus all updrafts; reconstructed, not prognostic) |
| `ρa`, `ρaʲ`, `ρa⁰` | area-weighted density: `ρ·a` where `a` is the subdomain's area fraction; `draft_area(ρa, ρ) = a` |
| `ᶜ` / `ᶠ` prefixes | cell-center / cell-face fields |
| `ᶜwᵣ`, `ᶜwₛ`, `ᶜwₗ`, `ᶜwᵢ` | terminal (sedimentation) velocities of rain, snow, cloud liquid, cloud ice [m/s, positive down] |
| `ᶜsslt_ξ` | sea salt hygroscopic growth factor ξ = r_wet/r_dry (cached) |
| SSLT0x | the five sea salt dry-radius bins (MERRA-2 convention) |
| SGS quadrature | integrating microphysics over assumed sub-grid (T, q_tot) fluctuations instead of the mean state |

**Liu et al. (2001) symbols:** α = updraft scavenging efficiency per meter of
ascent; C₁ = rate constant for conversion of cloud water to precipitation
[s⁻¹]; L = cloud condensed water content [kg m⁻³]; Qₖ = grid-scale
precipitation formation rate in layer k [kg m⁻³ s⁻¹]; Fₖ = fraction of the
grid cell in layer k experiencing precipitation; F₀ = its maximum. Feng
(2007): Λ = below-cloud scavenging coefficient [h⁻¹], R = rain rate [mm/h].

---

## 0. Structural mapping: Liu's three processes collapse to two (plus bookkeeping)

Liu's operator split — convective-updraft scavenging (α·dz), stratiform
rainout, convective-anvil rainout — was forced by an offline CTM whose
archived convective mass fluxes hid the inside of convection. All three are
the **same physical process**: aerosol residing in cloud water is removed at
the rate cloud water converts to precipitation. Written as rates, Liu's
updraft term is `k = α·w = C₁` (s⁻¹, conditional on being in cloud), and his
grid-mean rainout is `F·(1−e^{−C₁Δt})/Δt` — whose linear limit is `F·C₁ =
F₀·Q/L` **with C₁ cancelling identically** (mass balance: scavenged aerosol =
(aerosol per cloud water) × (cloud-water→precip rate); no rate constant can
survive in that product; the cancellation holds even with Liu's
`C₁ = C₁,min + Q/L`). F and C₁ are bookkeeping that splits one rate into
"how wide" × "how fast"; the split has consequences only through (a) the
per-step saturation cap (removal ≤ F per step — the exponential), and
(b) F's reuse by washout, re-evaporation, and the downward max propagation.
One process, two area closures.

In prognostic EDMF the subdomain decomposition *is* that geometry, so the
scheme reduces to:

| Process | Applied to | Geometry closure |
|---|---|---|
| **In-cloud scavenging** (one kernel, §1) | updraft `j`: `Y.c.sgsʲs.:(j).SSLT0x` (already prognostic!) + `ρaʲ`-weighted mirror on `Y.c.ρSSLT0x` | none — `F ≡ 1` where `q_lclʲ > 0`; `ρaʲ` is the area weight |
|  | environment: grid-mean `Y.c.ρSSLT0x`, driven by the environment state | precipitating-fraction `F_k` closure (§2.2) |
| **Below-cloud washout** (§3) | grid-mean `Y.c.ρSSLT0x` (sub-cloud environment air; no updraft analogue) | column-max `F` from overhead |
| **Re-evaporation release** (§2.3) | grid-mean, from a precip-borne scavenged-load accumulator | follows the precipitating column |

Liu's *convective-anvil* rainout dissolves entirely: detrained updraft
condensate becomes environment cloud water, so anvil precipitation formation
appears in the environment's `Q` and is scavenged by the environment term —
no third operator.

Facts that make this clean, and one rule that keeps it conservative:
- **Updraft aerosol tracers already exist** (`setups/common/prognostic_variables.jl:141–157`)
  and are entrained/advected/mass-fluxed by the generic SGS machinery
  (`sgs_tracer_names`, `edmfx_entr_detr.jl:507`, `edmfx_sgs_flux.jl:141`).
- **Conservation rule**: the environment tracer is *reconstructed*
  (`ᶜspecific_env_value`), so every updraft sink MUST be mirrored onto the
  grid mean with the `ρaʲ` weight (pattern: `microphysics/tendency.jl:139–147`),
  else the removed mass reappears in the diagnosed environment. Same pattern
  as the dry-dep updraft mirror.
- **Double-counting rule**: the environment term must be driven by the
  *environment-only* formation rate (`ᶜmp_tendency⁰`, weighted by `ρa⁰`),
  never a grid-mean total — otherwise updraft-formed precipitation scavenges
  aerosol twice. Conveniently, under PrognosticEDMFX the grid-mean
  `ᶜmp_tendency` is not even populated (`microphysics_cache.jl:906–947`), so
  the cache structure steers you right.
- Detrainment needs no special handling: scavenging acts on `χʲ`; detrained
  air carries the already-depleted value back through the mass-flux
  difference term. Liu's Lagrangian "lose α·dz during ascent" emerges from
  local-sink + updraft vertical advection of `χʲ`.
- **Diagnostics stay split even though the kernel is shared**: accumulate
  updraft vs environment scavenging fluxes separately — the convective share
  of wet removal ranges 10–85% across models and is only knowable from these
  diagnostics (see `wet_deposition_spread_and_budgets.md`).

---

## 1. The in-cloud scavenging kernel (shared by both subdomains)

One first-order sink per bin, per subdomain:

```
∂χ/∂t = − F · f_act(bin) · c₁ · χ ,   c₁ = (Q/L)|in-cloud   [s⁻¹]
```

**`c₁` is intensive** — the cloud-water→precipitation conversion rate *per
unit cloud water inside the cloud* (Liu's C₁) — and `F` is the
precipitating-area fraction (≡ 1 in updrafts; §2.2 closure in the
environment). `f_act` is the activated fraction (§5; ≡ 1 for draft 1).

**Intensive/extensive pairing rule (the real double-counting hazard).**
Because conversion only happens where cloud water is, the diagnosed ratios in
K2/K3 below (`dq_rai_dt / q_lcl` on a subdomain's mean state) have the same
area dilution in numerator and denominator — it cancels, so **K2/K3 yield the
intensive `c₁`**, which is *correct to multiply by F*. Equivalently, F can be
eliminated: `F·c₁ = Q_grid/L_ic` (grid-mean formation rate over *in-cloud*
condensed water), capped at `F₀·c₁`. In Liu's own stratiform closure this
cancellation is exact — `F·C₁ = [F₀·Q/(L·C₁)]·C₁ = F₀·Q/L`, even with
`C₁ = C₁,min + Q/L` — i.e. C₁ never affects the linear rainout rate, only the
per-step saturation (`F·(1−e^{−C₁Δt})` ≤ F) and the F that washout and
re-evaporation reuse. The two *wrong* pairings: an intensive `c₁` with no F
(scavenges clear-sky aerosol as if in cloud — overcounts), or an
already-area-diluted grid-mean Q/L multiplied by F (double-dilutes —
undercounts).

This rate form is exactly Liu's physics — his α·dz *is* C₁·dt along the
parcel — but never divides by w, is unconditionally regularizable
(`min(k, cap/dt)`, the `limit_entrainment` idiom), and reduces to Liu when
c₁ is constant.

### K. The intensive conversion rate c₁ — menu, cheapest first

Each option is evaluated on the SUBDOMAIN's own state: updraft `j` uses
(`ᶜmp_tendencyʲs.:(j)`, `q_lclʲ`), environment uses (`ᶜmp_tendency⁰`,
environment cloud water).

| Option | What | Cost | Caveats |
|---|---|---|---|
| **K1. Liu constants** | `C₁ = 5×10⁻³ s⁻¹` (updraft) / `10⁻⁴ s⁻¹ + Q/L` with `L = 1.5×10⁻³ kg m⁻³` (environment), gated on cloud water > 0 | 2–3 ClimaParams keys | The calibration anchor; keep as fallback/toggle. Keeping L and C₁ constant is also what keeps the F-inversion (§2.2) meaningful. |
| **K2. Net cached rate (recommended first draft)** | `k = max(0, dq_rai_dt + dq_sno_dt) / max(q_lcl + q_icl, ε)` from the subdomain's cached tendencies | **zero new plumbing** — cached every stage (`microphysics_cache.jl:881–947`) | `dq_rai_dt` is the *net* rain tendency (includes rain evap ≤ 0, melt/freeze exchange); the `max(0,·)` clamp handles the sign but misattributes vapor terms. In-subdomain (specific) units — exactly Liu's "in-cloud" convention. |
| **K3. Process-level C₁ (the "right" one)** | `k = (S_acnv_lcl_rai + S_accr_lcl_rai [+ S_accr_lcl_sno_*]) / max(q_lcl, ε)` — i.e. (autoconversion of cloud liquid to rain + rain collecting cloud liquid) per unit cloud liquid — via `BMT.bulk_microphysics_tendencies(InstantaneousVerbose(), …)` on the subdomain state | one extra `InstantaneousVerbose` call per subdomain in `set_microphysics_tendency_cache!` (`microphysics_cache.jl:881`), or extend `MP1_NT`; extractor template at `microphysics_diagnostics.jl:24–32` | Updraft path is non-quadrature — trivial substitution. Environment path uses SGS quadrature (`microphysics_cache.jl:924–940`); a mean-state verbose call is not quadrature-consistent (document and accept). Instantaneous vs production `LinearizedAverage` is a mild, documentable inconsistency. |
| **K4. Activated-fraction refinement** | multiply by per-bin `f_act` from CMAA (§5) instead of assuming all aerosol is in the condensed phase | moderate | Matters for SSLT01–02 number/CCN, not coarse mass (f_act ≈ 1). Defer; the seam is a per-bin factor in the kernel signature. |

### The phase gate — we can do strictly better than Liu's 258 K rule

Liu suppressed rainout below 258 K because the archive couldn't distinguish
rain from snow formation. **1M can**: with K3, restrict the kernel to the
*liquid* formation arms (`S_acnv_lcl_rai + S_accr_lcl_rai +
S_accr_lcl_sno_warm`) and sea salt is automatically not scavenged by
riming-free cold precipitation — no temperature threshold, and the transition
is process-based, not a step. This also errs toward *less* cold-cloud
removal, the direction the ¹³⁷Cs lifetime constraint favors (models are
systematically too fast on aged aerosol; see
`wet_deposition_spread_and_budgets.md` §1). Fallback under K1/K2: hard gate on
`ᶜT ≥ 258 K` (`TD.Parameters.T_freeze` accessor pattern,
`microphysics_cache.jl:1193–1196`) or smooth `TD.liquid_fraction` weight.

### L, if taken from the model instead of Liu's constants

| Option | What | Caveats |
|---|---|---|
| **L1. Liu constants** | `1.5×10⁻³` (environment) / `2×10⁻³ kg m⁻³` (updraft) | See K1: constants keep the F-inversion meaningful. |
| **L2. Subdomain prognostic** | environment: `ρ⁰·q_lcl⁰` (via `ᶜspecific_env_value`); updraft: `ρʲ·q_lclʲ` — note grid-mean `Y.c.ρq_lcl` is already kg m⁻³ | **Do not use `ᶜq_liq`/`ᶜq_liqʲs`** — they include rain (`precomputed_quantities.jl:682`; trap documented 3× in `cloud_fraction.jl:760–1036`). Environment grid-mean values are diluted by clear sky. |
| **L3. In-cloud (GC86's actual intent)** | `L_ic = ρq_lcl / max(ᶜcloud_fraction, ε)` | No cached in-cloud value exists; one lazy broadcast. The most defensible model-derived L for the environment term. Updrafts don't need it (the updraft *is* the cloud). |

---

## 2. Subdomain application

### 2.1 Updraft (Liu's "convective scavenging", F ≡ 1)

The kernel applies to `χʲ` wherever `q_lclʲ > 0` — no fraction, no w:

- Sink on `Yₜ.c.sgsʲs.:(j).SSLT0x` (unweighted) **and** `ρaʲ`-weighted mirror
  on `Yₜ.c.ρSSLT0x` (the §0 conservation rule).
- **Where it slots:** an `aerosol_convective_scavenging_tendency!` next to
  `microphysics_tendency!` (`remaining_tendency.jl:268`) or beside
  `aerosol_settling_tendency!` (`:323`); both see a fresh microphysics cache.
  No-op unless `turbconv_model isa PrognosticEDMFX`. If
  `microphysics_tendency_timestepping` is implicit, the explicit sink is still
  fine (it acts on aerosol, not the stiff moisture variables), but cap the
  rate at `~1/dt`.
- **w, for diagnostics/comparison only** (the per-meter α form): 
  `get_physical_w(ᶜuʲs.:(j), ᶜlg)` (`utilities.jl:417`; cached at
  `precomputed_quantities.jl:79`) — the 2M aerosol-activation code already
  does `max(0, w_component(Geometry.WVector(ᶜuʲs.:(j))))`
  (`microphysics_cache.jl:1070`). Must be floored; it is 0-clipped where
  `ρaʲ < ϵ` (`constrain_state.jl:205–214`). The production kernel never needs
  it.

### 2.2 Environment (Liu's "stratiform + anvil rainout") — the F_k closure

The environment is mostly clear sky, so the sub-grid geometry problem
survives here and only here. Liu: `F_k = F₀·Q_k/(L·C₁)` (stratiform) or
`F₀·Q_k/(Q_k + F₀·C₁·L)` (his convective-anvil form), `F_k = max(F_k,
F_{k+1})` downward.

**A degeneracy to be aware of:** F is an *inversion* of Q against assumed
in-cloud properties (L·C₁ = in-cloud precip production per unit area). If you
diagnose *both* L and C₁ from the model (K2/K3 + L2/L3), F ≈ cloud fraction by
construction and the formula stops adding information. So the menu is really
about which ONE of {L·C₁, F} you take from the model and which you keep as
the closure.

| Option | What | Caveats |
|---|---|---|
| **F1. Liu literal** | keep C₁, L constants (K1/L1); compute F from the diagnosed environment Q | Fewest changes to the paper; F becomes the model-informed quantity. Recommended for the first draft — it is the intended structure. |
| **F2. F ≡ cloud fraction proxy** | skip the inversion; `F = ᶜcloud_fraction` (already EDMF-area-weighted, `cloud_fraction.jl:982–1023`) with the kernel using in-cloud L (L3) | Conflates cloudy with precipitating (F biased high, rate correspondingly diluted — partially self-cancelling). What the old immediate-plan doc chose. |
| **F3. Built precipitating fraction** | `F_k` = running max from cloud top of `ᶜcloud_fraction · (precip flux > threshold)`, via `Operators.column_accumulate!(max, …)` — the exact idiom already used in `src/cosp/subcol.jl:122` | The only honest F; a genuinely new (small) diagnostic. `ᶜsampled_precip_fraction` exists but is COSP-gated and callback-stale (`precomputed_quantities.jl:225`, `callbacks.jl:139–168`) — not usable at tendency time. |

`F_k = max(F_k, F_{k+1})` downward propagation: the same
`column_accumulate!(max)` scan (top→down), needed by F1 and F3 anyway; it is
also what the washout term (§3) consumes.

The environment's intensive `c₁⁰` comes from `ᶜmp_tendency⁰` (K2) or its
verbose formation-only version (K3) as the ratio
`dq_*_dt⁰ / q_lcl⁰` — dilution cancels in the ratio (§1 pairing rule), so the
result multiplies F directly. Use *environment* fields, never grid-mean
totals (§0 double-counting rule: updraft-formed precipitation is already
accounted for by §2.1). The sink applies to the grid-mean `ρχ`; the small
inconsistency that F built from `ᶜcloud_fraction` includes the updrafts'
binary cloud contribution is second-order at typical updraft areas (~1%).

### 2.3 Re-evaporation (Liu: release 0.5·f of the scavenged load)

| Option | What | Caveats |
|---|---|---|
| **R1. Skip in draft 1** | pure sinks | What Stage 0 of the old branch did; lower-troposphere effect ~10% per Liu; single-digit-% burden effect for SS per de Bruine 2018 (coarse resuspended SS re-settles quickly). |
| **R2. Diagnosed f, constant 0.5** | evaporated fraction per layer `f = ρ·E·Δz / max(flux_in, ε)` with `E = −S_phase_change_vap_rai` (verbose; **clamped ≤ 0 in CM, negate it** — `Microphysics1M.jl:900–906`) and `flux_in = (Y.c.ρq_rai·ᶜwᵣ)` from the level above; release `0.5·f` (`1.0` when `f → 1`) of a scavenged-load column accumulator | Needs the same verbose cache as K3 plus a small top-down column accumulator for the scavenged load (again `column_accumulate!`). Full Liu bookkeeping; the *one* remaining constant is the 0.5 shrink-vs-evaporate split. |
| **R3. Better than 0.5?** | Not with 1M: distinguishing partial drop shrinkage from total evaporation requires drop *number* (2M rain carries `N_rai`; 1M does not). Keep 0.5 as a ClimaParams key; revisit under 2M where `dN_rai/dt|_evap / N_rai` vs `dq_rai/dt|_evap / q_rai` separates the two modes cleanly. The drop-size-dependent release of Gong et al. (2006) (57% water evap → ~20% aerosol release) is the literature-preferred middle ground. | |

---

## 3. Below-cloud washout — Feng (2007) power law

Liu used Dana–Hales 0.1 mm⁻¹; we replace with Feng's Λ = a·R^b (Λ in h⁻¹, R in
mm h⁻¹). From Feng (2007) Table 3, **marine** aerosol, bins in **ambient (wet)
diameter**:

| Feng bin | dp range (μm) | a | b |
|---|---|---|---|
| 1 (nucleation) | < 0.04 | 6.2×10⁻³ | 0.62 |
| 2 (accumulation) | 0.04–2.5 | 0.99×10⁻³ | 0.61 |
| 3 (coarse-a) | 2.5–16 | 0.80 | 0.79 |
| 4 (coarse-b) | ≥ 16 | 1.81 | 0.80 |

(Note bin 2 < bin 1: the Greenfield gap.)

### Precipitation rate R — fully available

`R [mm/h] = 3600 · Y.c.ρq_rai · ᶜwᵣ` (ρq_rai is kg m⁻³; 1 kg m⁻² = 1 mm).
`ᶜwᵣ` is cached every stage (`precomputed_quantities.jl:300`,
`microphysics_cache.jl:119`; positive-down magnitude; EDMF grid-mean version is
already mass-weighted over subdomains). Λ(R=0)=0, so the power law self-gates
to raining regions. Snow analogue `ρq_sno·ᶜwₛ` exists, but Feng (2007) is
rain-only — the snow fits are Feng (2009), not in hand; defer snow washout.
For the 1M rain-rate diagnosis do **not** use `ᶜlarge_scale_precipitation_flux`
(COSP-gated, callback-stale) or `surface_rain_flux` (positive-up, includes
cloud sedimentation). Under 0M there is no rain state, and `surface_rain_flux`
(the column integral of the 0M sink) is exactly what
`set_sslt_precipitation_shadow!` uses to build the precipitation shadow.

### Bin ↔ Feng-mode assignment (no refitting) — menu

Our tracers are the 5 bins, so per-bin (a, b) is the natural fit; the "3 Gong
modes" enter only through sub-bin mass weighting. Wet diameter of the bin mass
radius = `2 · ξ · settling_radii[bin]` — both already cached
(`ᶜsslt_ξ`, `p.tracers.sslt_settling_radii`).

| Option | What | Cost | Caveats |
|---|---|---|---|
| **3a. Static per-bin (first draft)** | assign at ξ(80% RH) ≈ 2: SSLT01–02 → Feng 2; SSLT03–04 → Feng 3; SSLT05 → Feng 4 | 2 ClimaParams vectors (`ssa_feng_a`, `ssa_feng_b` per bin) | Bins straddle Feng edges (SSLT03 wet ≈ 2–6 μm, SSLT04 ≈ 6–20 μm, SSLT05 ≈ 20–40 μm); assignment by where the *mass* sits. Simple, wrong only near boundaries. |
| **3b. RH-dynamic per-bin** | select (a, b) per level by the bin's wet diameter `2·ξ·r_bin` (branchless `ifelse` ladder over the 3 relevant windows) | free at runtime; no new params beyond the Feng table | Physically nicer (dry stratocumulus-topped MBL vs saturated columns shift SSLT03/05 across boundaries); coefficients become step functions of RH — small discontinuities. |
| **3c. Spectrum-weighted blend (best, still no refit)** | at cache build, compute each bin's mass fraction inside each Feng window from the cached per-bin lognormal moments `p.tracers.sslt_bin_moments` (closed-form `sslt_bin_moments`, read via `sslt_bin_moment(m, k)`; windows at fixed reference ξ), then `Λ_bin(R) = Σ_m w_{bin,m} · a_m R^{b_m}` | ~20 lines in `prognostic_aerosol_cache`, computed once at cache build (no quadrature, nothing added to ClimaParams) | Continuous in bin index, exactly consistent with the emission spectrum, and it *is* the "apply Feng to the Gong modes" idea expressed on the bin tracers. Reference-ξ choice (fix at 2) is the approximation. |
| **3d. DSD-derived swept volume (2026-08-25, now preferred)** | lift `Λ = accr_rate/(q_clo·E)` out of `CM1.accretion`'s closed form: `Λ_bin = E_bin · n₀·a₀χₐ·v₀χᵥ·λ⁻¹·Γ(ae+ve+Δa+Δv+1)·(λ⁻¹/r₀)^{ae+ve+Δa+Δv}` with `λ⁻¹ = CM1.lambda_inverse(rain.pdf, rain.mass, q_rai, ρ)`; only `E_bin` is new | one broadcastable function + `ssa_E_coll` per bin | Same MP DSD + power-law velocity as the model's own accretion and `ᶜwᵣ` — Λ and R become moments of one distribution, and Feng's exponent emerges analytically (`Λ ∝ R^{3.5/4.5} ≈ R^{0.78}` vs Feng's fitted 0.79–0.80). Full detail: immediate-plan doc §3b. Feng (3a–3c) demoted to cross-check. |

### Area and geometry

Washout applies over the precipitating fraction from overhead — the same
`F` menu (§2.2) and the same `column_accumulate!(max)` scan; Liu applies
washout below the rainout layers using the column-max F. Simplest consistent
set: F1 or F3 feeding both the environment kernel and washout. Wang (2011)'s
refinement (washout only on the *newly* precipitating area
`max(0, F_{k+1} − F_k)`) is a one-line variant once the scan exists. Washout
has no updraft analogue — it acts on sub-cloud *environment* air; below cloud
base the in-cloud kernel is already off in updrafts via the `q_lclʲ > 0`
gate.

---

## 4. Suggested first-draft assembly (fewest new parts that is still honest)

1. **In-cloud kernel**: **K3 is decided** (mentor review 2026-08-25 — net-RHS
   drivers rejected; verbose formation-only C₁ with accretion *included*, cold
   riming arm off; rationale in immediate-plan doc §3a). K2 survives only as a
   smoke-test fallback. K3 also deletes the 258 K gate via process-based phase
   discrimination. `f_act ≡ 1` with the K4 seam in the signature.
2. **Updraft application** (§2.1): kernel on `sgsʲs` bins + ρaʲ-weighted
   grid-mean mirror, gated on `q_lclʲ > 0`, capped at ~1/dt.
3. **Environment application** (§2.2): Liu-literal F (F1) from the diagnosed
   environment Q; L, C₁, F₀ as ClimaParams keys with Liu defaults;
   `column_accumulate!(max)` for the downward F. Separate updraft/environment
   scavenging diagnostics from day one.
4. **Washout**: DSD-derived swept-volume Λ with per-bin constant `E_bin` (3d)
   now; size-resolved `E(D, a_wet)` as the follow-up; Feng (3a) kept as the
   cross-check. Applied over the column-max F.
5. **Re-evap**: R2 (diagnosed f, 0.5 constant) — it reuses the verbose cache
   and the column machinery from (3); or defer (R1) if draft 1 must be
   minimal.

New cache requirement shared by K3/R2: one `InstantaneousVerbose` BMT call
per subdomain in `set_microphysics_tendency_cache!` caching 3–5 scalars
(liquid formation, ice formation, rain evap) — either extend `MP1_NT` or add a
parallel small NamedTuple field. Scratch warning: that function already uses
`ᶜtemp_scalar` … `ᶜtemp_scalar_7` (`microphysics_cache.jl:913–922`).

Everything else (tendency dispatch verbs, per-bin loops, ClimaParams keys,
updraft mirror, no-op guards) follows the dry-deposition port patterns on this
branch verbatim.

---

## 5. Activation coupling (Abdul-Razzak–Ghan via CMAA)

Should rainout remove only aerosol activated into cloud droplets (nucleation
scavenging), rather than Liu's flat 100% of in-cloud aerosol?

**Physics: yes in principle; for sea-salt mass, numerically a no-op.**
κ-Köhler critical supersaturations for our bins (κ = 1.12): ~0.29% at the
smallest bin edge (dry r = 0.03 μm), < 0.05% for SSLT02–05. Marine S_max is
~0.1–0.4% (stratiform) and higher in updrafts, so SSLT02–05 always activate
and SSLT01 — which carries ~0.05% of emitted mass — partially activates only
in weak stratiform clouds. Liu's f = 1 is exact for the sea-salt mass budget.
The activated fraction becomes load-bearing for (a) aerosol *number* (a
2-moment scheme scavenging number at f = 1 would wipe out the CCN tail that
should survive) and (b) lower-κ species (dust, BC) sharing the framework.

**Current wiring facts:**
- Under 1M, CMAA is never called; droplet number is prescribed. The ARG
  wrapper `aerosol_activation_sources` (`microphysics_wrappers.jl:430–530`)
  exists only in the 2M cache path (`microphysics_cache.jl:1011/1071/1110` —
  feeding `dn_lcl_dt`), builds a bimodal seasalt+sulfate `Mode_κ` distribution
  from the *prescribed MERRA-2* aerosols, and is already called per updraft
  with the updraft's own w. Naive reuse under 2M would be inconsistent:
  activation sees the climatology while scavenging acts on prognostic bins.
- The prognostic-bin → `Mode_κ` bridge (per-bin lognormal fits +
  number-from-mass inversion) existed inert as `sea_salt_activation.jl` on
  `origin/zg/ssa-growth-drydep` (deliberately not ported); the per-bin
  log-moment fits are recomputable at cache build from the cached closed-form
  bin moments `p.tracers.sslt_bin_moments` on this branch.

**Interactions with the other two processes:**
- *Updraft scavenging*: ARG's natural home — S_max is an updraft-cloud-base
  quantity, and EDMF supplies wʲ. At updraft velocities all five bins
  activate, so ARG *derives* Liu's "in-cloud aerosol is in the condensed
  phase" rather than replacing it. The genuine effect is competition: giant
  sea-salt CCN suppress S_max, reducing activation of small sulfate/SSLT01 —
  an ACI effect that matters once CDNC is interactive, and the argument for
  one shared activation call feeding both CDNC and scavenging (number
  scavenged ≡ number activated; no double bookkeeping). Environment/stratiform
  activation would need an eddy-w scale from the prognostic SGS TKE (standard
  GCM practice).
- *Below-cloud washout*: no direct interaction. Washout is collection of
  interstitial aerosol below cloud base — activation state is irrelevant
  there, and the size dependence is already carried by ξ in the Feng
  assignment. (Second-order and out of scope for fixed mass bins:
  re-evaporation releases cloud-processed, coagulated — larger — particles.)

**Tiering:** draft 1 keeps f_act = 1 but takes a per-bin `f_act` argument in
the in-cloud term's signature; tier 2 wires ARG (one call per updraft on the
prognostic-bin distribution, TKE-based w for the environment) — triggered by
prognostic number or a second species, not by anything in the mass-only
scheme.
