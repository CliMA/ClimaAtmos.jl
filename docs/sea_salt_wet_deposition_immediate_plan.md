# Sea-Salt Wet Deposition — Immediate Implementation Plan

> **Scope note:** This is atmospheric/climate physics for the ClimaAtmos Earth
> system model. "Wet deposition," "scavenging," "activation," and "collection
> efficiency" are standard cloud-microphysics terms for how rain removes
> natural sea-spray salt particles from the air — the process that controls
> aerosol lifetime, cloud droplet formation, and aerosol-cloud radiative
> effects in climate simulations. Nothing in this document concerns biology,
> pathogens, or dispersal of any agent; the only "particles" are NaCl from
> ocean spray and the only "removal" is rainfall washing them into the ocean.

*A short, buildable spec for the first version, written to be diffed against the current ClimaAtmos architecture. The full rationale and staging live in `sea_salt_wet_deposition_implementation_plan.md`; the scheme survey in `wet_deposition_schemes_audit_sea_salt.md`. This doc is the "what to build now."*

---

## The one decision that shapes the code

Wet removal has **two levers, split by mode**, and the code must carry both because the model targets aerosol–cloud interactions (ACI):

- **Accumulation/Aitken mode (number, CCN, ACI) → in-cloud *nucleation* scavenging via the activated fraction.** Below-cloud impaction is weak here (Greenfield gap), so the accuracy-controlling sink is activation. Drive it from the **same `CMAA` activation the model already uses for CDNC**, not a flat `f_soluble = 1`.
- **Coarse/giant mode (mass, lifetime) → below-cloud washout with a coarse-mode `aP^b` coefficient**, plus near-complete in-cloud rainout.

Both are first-order sinks and fit in **one new explicit tendency** with the same structure as `sea_salt_settling_tendency!`. Number-conserving cloud-borne aerosol and re-evaporation are deferred (see §5), but the in-cloud term is written activation-based from day one so the accumulation mode is not structurally wrong.

---

## 1. Scope of the immediate version

Per bin `SSLTxx`, add a wet-removal tendency on `Y.c.ρSSLTxx`:

```
ρχₜ  −=  ρχ · (1 − exp(−(k_in + k_below)·Δt)) / Δt        # stable exponential sink
```

with

```
k_in    = F · f_act(bin) · c₁                             # in-cloud nucleation (rainout)
k_below = (1 − F) · Λ(bin, q_rai)                         # below-cloud washout (rain)
```

with the **process-level conversion rate** (see §3a — this replaced the earlier
`P_form / max(l_c, l_min)` form after mentor review):

```
c₁ = (S_acnv_lcl_rai + S_accr_lcl_rai [+ S_accr_lcl_sno_warm]) / max(q_lcl, ε)
```

- `F` = precipitating-area proxy = `p.precomputed.ᶜcloud_fraction`.
- `f_act(bin)` = **activated fraction** for that bin from `CMAA` (§3). Accumulation bins get `f_act < 1`; coarse bins → ~1. This is the key change from the earlier flat-solubility sketch.
- `S_*` = individual microphysics process rates (kg kg⁻¹ s⁻¹) from `BMT.bulk_microphysics_tendencies(InstantaneousVerbose(), …)` — **not** net `dq_*_dt` tendencies and **not** the RHS of `ρq_lcl` (§3a).
- `q_lcl` = cloud liquid on the same state the `S_*` were evaluated on. Because CM's sink limiter already enforces `ΣS·Δt ≤ q_lcl`, `c₁·Δt ≤ 1` holds by construction wherever the `ε` guard is inactive — no tunable `l_min` floor; keep a `min(c₁, 1/Δt)` cap as pure numerics.
- `Λ(bin, q_rai)` = below-cloud scavenging coefficient [s⁻¹]. Two interchangeable closures: Feng `a(bin)·P_r^{b(bin)}` with `P_r [mm h⁻¹] = 3600 · Y.c.ρq_rai · ᶜwᵣ`, or — preferred once built — the **DSD-derived swept-volume form** computed from the model's own Marshall–Palmer rain distribution (§3b), which needs no `(a, b)` fit at all.
- If Feng is used: `a(bin), b(bin)` are **mode-dependent**: coarse-mode values for SSLT04/05 (and upper SSLT03), accumulation-mode values for SSLT01–03. Small numbers for the accumulation bins (Greenfield gap) — do **not** reuse the coarse value across all bins.

Grid-mean, explicit, no new prognostic state. Snow term (`+ a_s·P_s^{b_s}`, with `f_act=0` for ice-precip formation since sea salt isn't an IN) is a one-line add once rain is verified.

---

## 2. Architecture mapping — new element ↔ existing analog

Everything here has a direct precedent in the current code, so the diff should read as "one more aerosol tendency."

| New element | Copy the pattern from | File |
|---|---|---|
| `sea_salt_wet_deposition_tendency!(Yₜ, Y, p, t)` | `sea_salt_settling_tendency!` (loop over `_aerosol_names`, per-bin `ρχ`) | new `src/parameterized_tendencies/aerosols/wet_deposition.jl` |
| Call site (explicit, after settling) | `sea_salt_settling_tendency!(Yₜ, Y, p, t)` at line 343 | `src/prognostic_equations/remaining_tendency.jl` |
| `include(...)` | the other `aerosols/` includes | `src/ClimaAtmos.jl` |
| No-op guard | `isempty(_aerosol_names(p.atmos.interactive_aerosols)) && return` | already used in `sea_salt.jl` |
| Stable per-step sink `-expm1(-k·dt)/dt` | (new; replaces settling's Courant cap — pure sink, unconditionally stable) | — |
| Gating (no new config flag) | active iff `seasalt isa PrognosticSeaSalt` && `microphysics_model isa NonEquilibriumMicrophysics1M`, by dispatch — matching how settling/dry-dep key off the species struct; default runs are bit-for-bit unchanged because they have no prognostic aerosols | `wet_deposition.jl` method table |
| Cached `ᶜprecip_formation_rate` | verbose liquid-formation sum (`S_acnv_lcl_rai + S_accr_lcl_rai + S_accr_lcl_sno_warm`, §3a) via the `_mp1m_source_term` extractor pattern | `src/cache/microphysics_cache.jl` (populate), `precomputed_quantities.jl` (allocate) |
| `Λ_bin(q_rai, ρ)` swept-volume helper | lift the closed form out of `CM1.accretion` with per-bin `E` (§3b); inputs `CM1.lambda_inverse` + `gamma_accr` | new `wet_deposition.jl`; params via `CAP.microphysics_1m_params` |
| Cached `ᶜsslt_f_act` (activated fraction/bin) | `ᶜsslt_r_wet` allocation (per-bin center NamedTuple) + `aerosol_activation_sources` | `precomputed_quantities.jl`, `microphysics_wrappers.jl` |
| `wetss` / per-bin deposition diagnostic | `emiss` / `emissslt0x` (`compute_*!` + `add_diagnostic_variable!`) | `src/diagnostics/tracer_diagnostics.jl` |
| Fields read: `ᶜcloud_fraction`, `ᶜwᵣ`, `ᶜwₛ`, `ρq_lcl`, `ρq_rai`, `ᶜT` | all already in `p.precomputed` / `Y.c` | `precomputed_quantities.jl` |
| Scratch: `p.scratch.ᶜtemp_scalar` for `k` | settling's `ᶜw` scratch usage (`sea_salt.jl:249`) | `temporary_quantities.jl` |

---

## 3. The genuinely new couplings

### 3a. In-cloud driver: process-level `S` terms — accretion included (mentor review, 2026-08-25)

Earlier drafts drove in-cloud removal from "RHS" quantities — the net tendency of
`ρq_lcl` / `dq_rai_dt` — with `l_c = specific(Y.c.ρq_lcl, ρ)` in a
`P_form / max(l_c, l_min)` rate. That driver is rejected, for three reasons:

1. **Net tendencies are the wrong sum.** `dq_lcl_dt` is dominated by
   condensation in an active cloud — a term that grows droplets *without adding
   aerosol* — and `dq_rai_dt` mixes in rain evaporation (an aerosol *source*,
   handled by re-evap later) and melt/freeze category bookkeeping. A removal
   rate built on either is sign- and magnitude-confused; `max(0, ·)` clamps
   hide, not fix, the misattribution.
2. **The production cache can't be decomposed.** `LinearizedAverage` returns a
   substepped effective average of the net tendencies; the process split is not
   recoverable from it. The driver must come from a dedicated
   `InstantaneousVerbose` call (extractor template:
   `_mp1m_source_term`, `src/diagnostics/microphysics_diagnostics.jl:24–32`).
3. **The `l_min` floor was a tuning wart.** With process-level `S` and the same
   `q_lcl` the microphysics limiter saw, `ΣS·Δt ≤ q_lcl` holds by construction,
   so the converted-fraction `c₁·Δt` is bounded by 1 wherever cloud water
   exists — only an `ε` division guard and a numerics-only `1/Δt` cap remain.

So the driver is the **liquid-formation process sum**:

```
Q = S_acnv_lcl_rai + S_accr_lcl_rai [+ S_accr_lcl_sno_warm]     ,   c₁ = Q / max(q_lcl, ε)
```

**Why accretion stays in the sum.** The suggestion to keep only autoconversion
("accretion isn't increasing aerosol deposition") has a true kernel but draws
the line in the wrong place:

- The removal law is mass balance for **droplet-borne** aerosol: aerosol
  dissolved in a cloud droplet goes wherever the droplet goes. Accretion
  converts cloud droplets into rain water exactly as autoconversion does — the
  collected droplet's salt ends up inside the falling raindrop. Accretion *is*
  nucleation-scavenging completion, not a side process.
- Quantitatively it is the **dominant** conversion path in warm marine clouds:
  once rain exists, the accretion:autoconversion ratio is typically 2–10
  (drizzling stratocumulus through mature warm rain — precisely sea salt's
  habitat). Autoconversion-only would undercount in-cloud removal severalfold
  where it matters most.
- Consistency check that catches the error immediately: with removal ∝
  `S_acnv` only, a raining cloud depletes `q_lcl` (via both processes) faster
  than it depletes droplet-borne aerosol, so the aerosol-per-cloud-water ratio
  grows without bound — activated salt "left behind" by the droplets it was
  dissolved in, which is unphysical.
- **Where the intuition is right:** accretion removes nothing from the
  *interstitial* (unactivated) aerosol. But neither does autoconversion — both
  act only on the activated fraction. That distinction is exactly what the
  `f_act(bin)` factor carries (§3c); it is not expressed by dropping an `S`
  term. (Raindrops do also impact-scavenge interstitial aerosol *inside* the
  cloud — a genuinely separate, small term, standardly neglected or folded
  into washout; we neglect it.)
- **Snow arms:** `S_accr_lcl_sno_warm` (collected liquid melts into rain)
  belongs in `Q`. The cold/riming arm `S_accr_lcl_sno_cold` physically removes
  droplet-borne aerosol into snow too, but mixed-phase clouds also evaporate
  droplets via WBF (releasing aerosol un-scavenged), so including riming
  without re-release biases removal high — and the ¹³⁷Cs lifetime constraint
  says models already remove aged aerosol too fast. Ship with the cold arm
  **off** (a toggle), revisit with re-evaporation. This process-split *is* the
  phase gate — no 258 K threshold needed.

Cache plumbing: one `InstantaneousVerbose` BMT call per subdomain in
`set_microphysics_tendency_cache!`, caching the 2–3 formation scalars (the same
verbose cache re-evap will need). Interim bring-up fallback only:
`max(ᶜmp_tendency.dq_rai_dt, 0) / max(q_lcl, ε)` — accept it for smoke tests,
not for results.

### 3b. Below-cloud Λ from the model's own rain DSD (swept volume)

*(Surveyed CloudMicrophysics v0.38.1 — the version pinned in
`.buildkite/Manifest-v1.11.toml` — on 2026-08-25.)*

The suggestion is correct: the machinery that "computes the area swept out
under the rain size distribution" already exists — it **is** the 1M accretion
kernel, and the scavenging coefficient factors out of it exactly.
`CM1.accretion` (`Microphysics1M.jl:491–514`) evaluates the closed form

```
accr_rate = q_clo · E · Λ ,
Λ = ∫ a(r)·v(r)·n(r) dr
  = n₀ · a₀χₐ · v₀χᵥ · λ⁻¹ · Γ(ae+ve+Δa+Δv+1) · (λ⁻¹/r₀)^(ae+ve+Δa+Δv)   [s⁻¹]
```

over the Marshall–Palmer rain distribution `n(r) = n₀·exp(−r/λ⁻¹)`
(**radius**-based, not diameter — a convention trap). With the ClimaParams
default rain parameters (`ae = 2, Δa = 0, χa = 1, a₀ = πr₀²`) the area factor
is exactly geometric, so `Λ = ∫ (π/4)D²·v(D)·n(D) dD` — the textbook
below-cloud scavenging coefficient with collection efficiency `E` pulled out
front. **Λ for a sea-salt bin is this same expression with `E` replaced by a
per-bin collection efficiency `E_bin`.** Everything needed is exported or one
call away:

- `λ⁻¹ = CM1.lambda_inverse(rain.pdf, rain.mass, q_rai, ρ)` — exported,
  `Microphysics1M.jl:126`; `n₀` is the constant `rain.pdf.n0` (1.6×10⁷ m⁻⁴).
  Or grab the whole bundle from `CM1.size_distr_parameters(mp, micro, thermo)`
  (`:375` → `n0_rai, λ_inv_rai, v0_rai`).
- `Γ(ae+ve+Δa+Δv+1)` is already precomputed as the `gamma_accr` field on
  `cmp.terminal_velocity.rain` (`parameters/TerminalVelocity.jl:60`).
- The structs come from the **same** `CAP.microphysics_1m_params(p.params)`
  that drives sedimentation (`terminal_velocity_utils.jl:56–64`) and
  accretion, so Λ, the rain rate `R = ρq_rai·ᶜwᵣ`, and the `S_accr_lcl_rai`
  in §3a are all moments of one distribution — internally consistent by
  construction, which the offline Feng fit can never be.

**Feng's power law becomes an output, not an input.** With power-law
`v(r) ∝ r^0.5`: `Λ ∝ (λ⁻¹)^{ae+ve+1} = (λ⁻¹)^{3.5}` while
`R ∝ (λ⁻¹)^{me+ve+1} = (λ⁻¹)^{4.5}`, so `Λ ∝ R^{3.5/4.5} = R^{0.78}` —
Feng (2007)'s fitted coarse-mode exponents are 0.79–0.80. The DSD-derived form
reproduces the fit's slope analytically and replaces the fitted prefactor
`a(bin)` with `E_bin` × model DSD, leaving `E_bin` as the *only* new physics.

**Tiering for `E_bin`:**

1. **First draft — per-bin constant `E_bin`** (mirrors the accretion kernel's
   own constant `E = 0.8`): evaluate the Slinn (1984) semi-empirical
   efficiency at the bin's wet radius (`ᶜsslt_ξ · p.tracers.sslt_settling_radii[bin]`) and a
   representative drop size; coarse bins → `E ≈ 1` (inertial impaction),
   SSLT01–02 → small (Greenfield gap). One ClimaParams vector `ssa_E_coll`
   per bin; closed form, one broadcast, GPU-trivial.
2. **Follow-up — `E(D, a_wet)` resolved**: CM has **no** Slinn/impaction code
   anywhere (verified — the only "collision efficiency" hits are the constant
   TOML scalars), so a size-resolved E is genuinely new. Two routes: numerical
   `Quadrature.integrate` with exponential-quantile bounds — the exact pattern
   P3 uses for its per-size collision kernel (`P3_processes.jl:152, 304`,
   `get_size_distribution_bounds`) — or `λ⁻¹ → Λ_bin` coefficients derived once at
   cache build from the cached closed-form bin moments
   `p.tracers.sslt_bin_moments` (no quadrature; RH-dependence via ξ like menu
   option 3c). If E is polynomial in D, it even
   stays closed-form: each `D^p` term shifts the Γ argument by `p`.
3. **Feng `aP^b` demoted to cross-check/calibration anchor**, not the scheme.

**Caveats found in the survey:**

- The calibrated `CliMA_1M.toml` overrides the rain *area* parameters
  (`Δa ≈ 3.0, χa ≈ 16.6`) — "area" is then no longer geometric `πr²`. Use the
  same `a(r)` the model's accretion actually runs (consistency beats textbook
  geometry), but record which TOML the run loads.
- Radius vs diameter: 1M is radius-space; Chen2022 call sites convert with
  `λ_inv_diameter = 2·λ_inv_radius`. All Slinn formulas are diameter-space.
- The accretion closed form assumes the power-law `Blk1MVelType` velocity —
  which is also what the 1M sedimentation `ᶜwᵣ` uses, so no inconsistency on
  this branch; revisit if rain velocity moves to Chen2022
  (`CO.Chen2022_exponential_pdf(a,b,c,λ_inv,k)`, `Common.jl:414`, provides the
  arbitrary-moment closed form for that case, `k = 2` for area-weighted).
- Snow analogue is the same expression with the snow PDF/velocity structs —
  defer with the snow term.

**Verification hooks:** (i) setting `E_bin = 0.8` must reproduce
`CM1.accretion(...)/q_clo` to rounding (factor ordering differs by a few
ulps); (ii) log–log slope of diagnosed
`Λ(R)` ≈ 0.78, compare against Feng coarse-mode 0.79.

**Derived `E_bin` defaults (2026-08-25, scratch calc against Feng marine):**
with ClimaParams-default rain DSD (`n₀ = 1.6e7`, power-law `v ∝ r^0.5`,
`C_drag = 0.55`), the analytic slope is exactly `(ae+ve+1)/(me+ve+1) = 3.5/4.5
= 0.778`, and matching `Λ_Feng(R)/Λ_DSD(R)` per bin at R = 0.5–2 mm/h gives
nearly R-independent efficiencies (varying < 15% over that decade — the
consistency that justifies a constant `E_bin`):

| Bin | Feng mode | `E_bin` |
|---|---|---|
| SSLT01, SSLT02 | 2 (accumulation; Greenfield gap) | `5.2e-4` |
| SSLT03, SSLT04 | 3 (coarse-a) | `0.42` |
| SSLT05 | 4 (coarse-b) | `0.94` |

These are the `ssa_collection_efficiency` ClimaParams defaults; they inherit
Feng's magnitude while the model's own DSD supplies the R-dependence.

### 3c. Activation-driven in-cloud scavenging

This is what makes the accumulation mode correct, and it reuses machinery that already exists.

- `aerosol_activation_sources` (`src/parameterized_tendencies/microphysics/microphysics_wrappers.jl:459-531`) already computes activated number for the SSLT modes via `CMAA.total_N_activated` / `CMAA.max_supersaturation`, feeding CDNC.
- **Add**: cache the per-bin activated *fraction* `f_act = N_act(bin) / N_total(bin)` into a new `p.precomputed.ᶜsslt_f_act` (allocate exactly like `ᶜsslt_r_wet`; populate in the same place activation is already evaluated). For sea salt this is ≈1 in the coarse bins and `<1`, supersaturation-dependent, in SSLT01–03 — which is precisely the accumulation-mode accuracy that flat `f_soluble` throws away.
- The in-cloud term then reads `f_act(bin)` instead of a constant. Consistency payoff: the aerosol number removed in-cloud matches the number activated into droplets that set CDNC — no double-book between CCN and scavenging.

The precip-formation driver feeding this term is fixed by §3a (process-level verbose `S` sum).

---

## 4. Parameters to add (`prescribed_aerosol_params`)

| Param | Meaning | Note |
|---|---|---|
| `ssa_E_coll[bin]` | per-bin collection efficiency in the DSD-derived Λ (§3b) | the one genuinely new physics knob; Slinn-evaluated defaults, coarse → ~1, accumulation → small (Greenfield gap) |
| `wash_a[bin]`, `wash_b[bin]` | Feng `aP^b` per bin, rain — **cross-check/fallback closure only** (§3b) | coarse-mode for SSLT04/05, accumulation-mode (small) for SSLT01–03 |
| `wash_a_snow[bin]`, `wash_b_snow[bin]` | same, snow | higher than rain; add with snow term |

No `l_min`: the process-level `c₁` (§3a) is bounded by the microphysics sink
limiter, so only an `ε` division guard and a `1/Δt` numerics cap remain.

Required addition to the (server-side, prepped) ClimaParams TOML — the same
uncommitted-keys situation as the other `ssa_*` parameters:

```toml
[ssa_collection_efficiency]
value = [5.2e-4, 5.2e-4, 0.42, 0.42, 0.94]
type = "float"
description = "Per-bin rain collection efficiency for sea salt below-cloud washout (Feng 2007 marine, anchored to the model rain DSD; see docs §3b)."
```
Activated fraction is *computed*, not a parameter. Note the intent: the small handful of coefficients (`ssa_E_coll`, or Feng `a` if that closure is active) are the calibration knobs against 210Pb / sea-salt obs — not free tuning to hide structural error.

---

## 5. Explicitly deferred (and why it's safe to defer *now*)

- **Cloud-borne aerosol tracer** (interstitial↔activated bookkeeping). The rigorous ACI endpoint; needs extra prognostic state. Deferring means in-cloud scavenging is a one-way sink for the activated fraction — fine for a first budget, but it cannot yet return activated aerosol on non-precipitating cloud evaporation.
- **Re-evaporation.** More important for accumulation number than coarse mass (returns CCN); pairs with the cloud-borne tracer. Deferred with it.
- **Prognostic sea-salt number / finer size resolution.** The current 5 mass-bins fix a lognormal per bin, so intra-bin activation selectivity is approximate. This — not the scavenging formula — is the real ceiling on ACI fidelity, and is the recommended next architectural question.
- **Convective/updraft (`ʲs`) in-cloud scavenging.** Grid-mean only for now, consistent with the existing settling TODO.
- **Wang (2011) cross-layer area partition** (`rainout→F_k`, `washout→max(0,F_{k+1}−F_k)`). One-cell vertical stencil; add after the local version verifies.

---

## 6. Verification (single-column first)

1. Flag off → bit-for-bit unchanged.
2. `PrecipitatingColumn` with imposed precip: tracer decays as `exp(−kΔt)` with expected `k`; check separately for an accumulation bin (in-cloud-dominated) and a coarse bin (washout-contributing).
3. Mass budget: column-integrated sink = `wetss` flux.
4. Activation consistency: number removed in-cloud == number activated to CDNC for the same step.
5. Lifetime sanity vs 210Pb-style tracer; coarse-mode lifetime should drop into the ~0.5–1 d range once washout is on.

---

## 7. Open decisions

1. **Activated-fraction source** — reuse `CMAA` per-bin fraction (recommended; consistent with CDNC) vs. a prescribed size-dependent `f_act` table (cheaper, less consistent).
2. ~~Precip-formation driver~~ — **decided** (mentor review, §3a): process-level
   verbose `S_acnv_lcl_rai + S_accr_lcl_rai + S_accr_lcl_sno_warm`, accretion
   included; net-`dq_*_dt` allowed only as a smoke-test fallback. Remaining
   sub-decision: cold riming arm on/off toggle default (recommend off until
   re-evaporation lands).
3. **Washout closure** — DSD-derived Λ with per-bin constant `E_bin` (§3b,
   recommended) vs. Feng `aP^b` table. If DSD-derived: which drop-size to
   evaluate Slinn's `E` at for the per-bin constant (DSD area-weighted mean
   drop diameter is the natural choice), and whether the run's TOML uses the
   geometric or `CliMA_1M`-calibrated rain area parameters.
4. **How soon to add the cloud-borne tracer** — gated on whether pristine-ocean ACI is an early deliverable.
