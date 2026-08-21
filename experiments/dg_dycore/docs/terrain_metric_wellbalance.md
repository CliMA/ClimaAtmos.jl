# FDDG terrain well-balance: the metric-identity defect

**Status:** open — diagnosis complete, genuine fix (discrete metric-identity
closure) not yet implemented. All exploratory changes are uncommitted.
**Branch:** `as/hi-res` (ClimaAtmos) + `as/bickley-jet` (ClimaCore, `dev`'d).

## Problem

The FDDG core (`interface_flux: curvilinear_roe_wb`, `wb_gravity: true`) over
`topography: earth` with `ic_source: rest` (JW06 `T`/`p`, zero wind) develops
NaNs from rest (~10800 s in the original Held–Suarez config). With `kappa4 = 0`
(a design constraint — the KEP scheme must be stable without hyperdiffusion),
there is no dissipative sink for the spurious force, so it grows.

## Diagnostic tool

`experiments/dg_dycore/rest_residual_probe.jl` builds the model + rest state,
evaluates the RHS once, and splits the momentum residual into radial (handled by
the vertical ρw pair) vs tangential (along-surface). A well-balanced scheme gives
machine-zero everywhere at rest.

Run: `julia --project=experiments/dg_dycore experiments/dg_dycore/rest_residual_probe.jl <config>`

## Diagnosis (measured)

At the JW06 rest state over earth/SLEVE terrain (helem16/npoly4/zelem20):

| quantity | value | meaning |
|---|---|---|
| `dρ`, `dρw`, momentum **radial** | ~1e-17 | machine zero ✓ |
| momentum **tangential** | **0.167** | the spurious force |
| `dρe` | 0.0126 | physical Held–Suarez thermal relaxation (NOT a defect) |
| flat (no terrain) JW06 tangential | 0.00168 | thermo residual + real meridional PGF (no metric) |

So the radial vertical PGF+buoyancy pair is discretely balanced; the imbalance is
entirely the **tangential (along-surface)** momentum, and it is a **terrain**
effect (flat is ~1000× smaller).

**Root cause:** the horizontal KG flux-differencing (`_fd_volume_node_total` in
ClimaCore `Operators/numericalflux.jl`) is strictly 2D (directions ξ¹,ξ²). Over
terrain the horizontal metric vectors `Ja^k` carry a W (radial) cross-term, and
the discrete horizontal metric identity does not close:
`Σ_{k=1,2} D_k(Ja^k) = −∂_ξ3(Ja³) ≠ 0`. The wb_gravity fluctuation removes the
pairwise *thermodynamic* imbalance but leaves this **geometric remainder**, which
the KG pairing multiplies by pressure. The cancelling term lives in the ξ³
direction — the experiment's *staggered FD* (vdivf2c…), a different operator
family — so it is absent from the 2D kernel by HEVI design.

Crucially the remainder is **not** a clean `p · (fixed grid field)`: it is a
nonlinear, state-dependent product of the KG two-point pairing with the metric.
This is why every source/reference correction fell short (below).

## Approaches tried and why they fail

| approach | JW06-terrain tangential | verdict |
|---|---|---|
| baseline | 0.167 | — |
| reference-state (JW06) tendency subtraction | ~0 | **kills the jet** — cancels the real meridional PGF (it rides the same rest tendency). Reverted (was committed as `1d5d73cf4`). |
| `wb_metric: :metric_source` (subtract `p·δ`, δ from isothermal rest) | 0.146 (13%) | insufficient — isothermal δ ≠ JW06 defect pattern (state-dependent). |
| non-isothermal ref (jw_temp@lat45) tendency subtraction | 0.133 (20%) | insufficient — no horizontally-uniform reference matches the lat-varying defect; the only reference that does (JW06) kills the jet. |
| naive `:gcl_curl` (subtract `p·div(ê_c)` via `hwdiv`+`vdivf2c3`) | 7.57 (45× **worse**) | the weak-form operator divergence is ~100× off the KG two-point defect and mis-patterned. |
| non-isothermal two-point gravity (ClimaCore kernel, `−Δp_ref/2`) | 0.16559 | **negligible** — `logmean(ρ)ΔΦ ≈ −Δp` to O(cell size) for any profile, so the old gravity was already near non-isothermal-exact. |

**Key correction:** the "isothermal 0.085 vs JW06 0.167" gap is mostly the
**pressure-magnitude** difference (defect ~ `p·hdiv`; isothermal-250 has lower `p`
than warmer JW06), *not* a thermodynamic pairing error. The residual is
**dominantly the metric defect**, and the non-isothermal-exact gravity — while a
genuine correctness refinement — does not address it.

## The genuine fix (next step): discrete metric-identity closure

Make the discrete metric identity hold across the 2D SEM (horizontal) and
staggered FD (vertical) operators, so `Σ_i D_i(Ja^i) ≡ 0` and the remainder
vanishes for all states — no source term, no reference, no jet suppression.

Concretely (Kopriva 2006 conservative curl-form, adapted to the mixed SEM/FD
tensor-product grid; the two operator families commute on the tensor grid, so
`div(curl) = 0` can hold discretely):

1. **ClimaCore `Geometry`/`Grids`:** compute the warped-extruded metric terms
   `Ja^i = −x̂_n · ∇_ξ × (X_l ∇_ξ X_m)` using the SEM horizontal derivative for
   ξ¹,ξ² and the FD vertical derivative for ξ³, replacing the analytic metrics for
   hypsography spaces.
2. **Experiment (`flux_form.jl`):** add the vertical `Ja³` metric flux for
   Cartesian momentum (currently absent) using the FD operator, so the 2D kernel's
   horizontal metric divergence and the vertical FD divergence cancel.
3. **Verify** with `rest_residual_probe.jl`: isothermal-terrain and JW06-terrain
   tangential → machine zero, **and** confirm a run still spins up the jet
   (`ū` O(10 m/s), the check the reference-subtraction fix failed).

This is a focused numerical-methods task (the naive shortcuts above are proven not
to work); it warrants a dedicated session, not more quick probes.

## Fundamental obstruction (why the Kopriva closure does NOT resolve it)

Attempting the Kopriva curl-form closure surfaced a fundamental obstruction, not
an implementation hurdle. Kopriva's metric-identity closure and Waruszewski's
well-balanced exactness both require a SINGLE summation-by-parts (SBP) operator in
all three directions, so that with that one operator: (a) `div(curl)=0` discretely,
AND (b) the two-point flux telescopes to `p_n·D(Ja)` (needed for well-balance with
hydrostatic/varying pressure). This HEVI scheme uses SEM collocation horizontally
and staggered FD vertically (vertical must stay implicit for the acoustic CFL):

- horizontal KG defect telescopes to `p_a·D_horiz(Ja)`;
- any vertical FD cancelling term carries a product-rule error `{Ja³}·D_vert(p)`,
  so it cannot cancel `p_a·D_horiz(Ja)` for varying pressure (the 22× overshoot);
- curl-form fixes the metric identity for a CONSTANT field (free-stream) — but the
  KG pairing already preserves free-stream; it does NOT fix the pressure-weighted
  well-balance, because the telescoping mismatch remains.

Deeper: the KG kernel's effective metric divergence `D_horiz(Ja)` is intrinsic to
its two-point structure — it manifests ONLY against a varying pressure field (a
constant field gives zero by free-stream), so it cannot be measured cleanly or
separated from the thermodynamics, and it is not reproduced by any standard
operator (`hwdiv` was 100× off). This single fact explains EVERY failed attempt
(22×, 45×, 13%, 20%, negligible gravity). Exact terrain well-balance for this
Cartesian-momentum HEVI SEM/FD scheme is therefore obstructed by the operator
split.

**Empirical confirmation (2026-08-20), two independent methods.** Computed the raw
metric divergence `D(ê_c·Ja)` (a) via `hwdiv`+`vdivf2c3` and (b) by running the
actual KG curvilinear kernel on a unit-metric state (`p≡1, ρ≡0, u≡0`, momentum
flux → `{ê_c·Ja}`). Both give **~7–8e-5**. But the *effective* rest defect is
`δ ≈ 0.085/p ≈ 8.5e-7` — **~100× smaller**. So the KG pairing suppresses the raw
divergence ~100× (its free-stream mechanism), and the quantity to cancel is a
100×-suppressed, state-entangled residual — NOT the metric divergence. Subtracting
`p·D(Ja)` injects ~42× the defect (0.167 → ~7). This is the definitive proof: no
artificial-state or operator computation recovers the effective defect, so the
curl-form / metric-divergence cancellation cannot be built for this scheme.

Routes that actually work:
1. Full 3D flux-differencing (Waruszewski) — vertical becomes an SBP
   flux-differencing direction too; correct, but ABANDONS HEVI (vertical explicit),
   forfeiting the acoustic-CFL advantage. Dycore-level redesign.
2. Accept an O(slope) terrain imbalance; start from a balanced/spun-up state (not
   JW06-at-rest) so the imbalance rides a small pressure and does not blow up
   without kappa4.
3. Use the production ClimaAtmos dycore (vector-invariant, covariant metric),
   which is already well-balanced over terrain. The sandbox's Cartesian-momentum
   choice is what creates the obstruction.

## Current code state (all uncommitted)

- **ClimaCore** (`~/Research/Codes/ClimaCore.jl`, `as/bickley-jet`): `dev`'d into
  the experiment; `wb_gravity_cartesian_increment_curvilinear` uses the
  non-isothermal-exact `−(p_ref_b − p_ref_a)/2` form (requires `y.p_ref`).
  Revert: `Pkg.free ClimaCore` + `git checkout src/Operators/numericalflux.jl`.
- **ClimaAtmos** (`experiments/dg_dycore`): `wb_metric` config knob
  (`:none | :metric_source | :gcl_curl`); `:metric_source` works but is
  insufficient; `:gcl_curl` errors (needs the ClimaCore change above).
  `:none` is the bitwise baseline.

## Pragmatic note

`ic_source: rest` (JW06 `T` at zero wind, a 60 K equator–pole gradient with no
balancing jet) is an *unbalanced* stress-test IC. For climate-relevant runs a
balanced or gently-spun-up start sidesteps most of this; the metric-identity
defect is the genuine terrain concern and is what the fix above targets.
