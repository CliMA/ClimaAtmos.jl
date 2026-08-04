# Face-term requirements for kinetic-energy preservation in the vector-invariant DG-FD core

Status: derived 2026-07-30; implemented behind `face_set = :kep` on
`BaroclinicWaveDG` (fluxes in ClimaCore `numericalflux.jl`:
`vi_kep_scalars_flux`, `vi_kep_interface_scalars`,
`rho_weighted_jump_penalty_lift`). Verified by
`test/test_vi_kep_budget.jl` on a 2-hour spun-up baroclinic-wave state
(helem 4, jumps developed, KE ≈ 4×10²⁰ J, ledger-rate scale
P_ref ≈ 2.4×10¹⁷ W):

|                             | flat sphere                      | Hughes2023 double mountain |
|:--------------------------- |:-------------------------------- |:-------------------------- |
| `P_adv`, `:kep`             | −8.0×10² W (3×10⁻¹⁵ rel)         | −7.9×10² W (3×10⁻¹⁵ rel)   |
| `P_adv`, `:kg` (same state) | +2.1×10⁸ W (spurious production) | −1.5×10¹¹ W                |
| `P_pen`                     | −2.6×10¹¹ W ≤ 0                  | −2.6×10¹¹ W ≤ 0            |

The `:kep` closure is at roundoff and UNCHANGED by the terrain warp (§6);
the `:kg` violation grows by three orders of magnitude over terrain.
Configuration for the forced Rossby-wave-train run:
`configs/baroclinic_wave_double_mountain_kep.yml`.

## 1. Setting and claim

Vector-invariant DG-FD system (state `Y.c = (ρ, ρe, uₕ::Covariant12)`,
`Y.f = (w::Covariant3)`), horizontal terms discretized with nodal GLL
DG (no DSS), vertical with staggered FD. The kinetic-energy budget of the
**horizontal advective dynamics** — mass advection, the ∇K gradient, the
vertical-vorticity Lamb term, and all their face terms — can be made to
close **exactly** (to machine roundoff, on flat *and* terrain-warped grids)
by choosing a compatible set of volume/interface fluxes. The mechanism is
NOT the Kennedy–Gruber two-point telescoping of the flux-form core; it is

 1. **pointwise Lamb orthogonality** — `u·(ω×u) = 0` at every node, for any
    discrete ω (liftings included), any metric;
 2. **SBP duality** between the strong gradient in the momentum equation and
    the flux-differencing divergence in continuity;
 3. **matched face terms** — derived in §3; this is the only place where a
    *choice* exists, and the requirements are the deliverable of this note.

The vertical (staggered FD) couplings — `Ic(ᶠω¹² × ᶠu³)` against
`−ᶠω¹² × ᶠu¹²`, and the C2F/F2C interpolations inside K — close only to
interpolation truncation, exactly as in the CG ClimaAtmos core. All claims
of exactness below are for the horizontal ledger at fixed vertical level.

## 2. Discrete objects and notation

Per horizontal direction `k ∈ {1,2}` and GLL line of nodes, let
`D` be the differentiation matrix, `w` the quadrature weights,
`Q = diag(w)·D`, with the SBP property `Q + Qᵀ = B = diag(−1, 0, …, 0, +1)`.
Row sums of `Q` vanish (consistency); column sums equal `diag(B)`.

Per node `i`:

  - `(Ja)ᵢ` — the (non-unit) metric vector `J ∂ξᵏ/∂x` in the local orthonormal
    horizontal frame (what the kernels pass as `nvec`), quadrature weights
    absorbed by the kernel;
  - `ũᵢ = uvᵢ · (Ja)ᵢ` — the **nodal contravariant volume flux**;
  - `Gᵢ = ρᵢ ũᵢ` — the **nodal contravariant mass flux**;
  - `Kᵢ`, `Φᵢ`, `pᵢ` — nodal scalars.

Discrete total (horizontal) KE: `⟨ρK⟩ = Σᵢ (WJ)ᵢ ρᵢKᵢ`. Its advective
evolution collects `Σᵢ (WJ)ᵢ [Kᵢ ∂ₜρᵢ + ρᵢ (uᵏ ∂ₜu_k)ᵢ]`; note
`∂K/∂u_k = uᵏ` (contravariant), so in the orthonormal frame the momentum
contraction is `uvᵢ · ∂ₜuvᵢ` — frame-independent, metric enters only
through the nodal values above. **The entire ledger below is therefore
metric-transparent**: it holds identically on terrain-warped grids, because
the conditions are internal-consistency conditions between contractions
that all use the same surface geometry.

Kernel conventions (ClimaCore `numericalflux.jl`):

  - `add_flux_differencing_divergence!(F♯, dydt, y)` adds, at node `i`,
    the strong-form sum `−2wᵢ Σₘ Dᵢₘ F♯((Ja)ᵢ, (Ja)ₘ, yᵢ, yₘ)` plus the
    own-side consistent-flux boundary lifts (weak-equivalent form);
  - `add_numerical_flux_internal!(F*, dydt, y)` adds the antisymmetric SAT
    `∓ sWJ F*` so the combination realizes the standard `F* − F(y⁻)·n̂`;
  - `lifting_correction(fn, …)` adds the symmetric per-side lifting
    `sWJ·fn(n̂_own, own, other)` and divides by `WJ`.

## 3. The volume ledger

The two advective volume terms whose KE contributions must pair are the
continuity flux differencing (weight `Kᵢ`) and the momentum `−∂ₖK`
(weight `Gᵢ` after the `ρ uᵏ` contraction; the `ρ` from the KE weight and
the `1/ρ` absent from `∇K` combine to exactly `G`):

```
KE_vol = − Σᵢₘ Qᵢₘ Aᵢₘ,   Aᵢₘ = 2Kᵢ M♯(i,m) + Gᵢ Kₘ
```

where `M♯` is the symmetric two-point mass flux. Splitting `Q` into its
antisymmetric part and `B/2`, the volume production reduces to
element-boundary terms **iff the antisymmetric part of `A` is a difference
form**, `Aᵢₘ − Aₘᵢ = cᵢ − cₘ` for some nodal `c` (then
`Σ Qᵢₘ(cᵢ−cₘ) = −Σ B c`, a pure boundary term).

**The Kennedy–Gruber mass flux fails this test.** With
`M♯ = {ρ}{ũ}` (product of averages):

```
Aᵢₘ − Aₘᵢ = {K}(Gᵢ − Gₘ) + ½(Kᵢ − Kₘ)(ρᵢũₘ + ρₘũᵢ)
```

which is not a difference form — the VI pairing with a pointwise strong
`∇K` produces volume KE at O(jump²) with the KG flux. (The KG flux is
KEP for the *flux-form momentum* pairing, and for the *fluctuation-form*
momentum operator `kg_massflux_fluctuation`, whose docstring records the
compatible identity. It is not KEP for the vector-invariant pairing.)

**Requirement V (volume): the mass flux must be the average of nodal
contravariant mass fluxes**

```
M♯(i,m) = {G} = ½(ρᵢ ũᵢ + ρₘ ũₘ)          (average of products)
```

Then `Aᵢₘ − Aₘᵢ = KᵢGᵢ − KₘGₘ = cᵢ − cₘ` with `c = KG` — the discrete KE
flux, exactly the continuum `∇·(ρuK)`. Working out the constants
(`Aᵢᵢ = 3KᵢGᵢ`, boundary split `+½ΣBc − (3/2)ΣB(KG)`), the strong-form
volume production per element reduces to

```
KE_vol = − Σ_∂ B (K G)        (element-boundary values only).
```

`M♯ = {G}` is symmetric, consistent (`M♯(n,n,y,y) = ρ u·n`), jointly
linear in the metric vectors, and still averages *nodal contravariant
fluxes*, so free-stream preservation continues to avoid the discrete
metric identities (same argument as the KG flux's docstring).

The identical ledger with `K → Φ` covers the potential-energy exchange:
requirement V is the same, `c = ΦG`.

The matching **ρe advective flux** must ride on the same mass flux or
constant-`e` states are corrupted at the flux level:

```
F♯_ρe = {e}·M♯ + {p}{ũ}     (advective part on M♯; pressure-work part central)
```

(`{ρ}{e}{ũ}` with `M♯ = {G}` would give `F_ρe ≠ e F_ρ` for constant `e`.)

## 4. The face ledger

Consider one interior face, sides `L|R`, outward normal of `L` toward `R`;
`[[a]] = a_L − a_R`, `{a} = ½(a_L + a_R)`. Collect the three face-node
contributions (per unit `sWJ`):

  - **volume boundary residual** (from §3): `−[[K G]]`;
  - **continuity SAT** with interface mass flux `F*`:
    `K_L·(−(F* − G_L)) + K_R·(+(F* − G_R)) = −[[K]] F* + [[K G]]`;
  - **momentum ∇K lifting** with central trace `K* = {K}`
    (`central_gradient_lift`): each side receives `(K* − K_own) n̂_own`;
    contracting with `ρ uv` gives `+½ G_L [[K]] + ½ G_R [[K]] = [[K]]{G}`.

Total advective face production:

```
P_face = −[[KG]] + (−[[K]]F* + [[KG]]) + [[K]]{G} = [[K]] ({G} − F*)
```

**Requirement F1 (interface mass flux):** `F* = {G} = {ρ u·n̂}` — the
central average of the nodal *normal mass fluxes*, matching the volume
two-point flux. Then `P_face ≡ 0` at every face, every metric.

**Requirement F2 (no interface mass dissipation):** any dissipative
augmentation `F* = {G} − α[[ρ]]` produces `P_face = +α[[ρ]][[K]]` —
**sign-indefinite**. Exact KEP therefore requires a *central density flux*
(the same conclusion as Gassner's KEP-DG for Euler). Acoustic/thermal
face dissipation must act through `ρe` (a `λ/2 [[ρe]]` Rusanov penalty is
KE-inert: K is diagnostic) and through the velocity penalties below.

**Requirement F3 (gradient traces):** `K* = {K}` and `p* = {p}` central
liftings (already the sandbox default, `central_gradient_lift`). `Φ` is
continuous across faces, so its lifting contributes `[[Φ]](…) = 0`
identically — including on terrain-warped grids, where `z` remains
continuous.

**Requirement F4 (KE-dissipative velocity penalties):** the plain
`jump_penalty_lift` (`δu_own = (λ̄/2)(u_other − u_own)`) yields face KE

```
P_pen = −(λ̄/2) [[u]]·[[ρu]] = −(λ̄/2) ( {ρ} |[[u]]|² + [[ρ]] {u}·[[u]] )
```

— dissipative only to `O([[ρ]])`. The exactly sign-definite form is the
**ρ-weighted penalty**

```
δu_own = (λ̄/2) ({ρ}/ρ_own) (u_other − u_own)
  ⇒  P_pen = −(λ̄/2) {ρ} |[[u]]|²  ≤ 0   exactly.
```

**Lamb term (no requirement):** `−(f + ω³) × uₕ` contracted with `ρ uv`
vanishes pointwise for *any* nodal `ω³`, including the
`central_curl3_lift` face corrections — the vorticity liftings never enter
the KE ledger. (They govern enstrophy, not energy.)

## 5. Summary of the compatible set (`face_set = :kep`)

| slot                 | legacy (`:kg`)           | KEP requirement                               |
|:-------------------- |:------------------------ |:--------------------------------------------- |
| volume mass flux     | `{ρ}{ũ}` (KG)            | `{ρũ}` (nodal-flux average)                   |
| volume ρe flux       | `({ρ}{e}+{p}){ũ}`        | `{e}{ρũ} + {p}{ũ}`                            |
| interface mass flux  | KG + Rusanov `−λ/2[[ρ]]` | central `{ρ u·n̂}`, **no ρ penalty**          |
| interface ρe flux    | KG + Rusanov             | same central + Rusanov on `[[ρe]]` (KE-inert) |
| K, p gradient traces | central                  | central (unchanged)                           |
| velocity penalties   | `(λ̄/2)[[u]]`            | `(λ̄/2)({ρ}/ρ_own)[[u]]` (exact KE sink)      |
| ω liftings           | central                  | unchanged (KE-inert)                          |

With this set the semi-discrete horizontal advective KE production is zero
to roundoff; all remaining face KE tendencies are the sign-definite
velocity penalties. Pressure–internal-energy exchange (`−ũ·∂p` vs the
`{p}{ũ}` work flux) is energy-*consistent* at faces by F3 but is a
physical exchange, not an advective production — total energy remains
exactly conserved through the flux-form `ρe` equation regardless.

## 6. Topography

Every condition above is a *matching* condition between contractions that
share one surface geometry (`sWJ`, `n̂`) and one set of nodal metric
vectors. Nothing requires the metric to be unwarped, and nothing requires
discrete metric identities. Consequently **the KEP property holds exactly
on terrain-following (hypsography-warped) grids**.

Moreover, the kernels' horizontal (UVAxis) projection of face normals is
**lossless under LinearAdaption**, not an approximation: the horizontal
coordinates are unwarped (`∂ξʰ/∂z ≡ 0`), so `∇ξ¹`, `∇ξ²` — the exact
normals of the (physically tilted) ξʰ-faces — are exactly horizontal, and
`w` carries no flux through them in the contravariant sense. The
horizontal DG divergence along coordinate surfaces is therefore exactly
curvilinear; ALL tilt transport belongs to `u³ = u·∇ξ³`, which the core
carries at the equation level via the CG machinery
(`ᶠu³ = CT3(C123(w)) + ᶠwinterp(ρJ, CT3(uₕ))`,
`K = ½‖C123(uₕ) + C123(Ic(w))‖²`). The ledger — being metric-transparent —
accommodates these terms without change: they alter `G` and `K` *values*,
not the compatibility conditions.

## 7. What remains truncation-level

  - The staggered cross pair `uₕ·Ic(ᶠω¹²×ᶠu³)` vs `w·(ᶠω¹²×ᶠu¹²)` — the
    continuum triple product `ω·(u×u) = 0` is split across C2F/F2C
    interpolations.
  - Vertical (FD) advective transport KE consistency — as in CG ClimaAtmos.
  - The `VanLeer` vertical upwinding — intentionally dissipative.

The KE-budget diagnostic in the test therefore separates: (h) horizontal
advective production — must be roundoff with `:kep`; (v) vertical/staggered
residual — truncation, comparable between face sets; (d) penalties + sponge
— must be ≤ 0.

## 8. Findings from the double-mountain runs (what KEP does NOT buy)

Bringing up the Hughes2023 case at helem 8 separated FOUR distinct
properties; each was measured, and each crash mode had a different owner.
The KEP ledger closure itself was exact throughout (§6 — it never was the
problem).

 1. **Covariant increments from physical vectors** (a latent flat-grid bug
    in the original VI core): `C3(WVector(x), lg)` amplifies by
    O(∂z/∂ξʰ) ~ 10³ on warped grids (the covariant representation of a
    vertical vector has huge 1,2-components). The λ-scaled w-jump penalty
    went from 0.0012 to 1.60 m/s² through this conversion — instant crash.
    Fixed by penalizing the prognostic covariant dof directly. Related:
    explicit `Geometry.transform` to CT12 axes throws on warped grids
    (nonzero dropped components) — use `project`; the vector-type
    CONSTRUCTORS already project.
 2. **Kinematic lower BC**: from `w = 0` the JW06 wind violates `u³ = 0`
    at the 2 km peaks (u ≈ 5-8 m/s there); the zero-flux operator BCs dump
    the mismatch on the first cell (0.19 s⁻¹ relative ρe tendency). Fixed
    by the terrain-consistent SURFACE w (CA's surface-velocity constraint,
    applied statically; `initial_conditions.jl`). Adapting w in the
    INTERIOR (u³ ≡ 0 everywhere) is wrong — it breaks the staggered
    ∇K/Lamb shear cancellation at O(u·w/Δz) (measured 1.7 m/s² dw).
 3. **Cross-discretization GCL**: with the surface fix, the CT3(uₕ)
    vertical flux divergence is ~85-90% compensated by the horizontal
    along-surface flux (−27.5 vs +31.0 at the peak; net residual ≈ 3×10⁻⁵
    s⁻¹ relative — benign truncation, NOT the crash driver). The
    conservative-metric construction (face metric built to satisfy the
    discrete identity against the GLL derivative) would close it exactly —
    follow-up.
 4. **Terrain-following hyperdiffusion**: ANY field with vertical structure
    has an O(Δz_warp) signature along the warped coordinate surfaces
    (h_tot: g·Δz + cp·Γ·Δz; velocity: shear·Δz). Naive along-surface κ₄
    biharmonics turn this physical structure into spurious dipoles at the
    mountains (measured ~100 J/m³/s for h_tot at κ₄ = 1e16; the (u,v)
    version crashed even faster than κ₄ = 0). Over terrain the sandbox now
    restricts κ₄'s energy leg to flat grids; terrain-aware dissipation
    (perturbation-from-warped-reference or constant-z) is the follow-up.

Net: the pure-KEP double-mountain run integrates the forced adjustment
cleanly and, at this deliberately marginal resolution (2.3° ridges on
2.8° nodes), eventually meets mountain-wave breaking aloft with no
dissipation channel — the expected "KEP ≠ entropy stability" boundary.
The KE-budget verification (the point of the test case) is unconditional.

## 9. Entropy-dissipative extension (`face_set = :es`)

The `:kep` set controls the advective KE channel exactly but leaves the
thermodynamic (pressure-work/entropy) channel uncontrolled — the observed
failure mode over terrain (§8 item 2 timeline; blow-up as p → 0 aloft).
The `:es` set closes it with a single-slot construction that keeps the
KE ledger exact:

  - **Centrals unchanged** (`{ρũ}` mass, matching ρe flux): mass stays
    central, so §§3–4 apply verbatim — `:es` is exactly KEP.

  - **ρe interface dissipation in the entropy variable**: with
    `S = −ρs/(γ−1)` and `v = ∂S/∂ρe|_ρ = −ρ/p`,
    
        F*_ρe = F_central − (λ/2) w̄ (v⁺ − v⁻),   w̄ = p̄²/((γ−1)ρ̄) > 0.
    
    Because only the ρe slot is dissipated and ρ is untouched, the exact
    chain rule at fixed ρ gives the total interface entropy production
    `−(λ/2) w̄ [[v]]² ≤ 0` — provable with no matrix theory, no coupling to
    K or Φ. Near constant states it reduces to Rusanov on the
    internal-energy part of `[[ρe]]` (the plain `[[ρe]]` Rusanov it
    replaces is entropy-indefinite: its production `∝ [[v]][[ρe]]` has no
    sign). KE-inert since K is diagnostic.
  - **Velocity penalties (ρ-weighted) are already entropy-consistent**:
    they convert KE → (nothing) at fixed ρe, i.e. raise e_int, which is a
    physical-entropy increase.
  - **Status/limits**: this is entropy-dissipative interface dissipation
    for the scalar subsystem, not a full entropy-stability proof for the
    staggered VI system (the central advective terms produce entropy at
    O(jump³), as all KEP-but-not-EC centrals do; an entropy-conservative
    volume flux à la Chandrashekar/Ranocha would break the exact VI-KEP
    volume ledger of §3 — the two exactness properties compete at the
    volume level and `:es` chooses KEP there).

Verified by the third testset of `test/test_vi_kep_budget.jl`: on a
spun-up jumpy state the isolated `:es` dissipation gives `P_S < 0`, mass
fluxes are bit-identical to central, and the KE ledger stays at the
`:kep` roundoff level. Implementation:
`Operators.VIESInterfaceScalars(γ−1)` (callable, ClimaCore
`numericalflux.jl`).
