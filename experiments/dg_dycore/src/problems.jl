#=
Problem definitions — ENV-free port of the ClimaCore dg_problems.jl kwarg
structs. A problem is pure configuration; `DGSimulation(prob)` builds the
model/state/integrator from it (no include-time constants, no anonymous
modules — successive problems coexist in one session naturally).
=#

"""
    BaroclinicWaveFDDG(; kwargs...)

Flux-form FDDG baroclinic wave / balanced flow on the cubed sphere:
full (ρ, ρe, ρu⃗-Cartesian, ρw) system, KEP Kennedy–Gruber flux
differencing, explicit SSP-RK3 or HEVI (ARS343 + Newton, analytic
column Jacobian).

Keywords (defaults in parentheses):

  - `helem` (4), `npoly` (4), `zelem` (10), `zmax` (30e3): resolution
  - `stepper` (`:hevi`): `:hevi` or `:explicit`
  - `newton_max_iters` (2): Newton iterations per HEVI implicit stage.
    The per-stage transient the implicit solve must absorb grows with dt
    (vertical acoustic CFL ≫ 1 under zstretch), so larger dt needs more
    iterations — e.g. dt 60 at helem 16 needs 4 where dt 30 runs at 2
  - `dt` (60.0 for `:hevi`, 4.0 for `:explicit`) [s]
  - `t_end` (86400.0) [s]
  - `perturb` (true): JW perturbation on/off (off = balanced-flow test)
  - `κ₄` (`nothing` → SIPG-cap/10): absolute biharmonic coefficient
    [m⁴/s]; `0.0` for the pure-KEP configuration. Prefer `κ₄_frac`
    (fraction of the resolution/Δt-aware SIPG cap Δh³/((2npoly+1)²Δt),
    the CA-style ν₄ ∝ h³ scaling) — absolute values silently go unstable
    when helem or dt change.
  - `interface_flux` (`:rusanov`): `:rusanov` or `:roe` (wave-selective,
    Harten-floored — the pure-KEP interface). Paired automatically with the
    matching `volume_flux` family (KG / Ranocha / Waruszewski).
  - `pgf` (`:conservative`): momentum-pressure formulation —
    `:conservative` (full thermodynamic ``p`` in the momentum flux; historical
    behaviour) or `:conservative_pert` (perturbation ``pm = p − p_ref`` in the
    momentum flux plus the ``(ρ − ρ_ref) g`` buoyancy pair, energy keeps full
    ``p``). Differences the small ``p′`` ⇒ well-balanced over terrain.
  - `volume_flux` (`:kg`): horizontal two-point volume flux — `:kg`
    (Kennedy–Gruber), `:ranocha` (entropy-conservative), or `:waruszewski`
    (entropy-conservative and machine-precision well-balanced over terrain;
    carries gravity as the ``½ρ̂⟦φ⟧`` fluctuation). Pair `:waruszewski` /
    `:ranocha` with `pgf = :conservative_pert` for vertical well-balancing.
  - `wb_gravity` (`false`): DEPRECATED — the two-point geopotential
    fluctuation is now provided by `volume_flux = :waruszewski`. Setting this
    `true` errors (the standalone KG-gravity operator was removed upstream).
  - `zstretch` (`nothing`): `(dz_bottom, dz_top)` [m] stretched vertical grid
  - `sponge_τ` (1200.0) [s], `sponge_depth` (7.5e3) [m]: w-sponge peak
    rate 1/τ over the top `sponge_depth`; `τ = Inf` disables (canonical)
  - `sponge_uh` (false): additionally damp Cartesian horizontal momentum in
    the sponge (NOT part of the canonical baroclinic wave test — leave off)
  - `topography` (`:none`): `:none`, `:earth` (ETOPO2022 via
    SpaceVaryingInput, smoothed, LinearAdaption warp), or `:hughes2023`
    (analytic double mountain). CAUTION: for THIS (Cartesian flux-form)
    core the metric cross-terms are absent with `:rusanov`/`:roe` (flat-
    metric volume and interface); use `volume_flux = :waruszewski` with
    `pgf = :conservative_pert` for terrain well-balancing instead.
  - `topography_damping_factor` (5.0): damping factor for the `:earth`
    pre-smoothing diffusion (total smoothing ∝ log(factor)·Δh²)
  - `terrain_warp` (`:linear`): vertical coordinate adaption — `:linear`
    (`LinearAdaption`, terrain tilt decays linearly to zero at z = zmax) or
    `:sleve` (`SLEVEAdaption`, sinh-based exponential decay following Schär
    2002; coordinate surfaces flatten rapidly above the surface even with steep
    terrain, so the full terrain amplitude is retained at the surface while
    metric cross-terms in the mid-to-upper troposphere are greatly reduced —
    preferable over heavy horizontal smoothing when orographic detail matters).
  - `sleve_eta_h` (0.5): SLEVE — terrain warping applied only below
    `ηₕ × zmax`; grid is flat above. Ignored for `:linear`.
  - `sleve_s` (0.5): SLEVE decay rate (`s` in Schär 2002); larger = slower
    decay = terrain influence extends higher. Constraint: `s × zmax > max(z_sfc)`.
    Ignored for `:linear`.
  - `constants_mode` (`:parity`): `:parity` (ClimaCore-example literals) or
    `:clima_params` (Stage A2)
  - `dt_save` (21600.0) [s]: solution snapshot interval
  - `ndiag` (150): monitor print interval in steps
"""
Base.@kwdef struct BaroclinicWaveFDDG{FT <: AbstractFloat}
    helem::Int = 4
    npoly::Int = 4
    zelem::Int = 10
    zmax::FT = 30e3
    stepper::Symbol = :hevi
    newton_max_iters::Int = 2
    dt::FT = stepper == :hevi ? 60.0 : 4.0
    t_end::FT = 86400.0
    perturb::Bool = true
    κ₄::Union{Nothing, FT} = nothing
    κ₄_frac::Union{Nothing, FT} = nothing
    interface_flux::Symbol = :rusanov
    # Momentum-pressure formulation for the conservative (ρ,ρe,ρu⃗) flux family:
    #   :conservative      — full thermodynamic p in the momentum flux (pm = p);
    #                        the historical FDDG behaviour
    #   :conservative_pert — perturbation momentum pressure pm = p − p_ref plus
    #                        the (ρ−ρ_ref) g buoyancy pair; energy keeps the full
    #                        p. Differences the small p′ ⇒ well-balanced over
    #                        terrain (both horizontal flux and vertical ρw).
    pgf::Symbol = :conservative
    # Horizontal two-point VOLUME flux (all KEP):
    #   :kg          — Kennedy–Gruber (pressure-equilibrium preserving)
    #   :ranocha     — Ranocha (additionally entropy-conservative, Tadmor)
    #   :waruszewski — Waruszewski et al. (2022): entropy-conservative AND
    #                  machine-precision well-balanced over terrain — carries
    #                  gravity as the ½ρ̂⟦φ⟧ fluctuation, superseding `wb_gravity`.
    #                  Pair with `pgf = :conservative_pert` for vertical WB.
    volume_flux::Symbol = :kg
    wb_gravity::Bool = false
    # Terrain well-balancing backend for the horizontal-kernel GCL metric
    # defect (the spurious p·δ force from Σ_i ∂_ξi(Ja^i) ≠ 0 discretely over
    # warped grids):
    #   :none          — no correction (baseline; unstable over steep terrain)
    #   :metric_source — subtract p·δ, δ measured from the metric ALONE
    #                    (horizontal kernel on p≡1, u≡0, gravity off; IC-agnostic,
    #                    pressure-flux defect only)
    #   :gcl_curl      — discretely-consistent (curl-form) metric reconstruction
    #                    so δ ≡ 0 for all states (no source; also cancels the
    #                    advective defect)
    # δ is a pure GRID property — never calibrate it against an IC.
    wb_metric::Symbol = :none
    # Entropy correction (Chan et al. 2026, FCT): minimally-dissipative,
    # provably entropy-stable stabilization for the KEP volume flux — a
    # principled alternative to ∇⁴ hyperdiffusion (kept at κ₄ = 0). Blends the
    # high-order KEP flux with a low-order entropy-stable (central+LxF) flux by
    # the minimum element-wise amount that keeps the cell entropy residual ≥ 0.
    # Applies to the horizontal explicit advective flux (source terms/vertical
    # implicit acoustics are separate). false = no correction.
    entropy_correction::Bool = false
    zstretch::Union{Nothing, Tuple{FT, FT}} = nothing
    sponge_τ::FT = 1200.0
    sponge_depth::FT = 7.5e3
    sponge_uh::Bool = false
    topography::Symbol = :none
    topography_damping_factor::FT = 5.0
    terrain_warp::Symbol = :linear
    sleve_eta_h::FT = 0.5
    sleve_s::FT = 0.5
    constants_mode::Symbol = :parity
    # IC values: :setups (ClimaAtmos Setups.shallow_atmos_barowave_values,
    # verified formula-identical), :formulas (the examples' own JW06
    # expressions), or :rest (JW06 T/p profile, zero velocity — atmosphere
    # at rest; the right start for HS+terrain to avoid JW06 wind imbalance).
    ic_source::Symbol = :setups
    # Held–Suarez (1994) forcing via ClimaAtmos's own functions
    held_suarez::Bool = false
    # NetCDF diagnostics (ClimaAtmos-standard names incl. DG-consistent rv)
    # every diag_period seconds; nothing disables output
    output_dir::Union{Nothing, String} = nothing
    diag_period::FT = 86400.0
    dt_save::FT = 21600.0
    ndiag::Int = 150
end

BaroclinicWaveFDDG(; kwargs...) = BaroclinicWaveFDDG{Float64}(; kwargs...)

float_type(::BaroclinicWaveFDDG{FT}) where {FT} = FT

function validate(p::BaroclinicWaveFDDG)
    p.stepper in (:hevi, :explicit) ||
        error("stepper must be :hevi or :explicit")
    p.κ₄ !== nothing &&
        p.κ₄_frac !== nothing &&
        error("set κ₄ (absolute) or κ₄_frac (fraction of the SIPG cap), \
               not both")
    p.interface_flux in (:rusanov, :roe) ||
        error("interface_flux must be :rusanov or :roe")
    p.pgf in (:conservative, :conservative_pert) ||
        error("pgf must be :conservative or :conservative_pert")
    p.volume_flux in (:kg, :ranocha, :waruszewski) ||
        error("volume_flux must be :kg, :ranocha, or :waruszewski")
    p.wb_gravity &&
        error("wb_gravity is no longer backed by a ClimaCore operator on this \
               branch; use volume_flux = :waruszewski, which carries gravity as \
               the entropy-conservative ½ρ̂⟦φ⟧ fluctuation")
    p.ic_source in (:setups, :formulas, :rest) ||
        error("ic_source must be :setups, :formulas, or :rest")
    p.topography in (:none, :earth, :hughes2023) ||
        error("topography must be :none, :earth, or :hughes2023")
    p.terrain_warp in (:linear, :sleve) ||
        error("terrain_warp must be :linear or :sleve")
    p.wb_metric in (:none, :metric_source, :gcl_curl) ||
        error("wb_metric must be :none, :metric_source, or :gcl_curl")
    return p
end

"""
    BaroclinicWaveDG(; kwargs...)

Vector-invariant DG-FD baroclinic wave: state (ρ, ρe, uₕ::Covariant12, w).
Same resolution/time/IC/HS/output keywords as [`BaroclinicWaveFDDG`](@ref).
Unlike the FDDG core, terrain runs through the CG-shared covariant metric
machinery (full-metric K, contravariant ᶠu³, exact face normals under
LinearAdaption).

Additional keywords:

  - `momentum_adv` (`:vector_invariant`): `:vector_invariant` or
    `:fluctuation` (Route B; helem = 4 ONLY)
  - `face_set` (`:kg`): `:kg` (legacy KG + Rusanov + plain penalties),
    `:kep` (exact-KEP set; κ₄ = 0 / filter_Nc = 0 admissible), `:es`
    (:kep with entropy-variable ρe dissipation — entropy-dissipative AND
    still exactly KEP), or `:es2` (:es plus acoustic-selective Roe
    dissipation on ([[p′]], [[uₙ]]) with the contact wave kept central —
    damps the CFL-limiting acoustic modes; exact KEP relaxed to
    O(acoustic-jump²); p′ = p − p_ref via the composed hydrostatic
    reference, so terrain-following hydrostatic [[p]] jumps are not
    damped).
  - `terrain_u3` (`:full`): vertical transport velocity over terrain —
    `:full` (CT3(w) + CT3(uₕ), CG machinery) or `:wonly` (FDDG-style
    O(slope) approximation)
  - `ν_vert` (0.0) [m²/s]: peak vertical diffusivity on uₕ over the
    sponge sin² profile — breaking-wave momentum deposition aloft
    (sign-definite KE sink)
  - `ν_div_frac` (0.0): CAM-style divergence damping ν∇ₕ(∇ₕ·uₕ) as a
    fraction of the cap Δh²/((2npoly+1)²Δt) — scale-selective on
    divergent/acoustic transients; terrain-safe (balanced δ ≈ 0)
  - `κ₄` (`nothing` → SIPG-cap/10 for `:kg`, 0 for `:kep`) and
    `filter_Nc` (`nothing` → npoly for `:kg`, 0 for `:kep`): `:kg` NEEDS
    its stabilization. At zelem ≳ 20 also use `zstretch`.
  - `terrain_warp` (`:linear`), `sleve_eta_h`, `sleve_s`:
    see [`BaroclinicWaveFDDG`](@ref) — same semantics.
"""
Base.@kwdef struct BaroclinicWaveDG{FT <: AbstractFloat}
    helem::Int = 4
    npoly::Int = 4
    zelem::Int = 10
    zmax::FT = 30e3
    stepper::Symbol = :hevi
    newton_max_iters::Int = 2
    dt::FT = stepper == :hevi ? 60.0 : 4.0
    t_end::FT = 86400.0
    perturb::Bool = true
    momentum_adv::Symbol = :vector_invariant
    face_set::Symbol = :kg
    terrain_u3::Symbol = :full
    κ₄::Union{Nothing, FT} = nothing
    κ₄_frac::Union{Nothing, FT} = nothing
    ν_vert::FT = 0.0
    ν_div_frac::FT = 0.0
    filter_Nc::Union{Nothing, Int} = nothing   # nothing → npoly
    zstretch::Union{Nothing, Tuple{FT, FT}} = nothing
    sponge_τ::FT = 1200.0
    sponge_depth::FT = 7.5e3
    sponge_uh::Bool = false
    topography::Symbol = :none
    topography_damping_factor::FT = 5.0
    terrain_warp::Symbol = :linear
    sleve_eta_h::FT = 0.5
    sleve_s::FT = 0.5
    constants_mode::Symbol = :parity
    ic_source::Symbol = :setups
    held_suarez::Bool = false
    output_dir::Union{Nothing, String} = nothing
    diag_period::FT = 86400.0
    dt_save::FT = 21600.0
    ndiag::Int = 150
end

BaroclinicWaveDG(; kwargs...) = BaroclinicWaveDG{Float64}(; kwargs...)

float_type(::BaroclinicWaveDG{FT}) where {FT} = FT

function validate(p::BaroclinicWaveDG)
    p.stepper in (:hevi, :explicit) ||
        error("stepper must be :hevi or :explicit")
    p.momentum_adv in (:vector_invariant, :fluctuation) ||
        error("momentum_adv must be :vector_invariant or :fluctuation")
    p.κ₄ !== nothing &&
        p.κ₄_frac !== nothing &&
        error("set κ₄ (absolute) or κ₄_frac (fraction of the SIPG cap), \
               not both")
    p.face_set in (:kg, :kep, :es, :es2) ||
        error("face_set must be :kg, :kep, :es, or :es2")
    p.face_set in (:kep, :es, :es2) &&
        p.momentum_adv == :fluctuation &&
        error("face_set = :kep/:es pair with :vector_invariant only \
               (the fluctuation form is KE-compatible with the KG set)")
    p.terrain_u3 in (:wonly, :full) ||
        error("terrain_u3 must be :wonly or :full")
    p.ic_source in (:setups, :formulas, :rest) ||
        error("ic_source must be :setups, :formulas, or :rest")
    p.topography in (:none, :earth, :hughes2023) ||
        error("topography must be :none, :earth, or :hughes2023")
    p.terrain_warp in (:linear, :sleve) ||
        error("terrain_warp must be :linear or :sleve")
    return p
end

"""
    MountainWaveDG(; kwargs...)

Agnesi mountain wave on an x-periodic quasi-2D slab (one y element),
vector-invariant DG-FD core: isothermal base state `T₀` + uniform wind
`U₀` over `h(x) = h₀/(1 + (x/a)²)` (LinearAdaption warp). The CPU-cheap
terrain testbed for the face sets and the Exner PGF — with `U₀ = 0` any
motion is spurious, so max|w| directly measures the discrete PGF
residual; with `U₀ > 0` the classic linear wave (λ_z = 2πU₀/N).

Shares `face_set`/`terrain_u3`/κ₄/ν/sponge/stepper keywords
with [`BaroclinicWaveDG`](@ref). Plane-specific: `xmax` (domain length),
`h₀`, `a`, `U₀`, `T₀`. Defaults run 10 simulated hours in minutes.
"""
Base.@kwdef struct MountainWaveDG{FT <: AbstractFloat}
    helem::Int = 40
    npoly::Int = 4
    zelem::Int = 40
    zmax::FT = 30e3
    xmax::FT = 600e3
    h₀::FT = 250.0
    a::FT = 25e3
    U₀::FT = 20.0
    T₀::FT = 250.0
    stepper::Symbol = :hevi
    newton_max_iters::Int = 2
    # horizontal acoustic CFL ≈ 0.3: c_s·dt·(2npoly+1)/Δx_elem
    dt::FT = 0.3 * (xmax / helem) / (310 * (2 * npoly + 1))
    t_end::FT = 36000.0
    momentum_adv::Symbol = :vector_invariant
    face_set::Symbol = :kep
    terrain_u3::Symbol = :full
    κ₄::Union{Nothing, FT} = nothing
    κ₄_frac::Union{Nothing, FT} = nothing
    ν_vert::FT = 0.0
    ν_div_frac::FT = 0.0
    filter_Nc::Union{Nothing, Int} = nothing
    zstretch::Union{Nothing, Tuple{FT, FT}} = nothing
    sponge_τ::FT = 300.0
    sponge_depth::FT = 12e3
    sponge_uh::Bool = false
    # perturb=false additionally logs the drift metrics at the end (for
    # the U₀ = 0 well-balance test); the IC itself never has a perturbation
    perturb::Bool = true
    constants_mode::Symbol = :clima_params
    held_suarez::Bool = false
    output_dir::Union{Nothing, String} = nothing
    diag_period::FT = 3600.0
    dt_save::FT = 3600.0
    ndiag::Int = 120
end

MountainWaveDG(; kwargs...) = MountainWaveDG{Float64}(; kwargs...)

float_type(::MountainWaveDG{FT}) where {FT} = FT

function validate(p::MountainWaveDG)
    p.stepper in (:hevi, :explicit) ||
        error("stepper must be :hevi or :explicit")
    p.momentum_adv == :vector_invariant ||
        error("MountainWaveDG supports :vector_invariant only")
    p.face_set in (:kg, :kep, :es, :es2) ||
        error("face_set must be :kg, :kep, :es, or :es2")
    p.terrain_u3 in (:wonly, :full) ||
        error("terrain_u3 must be :wonly or :full")
    p.κ₄ !== nothing &&
        p.κ₄_frac !== nothing &&
        error("set κ₄ (absolute) or κ₄_frac (fraction of the SIPG cap), \
               not both")
    p.held_suarez && error("held_suarez does not apply to the plane case")
    return p
end

# vector-invariant-core problems (share tendency/Jacobian/face-set logic)
const VIProblem = Union{BaroclinicWaveDG, MountainWaveDG}
const DGProblem = Union{BaroclinicWaveFDDG, BaroclinicWaveDG, MountainWaveDG}

has_terrain(p::MountainWaveDG) = p.h₀ != 0
has_terrain(p) = p.topography != :none

# (T_min, T_sfc) of the Exner-PGF reference T_r = T_min + (T_sfc−T_min)Π⁷;
# isothermal cases match exactly with T_r ≡ T₀ (then Φ_r = −cₚT₀lnΠ = gz)
exner_reference(p::MountainWaveDG) = (p.T₀, p.T₀)
exner_reference(p) = (220.0, 290.0)
