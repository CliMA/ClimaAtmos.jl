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
  - `dt` (60.0 for `:hevi`, 4.0 for `:explicit`) [s]
  - `t_end` (86400.0) [s]
  - `perturb` (true): JW perturbation on/off (off = balanced-flow test)
  - `κ₄` (`nothing` → SIPG-cap/10): biharmonic coefficient; `0.0` for the
    pure-KEP configuration
  - `interface_flux` (`:rusanov`): `:rusanov` or `:roe` (wave-selective,
    Harten-floored — the pure-KEP interface)
  - `zstretch` (`nothing`): `(dz_bottom, dz_top)` [m] stretched vertical grid
  - `sponge_τ` (1200.0) [s], `sponge_depth` (7.5e3) [m]: w-sponge peak
    rate 1/τ over the top `sponge_depth`; `τ = Inf` disables (canonical)
  - `sponge_uh` (false): additionally damp Cartesian horizontal momentum in
    the sponge (NOT part of the canonical baroclinic wave test — leave off)
  - `topography` (`:none`): `:none`, `:earth` (ETOPO2022 via
    SpaceVaryingInput, smoothed, LinearAdaption warp), or `:hughes2023`
    (analytic double mountain). CAUTION: for THIS (Cartesian flux-form)
    core, terrain is geometry plumbing only — the metric cross-terms are
    absent, so valid for gentle smoothed slopes only.
  - `topography_damping_factor` (5.0): damping factor for the `:earth`
    pre-smoothing diffusion
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
    dt::FT = stepper == :hevi ? 60.0 : 4.0
    t_end::FT = 86400.0
    perturb::Bool = true
    κ₄::Union{Nothing, FT} = nothing
    interface_flux::Symbol = :rusanov
    zstretch::Union{Nothing, Tuple{FT, FT}} = nothing
    sponge_τ::FT = 1200.0
    sponge_depth::FT = 7.5e3
    sponge_uh::Bool = false
    topography::Symbol = :none
    topography_damping_factor::FT = 5.0
    constants_mode::Symbol = :parity
    # IC values: :setups (ClimaAtmos Setups.shallow_atmos_barowave_values,
    # verified formula-identical) or :formulas (the examples' own JW06
    # expressions). Both get the discrete-hydrostatic ρ correction.
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
    p.interface_flux in (:rusanov, :roe) ||
        error("interface_flux must be :rusanov or :roe")
    p.ic_source in (:setups, :formulas) ||
        error("ic_source must be :setups or :formulas")
    p.topography in (:none, :earth, :hughes2023) ||
        error("topography must be :none, :earth, or :hughes2023")
    # The tendency cutoff filter is a projection applied after the KEP
    # fluxes; the KE pairing is bilinear with the state outside the
    # projection, so filtering voids the KEP telescoping this scheme's
    # stability rests on (measured destabilization). It is therefore not
    # even exposed as an option here.
    return p
end

"""
    BaroclinicWaveDG(; kwargs...)

Vector-invariant DG-FD baroclinic wave: state (ρ, ρe, uₕ::Covariant12, w).
Same resolution/time/IC/HS/output keywords as [`BaroclinicWaveFDDG`](@ref).
Unlike the FDDG core, terrain runs through the CG-shared covariant metric
machinery (full-metric K, contravariant ᶠu³; remaining O(slope)
approximations: docs/vi_kep_face_terms.md §6/§8).

Additional keywords:

  - `momentum_adv` (`:vector_invariant`): `:vector_invariant` or
    `:fluctuation` (Route B; helem = 4 ONLY)
  - `face_set` (`:kg`): `:kg` (legacy KG fluxes + Rusanov + plain
    penalties) or `:kep` (the KEP-compatible set of
    docs/vi_kep_face_terms.md — the horizontal advective KE budget closes
    to roundoff, so κ₄ = 0 / filter_Nc = 0 become admissible)
  - `terrain_u3` (`:full`): vertical transport velocity over terrain —
    `:full` (CT3(w) + CT3(uₕ), CG machinery) or `:wonly` (FDDG-style
    O(slope) approximation)
  - `κ₄` (`nothing` → SIPG-cap/10 for `:kg`, 0 for `:kep`) and
    `filter_Nc` (`nothing` → npoly for `:kg`, 0 for `:kep`): `:kg` NEEDS
    its stabilization. At zelem ≳ 20 also use `zstretch`.
"""
Base.@kwdef struct BaroclinicWaveDG{FT <: AbstractFloat}
    helem::Int = 4
    npoly::Int = 4
    zelem::Int = 10
    zmax::FT = 30e3
    stepper::Symbol = :hevi
    dt::FT = stepper == :hevi ? 60.0 : 4.0
    t_end::FT = 86400.0
    perturb::Bool = true
    momentum_adv::Symbol = :vector_invariant
    face_set::Symbol = :kg
    terrain_u3::Symbol = :full
    κ₄::Union{Nothing, FT} = nothing
    filter_Nc::Union{Nothing, Int} = nothing   # nothing → npoly
    zstretch::Union{Nothing, Tuple{FT, FT}} = nothing
    sponge_τ::FT = 1200.0
    sponge_depth::FT = 7.5e3
    sponge_uh::Bool = false
    topography::Symbol = :none
    topography_damping_factor::FT = 5.0
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
    p.face_set in (:kg, :kep) || error("face_set must be :kg or :kep")
    p.face_set == :kep &&
        p.momentum_adv == :fluctuation &&
        error("face_set = :kep pairs with :vector_invariant only \
               (the fluctuation form is KE-compatible with the KG set)")
    p.terrain_u3 in (:wonly, :full) ||
        error("terrain_u3 must be :wonly or :full")
    p.ic_source in (:setups, :formulas) ||
        error("ic_source must be :setups or :formulas")
    p.topography in (:none, :earth, :hughes2023) ||
        error("topography must be :none, :earth, or :hughes2023")
    return p
end

const DGProblem = Union{BaroclinicWaveFDDG, BaroclinicWaveDG}
