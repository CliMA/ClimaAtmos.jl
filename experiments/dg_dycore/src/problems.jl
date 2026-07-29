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
  - `sponge_τ` (1200.0) [s]: w-sponge peak rate 1/τ over the top 7.5 km;
    `Inf` disables the sponge entirely (fully canonical setup)
  - `sponge_uh` (false): additionally damp Cartesian horizontal momentum in
    the sponge (NOT part of the canonical baroclinic wave test — leave off)
  - `topography` (`:none`): `:none` (flat) or `:earth` (ETOPO2022 60arcsec
    ClimaArtifacts orography regridded onto the GLL nodes via
    SpaceVaryingInput, diffusion-smoothed, LinearAdaption terrain-following
    warp). CAUTION: the horizontal DG fluxes are evaluated along the warped
    coordinate surfaces; the terrain metric cross-terms (uₕ·∇z_sfc transport
    through sloped surfaces, true-horizontal pressure gradient) are not yet
    included, so this is geometry plumbing — valid only for gentle smoothed
    slopes, pending a curvilinear (contravariant-flux) extension of the core.
  - `topography_damping_factor` (5.0): smallest-resolved-scale damping factor
    for the pre-smoothing diffusion (ClimaAtmos recipe)
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
    p.topography in (:none, :earth) ||
        error("topography must be :none or :earth")
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
Same resolution/time/IC/HS/output keywords as [`BaroclinicWaveFDDG`](@ref),
plus:

  - `momentum_adv` (`:vector_invariant`): `:vector_invariant` or
    `:fluctuation` (Route B; validated at helem = 4 ONLY — violently
    unstable at helem ≥ 16)
  - `κ₄` (`nothing` → SIPG-cap/10) and `filter_Nc` (`nothing` → npoly):
    this formulation NEEDS its stabilization (its momentum advection has no
    KEP property); at zelem ≳ 20 also use `zstretch = (300.0, 3000.0)`.
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
    κ₄::Union{Nothing, FT} = nothing
    filter_Nc::Union{Nothing, Int} = nothing   # nothing → npoly
    zstretch::Union{Nothing, Tuple{FT, FT}} = nothing
    sponge_τ::FT = 1200.0
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
    p.ic_source in (:setups, :formulas) ||
        error("ic_source must be :setups or :formulas")
    p.topography in (:none, :earth) ||
        error("topography must be :none or :earth")
    return p
end

const DGProblem = Union{BaroclinicWaveFDDG, BaroclinicWaveDG}
