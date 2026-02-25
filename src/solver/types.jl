import ClimaCore.Quadratures.GaussQuadrature as GQ
import StaticArrays as SA
import Thermodynamics as TD
import Dates

import ClimaParams as CP
import ClimaUtilities.ClimaArtifacts: @clima_artifact
import LazyArtifacts

abstract type AbstractMicrophysicsModel end

struct DryModel <: AbstractMicrophysicsModel end
struct EquilibriumMicrophysics0M <: AbstractMicrophysicsModel end
struct NonEquilibriumMicrophysics1M <: AbstractMicrophysicsModel end
struct NonEquilibriumMicrophysics2M <: AbstractMicrophysicsModel end
struct NonEquilibriumMicrophysics2MP3 <: AbstractMicrophysicsModel end

const NonEquilibriumMicrophysics = Union{
    NonEquilibriumMicrophysics1M,
    NonEquilibriumMicrophysics2M,
    NonEquilibriumMicrophysics2MP3,
}
const MoistMicrophysics = Union{
    EquilibriumMicrophysics0M,
    NonEquilibriumMicrophysics1M,
    NonEquilibriumMicrophysics2M,
    NonEquilibriumMicrophysics2MP3,
}

"""
    TracerNonnegativityMethod

Family of methods for enforcing tracer nonnegativity.

There are four methods for enforcing tracer nonnegativity:
- `TracerNonnegativityElementConstraint{qtot}`: Enforce nonnegativity by instantaneously redistributing
    tracer mass within an element (i.e. horizontally)
- `TracerNonnegativityVaporConstraint{qtot}`: Enforce nonnegativity by instantaneously redistributing
    tracer mass between vapor (`q_vap = q_tot - q_cond`) and each tracer
- `TracerNonnegativityVaporTendency`: Enforce nonnegativity by applying a tendency to each tracer,
    exchanging tracer mass between vapor (`q_vap`) and each tracer over time
- `TracerNonnegativityVerticalWaterBorrowing`: Enforce nonnegativity using VerticalMassBorrowingLimiter,
    which redistributes tracer mass vertically. Note: `qtot` parameter is not applicable to this method.

`qtot` is a boolean that is `true` if q_tot is among the constrained tracers, and `false` otherwise.

# Constructor

    TracerNonnegativityMethod(method::String; include_qtot = false)

Create a microphysics tracer nonnegativity constraint.

Depending on the microphysics model, the constrained tracers include:
- `ρq_liq`, `ρq_ice`, `ρq_rai`, `ρq_sno`,
- If `include_qtot` is true, `q_tot` is also among the constrained tracers.

# Arguments:
- `method`: Can be
    - "elementwise_constraint": constructs `TracerNonnegativityElementConstraint{include_qtot}()`
    - "vapor_constraint": constructs `TracerNonnegativityVaporConstraint{include_qtot}()`
    - "vapor_tendency": constructs `TracerNonnegativityVaporTendency()`

# Keyword arguments:
- `include_qtot`: (default: `false`) Boolean that is `true` if q_tot is among the constrained tracers.
"""
abstract type TracerNonnegativityMethod end
abstract type TracerNonnegativityConstraint{qtot} <: TracerNonnegativityMethod end
struct TracerNonnegativityElementConstraint{qtot} <: TracerNonnegativityConstraint{qtot} end
struct TracerNonnegativityVaporConstraint{qtot} <: TracerNonnegativityConstraint{qtot} end
struct TracerNonnegativityVaporTendency <: TracerNonnegativityMethod end
struct TracerNonnegativityVerticalWaterBorrowing <: TracerNonnegativityConstraint{false} end

function TracerNonnegativityMethod(method::String; include_qtot::Bool = false)
    if method == "elementwise_constraint"
        return TracerNonnegativityElementConstraint{include_qtot}()
    elseif method == "vapor_constraint"
        return TracerNonnegativityVaporConstraint{include_qtot}()
    elseif method == "vapor_tendency"
        include_qtot &&
            error("TracerNonnegativityVaporTendency does not support `include_qtot = true`")
        return TracerNonnegativityVaporTendency()
    elseif method == "vertical_water_borrowing"
        include_qtot &&
            error("TracerNonnegativityVerticalWaterBorrowing does not support \
                `include_qtot = true`")
        return TracerNonnegativityVerticalWaterBorrowing()
    else
        error("Invalid tracer nonnegativity method: $method")
    end
end

"""

    AbstractSGSamplingType

How sub-grid scale diagnostic should be sampled in computing cloud fraction.
"""
abstract type AbstractSGSamplingType end

"""
    SGSMean

Use the mean value.
"""
struct SGSMean <: AbstractSGSamplingType end



"""
    AbstractCloudModel

How to compute the cloud fraction.
"""
abstract type AbstractCloudModel end

"""
    GridScaleCloud

Compute the cloud fraction based on grid mean conditions.
"""
struct GridScaleCloud <: AbstractCloudModel end

"""
    QuadratureCloud

Compute the cloud fraction using Sommeria-Deardorff moment matching.
"""
struct QuadratureCloud <: AbstractCloudModel end


"""
    MLCloud

Compute the cloud fraction using a machine learning model.
"""
struct MLCloud{M} <: AbstractCloudModel
    model::M
end

function MLCloud_constructor(model)
    static_model = Adapt.adapt_structure(SA.SArray, model)
    return MLCloud{typeof(static_model)}(static_model)
end

abstract type AbstractSST end
struct ZonallySymmetricSST <: AbstractSST end
struct RCEMIPIISST <: AbstractSST end
struct ExternalTVColumnSST <: AbstractSST end

abstract type AbstractInsolation end
struct IdealizedInsolation <: AbstractInsolation end
struct TimeVaryingInsolation <: AbstractInsolation
    # TODO: Remove when we can easily go from time to date
    start_date::Dates.DateTime
end
struct RCEMIPIIInsolation <: AbstractInsolation end
struct GCMDrivenInsolation <: AbstractInsolation end
struct ExternalTVInsolation <: AbstractInsolation end

"""
    AbstractCloudInRadiation

Describe how cloud properties should be set in radiation.

This is only relevant for RRTMGP.
"""
abstract type AbstractCloudInRadiation end

"""
    InteractiveCloudInRadiation

Use cloud properties computed in the model
"""
struct InteractiveCloudInRadiation <: AbstractCloudInRadiation end

"""
    PrescribedCloudInRadiation

Use monthly-average cloud properties from ERA5.
"""
struct PrescribedCloudInRadiation <: AbstractCloudInRadiation end

abstract type AbstractSurfaceTemperature end
struct PrescribedSST <: AbstractSurfaceTemperature end
@kwdef struct SlabOceanSST{FT} <: AbstractSurfaceTemperature
    # optional slab ocean parameters:
    depth_ocean::FT = 40 # ocean mixed layer depth [m]
    ρ_ocean::FT = 1020 # ocean density [kg / m³]
    cp_ocean::FT = 4184 # ocean heat capacity [J/(kg * K)]
    q_flux::Bool = false # use Q-flux (parameterization of horizontal ocean mixing of energy)
    Q₀::FT = -20 # Q-flux maximum mplitude [W/m²]
    ϕ₀::FT = 16 # Q-flux meridional scale [deg]
end


### -------------------- ###
### Hyperdiffusion model ###
### -------------------- ###

@kwdef struct Hyperdiffusion{FT}
    ν₄_vorticity_coeff::FT
    divergence_damping_factor::FT
    prandtl_number::FT
end

"""
    cam_se_hyperdiffusion(FT)

Create a Hyperdiffusion with CAM_SE preset coefficients.

These coefficients match hyperviscosity coefficients from:
(Lauritzen et al. (2017))[https://doi.org/10.1029/2017MS001257]
for equations A18 and A19, scaled by `(1.1e5 / (sqrt(4 * pi / 6) * 6.371e6 / (3*30)) )^3 ≈ 1.238`
"""
cam_se_hyperdiffusion(::Type{FT}) where {FT} =
    Hyperdiffusion{FT}(;
        ν₄_vorticity_coeff = 0.150 * 1.238,
        divergence_damping_factor = 5,
        prandtl_number = 0.2,
    )

### ------------------------------------ ###
### Prescribed vertical diffusion models ###
### ------------------------------------ ###

abstract type AbstractVerticalDiffusion end
@kwdef struct VerticalDiffusion{DM, FT} <: AbstractVerticalDiffusion
    C_E::FT
end
VerticalDiffusion{FT}(; disable_momentum_vertical_diffusion, C_E) where {FT} =
    VerticalDiffusion{disable_momentum_vertical_diffusion, FT}(; C_E)

disable_momentum_vertical_diffusion(::VerticalDiffusion{DM}) where {DM} = DM
@kwdef struct DecayWithHeightDiffusion{DM, FT} <: AbstractVerticalDiffusion
    H::FT
    D₀::FT
end
DecayWithHeightDiffusion{FT}(; disable_momentum_vertical_diffusion, H, D₀) where {FT} =
    DecayWithHeightDiffusion{disable_momentum_vertical_diffusion, FT}(; H, D₀)

disable_momentum_vertical_diffusion(::DecayWithHeightDiffusion{DM}) where {DM} = DM
disable_momentum_vertical_diffusion(::Nothing) = false


### --------------------- ###
### Eddy Viscosity Models ###
### --------------------- ###

abstract type EddyViscosityModel end

"""
    SmagorinskyLilly{AXES}

Smagorinsky-Lilly eddy viscosity model.

`AXES` is a symbol indicating along which axes the model is applied. It can be
- `:UVW` (all axes)
- `:UV` (horizontal axes)
- `:W` (vertical axis)
- `:UV_W` (horizontal and vertical axes treated separately).

# Examples
Construct a model instance by passing the selected axes as a keyword argument:
```julia
smagorinsky_lilly = SmagorinskyLilly(; axes = :UV_W)
```
"""
struct SmagorinskyLilly{AXES} <: EddyViscosityModel end

function SmagorinskyLilly(; axes::Symbol)
    @assert axes in (:UVW, :UV, :W, :UV_W) "axes must be one of :UVW, :UV, :W, or :UV_W, got :$axes"
    return SmagorinskyLilly{axes}()
end

"""
    is_smagorinsky_UVW_coupled(model)

Check if the Smagorinsky model is coupled along all axes.
"""
is_smagorinsky_UVW_coupled(::SmagorinskyLilly{AXES}) where {AXES} = AXES == :UVW
is_smagorinsky_UVW_coupled(::Nothing) = false

"""
    is_smagorinsky_vertical(model)

Check if the Smagorinsky model is applied along the vertical axis.

See also [`is_smagorinsky_horizontal`](@ref).
"""
is_smagorinsky_vertical(::SmagorinskyLilly{AXES}) where {AXES} =
    AXES == :UVW || AXES == :W || AXES == :UV_W
is_smagorinsky_vertical(::Nothing) = false

"""
    is_smagorinsky_horizontal(model)

Check if the Smagorinsky model is applied along the horizontal axes.

See also [`is_smagorinsky_vertical`](@ref).
"""
is_smagorinsky_horizontal(::SmagorinskyLilly{AXES}) where {AXES} =
    AXES == :UVW || AXES == :UV || AXES == :UV_W
is_smagorinsky_horizontal(::Nothing) = false

@kwdef struct AnisotropicMinimumDissipation{FT} <: EddyViscosityModel
    c_amd::FT
end

@kwdef struct ConstantHorizontalDiffusion{FT} <: EddyViscosityModel
    D::FT
end

### ------------- ###
### Sponge models ###
### ------------- ###

abstract type SpongeModel end
Base.broadcastable(x::SpongeModel) = tuple(x)

"""
    ViscousSponge{FT} <: SpongeModel

Viscous sponge model; dampen variables in proportion to the value of their Laplacian

Whenever `z > zd`, the viscous sponge model applies the tendency

 ```math
 \frac{∂χ}{∂t} = - β ⋅ ∇⋅(∇χ),   z > zd
 ```

 where `β = κ₂ ⋅ ζ` and `χ ∈ {uₕ, u₃, ρe_tot, GS_TRACERS}`;
 the grid-scale tracers `GS_TRACERS` depend on the microphysical model,
 but may include e.g. `ρq_tot`, `ρq_liq`, `ρq_ice`, ...
 If the `PrognosticEDMFX` scheme is used, the model is additionally applied to `χ ∈ {u₃ʲ}`.
 `κ₂` is a damping coefficient, and `ζ` is the damping function

 ```math
 ζ(z) = sin^2(π(z-zd)/(zmax-zd)/2)
 ```

 with `zd` the lower damping height and `zmax` the domain top height.

# Examples
```julia
# Apply damping above 20km with κ₂ = 10^6 m²/s²
sponge = ViscousSponge(Float32; zd = 20_000, κ₂ = 1e6)
```
"""
@kwdef struct ViscousSponge{FT} <: SpongeModel
    "Lower damping height, in meters"
    zd::FT
    "Damping coefficient, in m²/s²"
    κ₂::FT
end

ViscousSponge(params) = ViscousSponge(;
    zd = params.zd_viscous,
    κ₂ = params.kappa_2_sponge,
)

"""
    RayleighSponge{FT} <: SpongeModel

Rayleigh sponge model; dampen variables in proportion to their value

Whenever `z > zd`, the Rayleigh sponge model applies the tendency

 ```math
 \frac{∂χ}{∂t} = - β ⋅ χ,   z > zd
 ```

 where `β = α_χ ⋅ ζ` and `χ ∈ {uₕ, u₃}`;
 If `ρtke` is a prognostic variable, it is also damped;
 If the `PrognosticEDMFX` scheme is used, the model is additionally applied to
 `χ ∈ {u₃ʲ, mseʲ, q_totʲ}`, and
 `χ ∈ {q_liqʲ, q_raiʲ, q_iceʲ, q_snoʲ}` (depending on the microphysical model).
 `α_χ` is a damping coefficient for each variable, and `ζ` is the damping function

 ```math
 ζ(z) = sin^2(π(z-zd)/(zmax-zd)/2)
 ```

 with `zd` the lower damping height and `zmax` the domain top height.

 Separate damping coefficients are used:
 - `α_uₕ`: horizontal velocity, `uₕ`;
 - `α_w`: vertical velocity, `u₃`, `u₃ʲ`;
 - `α_sgs_tracer`: subgrid-scale tracer variables, `ρtke`, `mseʲ`, `q_totʲ`,
    `q_liqʲ`, `q_raiʲ`, `q_iceʲ`, `q_snoʲ`.

 By default, damping is only applied to vertical velocity, with:
 - `α_uₕ = 0`
 - `α_w = 1`
 - `α_sgs_tracer = 0`

# Examples
```julia
# Apply damping to vertical velocity, above 20km
sponge = RayleighSponge(Float32; zd = 20_000)
```
"""
@kwdef struct RayleighSponge{FT} <: SpongeModel
    "Lower damping height, in meters"
    zd::FT
    "Damping coefficient for horizontal velocity, by default 0 (no damping)"
    α_uₕ::FT = 0
    "Damping coefficient for vertical velocity, by default 1 (full damping)"
    α_w::FT = 1
    "Damping coefficient for subgrid-scale tracer variables, by default 0 (no damping)"
    α_sgs_tracer::FT = 0
end

RayleighSponge(params) = RayleighSponge(;
    zd = params.zd_rayleigh,
    α_uₕ = params.alpha_rayleigh_uh,
    α_w = params.alpha_rayleigh_w,
    α_sgs_tracer = params.alpha_rayleigh_sgs_tracer,
)


### ------------------- ###
### Gravity wave models ###
### ------------------- ###

abstract type AbstractGravityWave end
Base.@kwdef struct NonOrographicGravityWave{FT} <: AbstractGravityWave
    source_pressure::FT
    damp_pressure::FT
    source_height::FT
    Bw::FT
    Bn::FT
    dc::FT
    cmax::FT
    c0::FT
    nk::FT
    cw::FT
    cw_tropics::FT
    cn::FT
    Bt_0::FT
    Bt_n::FT
    Bt_s::FT
    Bt_eq::FT
    ϕ0_n::FT
    ϕ0_s::FT
    dϕ_n::FT
    dϕ_s::FT
end

abstract type OrographicGravityWave <: AbstractGravityWave end

Base.@kwdef struct LinearOrographicGravityWave{S} <: OrographicGravityWave
    topo_info::S = Val(:linear)
end

Base.@kwdef struct FullOrographicGravityWave{FT, S, T} <: OrographicGravityWave
    γ::FT
    ϵ::FT
    β::FT
    h_frac::FT
    ρscale::FT
    L0::FT
    a0::FT
    a1::FT
    Fr_crit::FT
    topo_info::S
    topography::T
end

abstract type AbstractForcing end
struct HeldSuarezForcing end
struct Subsidence{T} <: AbstractForcing
    prof::T
end
# TODO: is this a forcing?
struct LargeScaleAdvection{PT, PQ}
    prof_dTdt::PT # Set large-scale cooling
    prof_dqtdt::PQ # Set large-scale drying
end
# maybe need to <: AbstractForcing
struct GCMForcing{FT}
    external_forcing_file::String
    cfsite_number::String
end

"""
    ExternalDrivenTVForcing

Forcing specified by external forcing file.
"""
struct ExternalDrivenTVForcing{FT}
    external_forcing_file::String
end

struct ISDACForcing end

struct SCMCoriolis{U, V, FT}
    prof_ug::U
    prof_vg::V
    coriolis_param::FT
end

abstract type AbstractEnvBuoyGradClosure end
struct BuoyGradMean <: AbstractEnvBuoyGradClosure end

Base.broadcastable(x::BuoyGradMean) = tuple(x)

"""
    EnvBuoyGradVars

Variables used in the environmental buoyancy gradient computation.
"""
@kwdef struct EnvBuoyGradVars{FT}
    T::FT
    ρ::FT
    q_tot::FT
    q_liq::FT
    q_ice::FT
    cf::FT
    ∂qt∂z::FT
    ∂θli∂z::FT
end

function EnvBuoyGradVars(
    T,
    ρ,
    q_tot,
    q_liq,
    q_ice,
    cf,
    ∂qt∂z_∂θli∂z,
)
    (; ∂qt∂z, ∂θli∂z) = ∂qt∂z_∂θli∂z
    return EnvBuoyGradVars(T, ρ, q_tot, q_liq, q_ice, cf, ∂qt∂z, ∂θli∂z)
end

Base.eltype(::EnvBuoyGradVars{FT}) where {FT} = FT
Base.broadcastable(x::EnvBuoyGradVars) = tuple(x)

struct MixingLength{FT}
    master::FT
    wall::FT
    tke::FT
    buoy::FT
    l_grid::FT
end

function MixingLength(master, wall, tke, buoy, l_grid)
    return MixingLength(promote(master, wall, tke, buoy, l_grid)...)
end

abstract type AbstractEDMF end

"""
    Eddy-Diffusivity Only "EDMF"

This is EDMF without mass-flux.

This allows running simulations with TKE-based vertical diffusion.
"""
struct EDOnlyEDMFX <: AbstractEDMF end

struct PrognosticEDMFX{N, TKE, FT} <: AbstractEDMF
    a_half::FT # WARNING: this should never be used outside of `specific`
end
PrognosticEDMFX{N, TKE}(a_half::FT) where {N, TKE, FT} =
    PrognosticEDMFX{N, TKE, FT}(a_half)

"""
    PrognosticEDMFX(; n_updrafts = 1, prognostic_tke = false, area_fraction)

Create a PrognosticEDMFX model with the specified number of updrafts, TKE configuration, and area fraction.

# Arguments
- `n_updrafts::Int`: Number of updraft subdomains
- `prognostic_tke::Bool`: Whether to use prognostic TKE (true) or diagnostic TKE (false)
- `area_fraction`: "Small" area fraction threshold, is the `a_half` argument in `sgs_weight_function`
    - Note: Float type is inferred from this value
"""
function PrognosticEDMFX(;
    n_updrafts = 1,
    prognostic_tke = false,
    area_fraction::FT,
) where {FT}
    return PrognosticEDMFX{n_updrafts, prognostic_tke, FT}(area_fraction)
end

struct DiagnosticEDMFX{N, TKE, FT} <: AbstractEDMF
    a_half::FT # WARNING: this should never be used outside of `specific`
end
DiagnosticEDMFX{N, TKE}(area_fraction::FT) where {N, TKE, FT} =
    DiagnosticEDMFX{N, TKE, FT}(area_fraction)

"""
    DiagnosticEDMFX(; n_updrafts = 1, prognostic_tke = false, area_fraction)

Create a DiagnosticEDMFX model with the specified number of updrafts, TKE configuration, and area fraction.

# Arguments
- `n_updrafts::Int`: Number of updraft subdomains
- `prognostic_tke::Bool`: Whether to use prognostic TKE (true) or diagnostic TKE (false)
- `area_fraction`: Area fraction at half levels (float type is inferred from this value)
"""
function DiagnosticEDMFX(;
    n_updrafts = 1,
    prognostic_tke = false,
    area_fraction::FT,
) where {FT}
    return DiagnosticEDMFX{n_updrafts, prognostic_tke, FT}(area_fraction)
end

n_mass_flux_subdomains(::PrognosticEDMFX{N}) where {N} = N
n_mass_flux_subdomains(::DiagnosticEDMFX{N}) where {N} = N
n_mass_flux_subdomains(::EDOnlyEDMFX) = 0
n_mass_flux_subdomains(::Any) = 0

n_prognostic_mass_flux_subdomains(::PrognosticEDMFX{N}) where {N} = N
n_prognostic_mass_flux_subdomains(::Any) = 0

use_prognostic_tke(::EDOnlyEDMFX) = true
use_prognostic_tke(::PrognosticEDMFX{N, TKE}) where {N, TKE} = TKE
use_prognostic_tke(::DiagnosticEDMFX{N, TKE}) where {N, TKE} = TKE
use_prognostic_tke(::Any) = false

abstract type AbstractEntrainmentModel end
struct NoEntrainment <: AbstractEntrainmentModel end
struct PiGroupsEntrainment <: AbstractEntrainmentModel end
struct InvZEntrainment <: AbstractEntrainmentModel end

abstract type AbstractDetrainmentModel end

struct NoDetrainment <: AbstractDetrainmentModel end
struct PiGroupsDetrainment <: AbstractDetrainmentModel end
struct BuoyancyVelocityDetrainment <: AbstractDetrainmentModel end
struct SmoothAreaDetrainment <: AbstractDetrainmentModel end

abstract type AbstractTendencyModel end
struct UseAllTendency <: AbstractTendencyModel end
struct NoGridScaleTendency <: AbstractTendencyModel end
struct NoSubgridScaleTendency <: AbstractTendencyModel end

# Define broadcasting for types
Base.broadcastable(x::AbstractMicrophysicsModel) = tuple(x)
Base.broadcastable(x::AbstractForcing) = tuple(x)
Base.broadcastable(x::EDOnlyEDMFX) = tuple(x)
Base.broadcastable(x::PrognosticEDMFX) = tuple(x)
Base.broadcastable(x::DiagnosticEDMFX) = tuple(x)
Base.broadcastable(x::AbstractEntrainmentModel) = tuple(x)
Base.broadcastable(x::AbstractDetrainmentModel) = tuple(x)
Base.broadcastable(x::AbstractSGSamplingType) = tuple(x)
Base.broadcastable(x::AbstractTendencyModel) = tuple(x)

@kwdef struct RadiationDYCOMS{FT}
    "Large-scale divergence"
    divergence::FT = 3.75e-6
    alpha_z::FT = 1.0
    kappa::FT = 85.0
    F0::FT = 70.0
    F1::FT = 22.0
end

@kwdef struct RadiationISDAC{FT}
    F₀::FT = 72  # W/m²
    F₁::FT = 15  # W/m²
    κ::FT = 170  # m²/kg
end

import AtmosphericProfilesLibrary as APL

struct RadiationTRMM_LBA{R}
    rad_profile::R
    function RadiationTRMM_LBA(::Type{FT}) where {FT}
        rad_profile = APL.TRMM_LBA_radiation(FT)
        return new{typeof(rad_profile)}(rad_profile)
    end
end

abstract type PrescribedFlow{FT} end

struct ShipwayHill2012VelocityProfile{FT} <: PrescribedFlow{FT} end
function (::ShipwayHill2012VelocityProfile{FT})(z, t) where {FT}
    w1 = FT(1.5)
    t1 = FT(600)
    return t < t1 ? w1 * sinpi(FT(t) / t1) : FT(0)
end

"""
    get_ρu₃qₜ_surface(flow::PrescribedFlow{FT}, thermo_params, t) where {FT}

Computes the vertical transport `ρwqₜ` at the surface due to prescribed flow.

# Arguments
- `flow`: The prescribed flow model, see [`PrescribedFlow`](@ref).
- `thermo_params`: The thermodynamic parameters, needed to compute surface air density.
- `t`: The current time.
"""
function get_ρu₃qₜ_surface(flow::ShipwayHill2012VelocityProfile, thermo_params, t)
    # TODO: Get these values from the initial conditions:
    # lg_sfc = Fields.level(Fields.local_geometry_field(Y.f), CA.half)
    # ic = CA.InitialConditions.ShipwayHill2012()(p.params)
    # get_ρ(ls) = ls.ρ
    # ᶠρ_sfc = @. get_ρ(ic(lg_sfc)) <-- inconvenient since this materializes a Field
    # For now, just copy the values from the initial conditions:
    FT = eltype(thermo_params)
    rv_sfc = FT(0.015)  # water vapour mixing ratio at surface (kg/kg)
    q_tot_sfc = rv_sfc / (1 + rv_sfc)  # 0.0148 kg/kg
    p_sfc = FT(100_700)
    θ_sfc = FT(297.9)
    T =
        TD.saturation_adjustment(
            thermo_params,
            TD.pθ_li(),
            p_sfc,
            θ_sfc,
            q_tot_sfc;
            maxiter = 4,
            tol = FT(0),
        ).T
    ρ_sfc = TD.air_density(thermo_params, T, p_sfc, q_tot_sfc)  # 1.165 kg/m³
    w_sfc = Geometry.WVector(flow(0, t))
    return ρ_sfc * w_sfc * q_tot_sfc
end

struct TestDycoreConsistency end
struct ReproducibleRestart end

abstract type AbstractTimesteppingMode end
struct Explicit <: AbstractTimesteppingMode end
struct Implicit <: AbstractTimesteppingMode end

struct QuasiMonotoneLimiter end # For dispatching to use the ClimaCore QuasiMonotoneLimiter.

abstract type AbstractScaleBlendingMethod end
struct SmoothMinimumBlending <: AbstractScaleBlendingMethod end
struct HardMinimumBlending <: AbstractScaleBlendingMethod end
Base.broadcastable(x::AbstractScaleBlendingMethod) = tuple(x)

struct AtmosNumerics{EN_UP, TR_UP, ED_UP, SG_UP, ED_TR_UP, TDC, RR, LIM, DM, HD}
    """Enable specific upwinding schemes for specific equations"""
    energy_q_tot_upwinding::EN_UP
    tracer_upwinding::TR_UP
    edmfx_mse_q_tot_upwinding::ED_UP
    edmfx_sgsflux_upwinding::SG_UP
    edmfx_tracer_upwinding::ED_TR_UP
    """Add NaNs to certain equations to track down problems"""
    test_dycore_consistency::TDC
    """Whether the simulation is reproducible when restarting from a restart file"""
    reproducible_restart::RR
    limiter::LIM
    """Timestepping mode for diffusion: Explicit() or Implicit()"""
    diff_mode::DM
    """Hyperdiffusion model: nothing or Hyperdiffusion()"""
    hyperdiff::HD
end
Base.broadcastable(x::AtmosNumerics) = tuple(x)

"""
    AtmosNumerics(; kwargs...)

Create an AtmosNumerics struct. Upwinding schemes can be specified as symbols or strings,
which will be automatically converted to Val types for compile-time dispatch.
"""
function AtmosNumerics(;
    energy_q_tot_upwinding = :vanleer_limiter,
    tracer_upwinding = :vanleer_limiter,
    edmfx_mse_q_tot_upwinding = :first_order,
    edmfx_sgsflux_upwinding = :none,
    edmfx_tracer_upwinding = :first_order,
    test_dycore_consistency = nothing,
    reproducible_restart = nothing,
    limiter = nothing,
    diff_mode = Explicit(),
    hyperdiff = Hyperdiffusion{Float32}(;
        ν₄_vorticity_coeff = 0.150 * 1.238,
        divergence_damping_factor = 5,
        prandtl_number = 1.0,
    ),
    kwargs...,
)
    # Helper to convert symbols/strings to Val types, or keep Val types as-is
    parse_upwinding(x::Union{Symbol, String}) = Val(Symbol(x))
    parse_upwinding(x::Val) = x

    return AtmosNumerics(
        parse_upwinding(energy_q_tot_upwinding),
        parse_upwinding(tracer_upwinding),
        parse_upwinding(edmfx_mse_q_tot_upwinding),
        parse_upwinding(edmfx_sgsflux_upwinding),
        parse_upwinding(edmfx_tracer_upwinding),
        test_dycore_consistency,
        reproducible_restart,
        limiter,
        diff_mode,
        hyperdiff,
    )
end

const ValTF = Union{Val{true}, Val{false}}

struct EDMFXModel{
    EEM, EDM,
    ESMF <: ValTF, ESDF <: ValTF, ENP <: ValTF, EVD <: ValTF, EF <: ValTF,
    SBM <: AbstractScaleBlendingMethod,
}
    entr_model::EEM
    detr_model::EDM
    sgs_mass_flux::ESMF
    sgs_diffusive_flux::ESDF
    nh_pressure::ENP
    vertical_diffusion::EVD
    filter::EF
    scale_blending_method::SBM
end


# Convenience constructor that converts booleans to Val types
# This outer constructor allows passing booleans, which are converted to Val types
function EDMFXModel(;
    entr_model = nothing,
    detr_model = nothing,
    sgs_mass_flux::Union{Bool, ValTF} = false,
    sgs_diffusive_flux::Union{Bool, ValTF} = false,
    nh_pressure::Union{Bool, ValTF} = false,
    vertical_diffusion::Union{Bool, ValTF} = false,
    filter::Union{Bool, ValTF} = false,
    scale_blending_method,
    kwargs...,
)
    parse_val_tf(x::Bool) = Val(x)
    parse_val_tf(x::ValTF) = x
    # Convert booleans to Val types, keep Val types as-is
    return EDMFXModel(
        entr_model,
        detr_model,
        parse_val_tf(sgs_mass_flux),
        parse_val_tf(sgs_diffusive_flux),
        parse_val_tf(nh_pressure),
        parse_val_tf(vertical_diffusion),
        parse_val_tf(filter),
        scale_blending_method,
    )
end

# Grouped structs to reduce AtmosModel type parameters

"""
    SCMSetup

Groups Single-Column Model and Large-Eddy Simulation specific forcing, advection, and setup models.

These components are primarily used internally for testing, calibration, and research purposes
with single-column model setups. Most external users will not need these components.
"""
@kwdef struct SCMSetup{S, EF, LA, AT, SC}
    subsidence::S = nothing
    external_forcing::EF = nothing
    ls_adv::LA = nothing
    advection_test::AT = false
    scm_coriolis::SC = nothing
end

"""
    AtmosWater

Groups moisture and microphysics-related models and types.
"""
@kwdef struct AtmosWater{MM, CM, MTTS, TNM, SQ}
    microphysics_model::MM = DryModel()
    cloud_model::CM = QuadratureCloud()
    microphysics_tendency_timestepping::MTTS = nothing
    tracer_nonnegativity_method::TNM = nothing
    sgs_quadrature::SQ = nothing
end

"""
    AtmosRadiation

Groups radiation-related models and types.
"""
@kwdef struct AtmosRadiation{RM, IN}
    radiation_mode::RM = nothing
    insolation::IN = IdealizedInsolation()
end

"""
    AtmosTurbconv

Groups turbulence convection-related models and types.
"""
@kwdef struct AtmosTurbconv{EDMFX, TCM, SAM, SEDM, SNPM, SVM, SMM, SL, AMD, CHD}
    edmfx_model::EDMFX = nothing
    turbconv_model::TCM = nothing
    sgs_adv_mode::SAM = Explicit()
    sgs_entr_detr_mode::SEDM = Explicit()
    sgs_nh_pressure_mode::SNPM = Explicit()
    sgs_vertdiff_mode::SVM = Explicit()
    sgs_mf_mode::SMM = Explicit()
    smagorinsky_lilly::SL = nothing
    amd_les::AMD = nothing
    constant_horizontal_diffusion::CHD = nothing
end

"""
    AtmosGravityWave

Groups gravity wave-related models and types.
"""
@kwdef struct AtmosGravityWave{NOGW, OGW}
    non_orographic_gravity_wave::NOGW = nothing
    orographic_gravity_wave::OGW = nothing
end

"""
    AtmosSponge

Groups sponge-related models and types.
"""
@kwdef struct AtmosSponge{VS, RS}
    viscous_sponge::VS = nothing
    rayleigh_sponge::RS = nothing
end

"""
    AtmosSurface

Groups surface-related models and types.
"""
@kwdef struct AtmosSurface{ST, SM, SA}
    sfc_temperature::ST = ZonallySymmetricSST()
    surface_model::SM = PrescribedSST()
    surface_albedo::SA = ConstantAlbedo{Float32}(; α = 0.07)
end

# Add broadcastable for the new grouped types
Base.broadcastable(x::SCMSetup) = tuple(x)
Base.broadcastable(x::AtmosWater) = tuple(x)
Base.broadcastable(x::AtmosRadiation) = tuple(x)
Base.broadcastable(x::AtmosTurbconv) = tuple(x)
Base.broadcastable(x::AtmosGravityWave) = tuple(x)
Base.broadcastable(x::AtmosSponge) = tuple(x)
Base.broadcastable(x::AtmosSurface) = tuple(x)

struct AtmosModel{W, SCM, R, TC, PF, GW, VD, SP, SU, NU}
    water::W
    scm_setup::SCM
    radiation::R
    turbconv::TC
    prescribed_flow::PF
    gravity_wave::GW
    vertical_diffusion::VD
    sponge::SP
    surface::SU
    numerics::NU

    """Whether to apply surface flux tendency (independent of surface conditions)"""
    disable_surface_flux_tendency::Bool
end

# Map grouped struct types to their names in AtmosModel struct
const ATMOS_MODEL_GROUPS = (
    (AtmosWater, :water),
    (AtmosRadiation, :radiation),
    (AtmosTurbconv, :turbconv),
    (ShipwayHill2012VelocityProfile, :prescribed_flow),
    (AtmosGravityWave, :gravity_wave),
    (AtmosSponge, :sponge),
    (AtmosSurface, :surface),
    (AtmosNumerics, :numerics),
    (SCMSetup, :scm_setup),
)

# Auto-generate map from property_name to group_field
const GROUPED_PROPERTY_MAP = Dict{Symbol, Symbol}(
    property => group_field for
    (group_type, group_field) in ATMOS_MODEL_GROUPS for
    property in fieldnames(group_type)
)

# Forward property access: atmos.microphysics_model → atmos.water.microphysics_model
# Use ::Val constant for @generated compile-time access
@generated function Base.getproperty(
    atmos::AtmosModel,
    ::Val{property_name},
) where {property_name}
    if haskey(GROUPED_PROPERTY_MAP, property_name)
        group_field = GROUPED_PROPERTY_MAP[property_name]
        return quote
            group = getfield(atmos, $(QuoteNode(group_field)))
            getfield(group, $(QuoteNode(property_name)))
        end
    else
        return quote
            getfield(atmos, $(QuoteNode(property_name)))
        end
    end
end

@inline Base.getproperty(atmos::AtmosModel, property_name::Symbol) =
    getproperty(atmos, Val{property_name}())

Base.broadcastable(x::AtmosModel) = tuple(x)

"""
    AtmosModel(; kwargs...)

Create an AtmosModel with sensible defaults.

This constructor provides sensible defaults for a minimal dry atmospheric model with full customization through keyword arguments.

All model components are automatically organized into appropriate grouped sub-structs
internally:
- [`AtmosWater`](@ref)
- [`SCMSetup`](@ref)
- [`AtmosRadiation`](@ref)
- [`AtmosTurbconv`](@ref)
- [`AtmosGravityWave`](@ref)
- [`AtmosSponge`](@ref)
- [`AtmosSurface`](@ref)
- [`AtmosNumerics`](@ref)
The one exception is the top-level `disable_surface_flux_tendency` field, which is not grouped.

# Property Access
Arguments can be accessed both directly and through grouped structs:
```julia
model = AtmosModel(; microphysics_model = EquilibriumMicrophysics0M())
model.microphysics_model        # Direct access
model.water.microphysics_model  # Grouped access
```

# Example: Minimal model (uses defaults)
```julia
model = AtmosModel()  # Creates a basic dry atmospheric model
```

# Example: Dry model with Held-Suarez forcing and hyperdiffusion
```julia
model = AtmosModel(;
    radiation_mode = HeldSuarezForcing(),
    hyperdiff = Hyperdiffusion(;
        ν₄_vorticity_coeff = 1e15,
        divergence_damping_factor = 1.0,
        prandtl_number = 1.0
    )
)
```

# Example: Moist model with full radiation
```julia
model = AtmosModel(;
    microphysics_model = EquilibriumMicrophysics0M(),
    radiation_mode = RRTMGPI.AllSkyRadiation(),
)
```

# Default Configuration
The default AtmosModel provides:
- **Dry atmosphere**: DryModel()
- **Basic surface**: PrescribedSST() with ZonallySymmetricSST()
- **Cloud model**: QuadratureCloud() with SGS quadrature
- **Idealized insolation**: IdealizedInsolation()
- **Conservative numerics**: First-order upwinding with Explicit() timestepping
- **No advanced physics**: No radiation, turbulence, or forcing by default

# Available Structs

## AtmosWater
- `microphysics_model`: DryModel(), EquilibriumMicrophysics0M(), NonEquilibriumMicrophysics1M(), NonEquilibriumMicrophysics2M(), NonEquilibriumMicrophysics2MP3()
- `cloud_model`: GridScaleCloud(), QuadratureCloud()
- `microphysics_tendency_timestepping`: Explicit(), Implicit()
- `sgs_quadrature`: nothing or SGSQuadrature (subgrid-scale quadrature for microphysics tendencies)


## SCMSetup (Single-Column Model & LES specific - accessed via model.subsidence, model.external_forcing, etc.)
Internal testing and calibration components for single-column setups:
- `subsidence`: nothing or Bomex_subsidence, Rico_subsidence, DYCOMS_subsidence, etc
- `external_forcing`: nothing or external forcing objects (GCMForcing, ExternalDrivenTVForcing, ISDACForcing)
- `ls_adv`: nothing or LargeScaleAdvection()
- `advection_test`: Bool
- `scm_coriolis`: nothing or SCMCoriolis()

## AtmosRadiation
- `radiation_mode`: Radiation and atmospheric forcing modes
  - Global radiation: RRTMGPI.ClearSkyRadiation(), RRTMGPI.AllSkyRadiation()
  - Atmospheric forcing: HeldSuarezForcing() (for idealized dynamics)
  - SCM-specific: RadiationDYCOMS(), RadiationISDAC(), RadiationTRMM_LBA()
- `insolation`: IdealizedInsolation(), TimeVaryingInsolation(), etc.

## AtmosTurbconv
- `edmfx_model`: EDMFXModel()
- `turbconv_model`: nothing, PrognosticEDMFX(), DiagnosticEDMFX(), EDOnlyEDMFX()
- `sgs_adv_mode`, `sgs_entr_detr_mode`, `sgs_nh_pressure_mode`, `sgs_vertdiff_mode`, `sgs_mf_mode`: Explicit(), Implicit()
- `smagorinsky_lilly`: nothing or SmagorinskyLilly()
- `amd_les`: nothing or AnisotropicMinimumDissipation()
- `constant_horizontal_diffusion`: nothing or ConstantHorizontalDiffusion()

## AtmosGravityWave
- `non_orographic_gravity_wave`: nothing or NonOrographicGravityWave()
- `orographic_gravity_wave`: nothing or OrographicGravityWave()

## AtmosSponge
- `viscous_sponge`: nothing or ViscousSponge()
- `rayleigh_sponge`: nothing or RayleighSponge()

## AtmosSurface
- `sfc_temperature`: ZonallySymmetricSST(), RCEMIPIISST(), ExternalTVColumnSST()
- `surface_model`: PrescribedSST(), SlabOceanSST()
- `surface_albedo`: ConstantAlbedo(), RegressionFunctionAlbedo(), CouplerAlbedo()

## AtmosNumerics
- `energy_q_tot_upwinding`, `tracer_upwinding`, `edmfx_mse_q_tot_upwinding`, `edmfx_sgsflux_upwinding`, `edmfx_tracer_upwinding`: Val() upwinding schemes
- `test_dycore_consistency`: nothing or TestDycoreConsistency() for debugging
- `limiter`: nothing or QuasiMonotoneLimiter()
- `vertical_water_borrowing_species`: internal value `nothing` (apply to all tracers; config default is `~`), empty tuple (apply to none; config `[]`), or Tuple{Symbol, ...} from config string/list (e.g. `["ρq_tot"]`) to apply only to those tracers. See config `vertical_water_borrowing_species` in default_config.yml for YAML options.
  (Note: The vertical water borrowing limiter is created in the cache based on `AtmosWaterModel.tracer_nonnegativity_method`)
- `diff_mode`: Explicit(), Implicit() timestepping mode for diffusion
- `hyperdiff`: nothing or Hyperdiffusion()

## Top-level Options
- `vertical_diffusion`: nothing, VerticalDiffusion(), DecayWithHeightDiffusion()
- `disable_surface_flux_tendency`: Bool
"""
function AtmosModel(; kwargs...)
    group_kwargs, atmos_model_kwargs = _partition_atmos_model_kwargs(kwargs)

    # Create grouped structs - use provided complete objects or create from individual fields
    water = _create_grouped_struct(AtmosWater, atmos_model_kwargs, group_kwargs)
    scm_setup =
        _create_grouped_struct(SCMSetup, atmos_model_kwargs, group_kwargs)
    radiation =
        _create_grouped_struct(AtmosRadiation, atmos_model_kwargs, group_kwargs)
    turbconv =
        _create_grouped_struct(AtmosTurbconv, atmos_model_kwargs, group_kwargs)
    gravity_wave = _create_grouped_struct(
        AtmosGravityWave,
        atmos_model_kwargs,
        group_kwargs,
    )
    sponge =
        _create_grouped_struct(AtmosSponge, atmos_model_kwargs, group_kwargs)
    surface =
        _create_grouped_struct(AtmosSurface, atmos_model_kwargs, group_kwargs)
    numerics =
        _create_grouped_struct(AtmosNumerics, atmos_model_kwargs, group_kwargs)

    vertical_diffusion = get(atmos_model_kwargs, :vertical_diffusion, nothing)
    disable_surface_flux_tendency =
        get(atmos_model_kwargs, :disable_surface_flux_tendency, false)

    prescribed_flow = get(atmos_model_kwargs, :prescribed_flow, nothing)

    return AtmosModel{
        typeof(water),
        typeof(scm_setup),
        typeof(radiation),
        typeof(turbconv),
        typeof(prescribed_flow),
        typeof(gravity_wave),
        typeof(vertical_diffusion),
        typeof(sponge),
        typeof(surface),
        typeof(numerics),
    }(
        water,
        scm_setup,
        radiation,
        turbconv,
        prescribed_flow,
        gravity_wave,
        vertical_diffusion,
        sponge,
        surface,
        numerics,
        disable_surface_flux_tendency,
    )
end

"""
    _create_grouped_struct(StructType, atmos_model_kwargs, group_kwargs)

Helper function that creates a single grouped struct.
Uses the ATMOS_MODEL_GROUPS mapping to find the field name from the struct type.
Uses provided complete object or creates from individual fields.
"""
function _create_grouped_struct(StructType, atmos_model_kwargs, group_kwargs)
    field_name = get(Dict(ATMOS_MODEL_GROUPS), StructType, nothing)
    @assert !isnothing(field_name) "StructType $StructType not found in ATMOS_MODEL_GROUPS"
    complete_object = get(atmos_model_kwargs, field_name, nothing)
    return isnothing(complete_object) ?
           StructType(; group_kwargs[field_name]...) : complete_object
end

"""
    _partition_atmos_model_kwargs(kwargs)

Partition the given kwargs into grouped and direct kwargs matching the AtmosModel struct.

Helper function for the AtmosModel constructor.
"""
function _partition_atmos_model_kwargs(kwargs)

    # Merge default minimal model arguments with given kwargs

    # group_kwargs contains a Dict for each group in ATMOS_MODEL_GROUPS
    group_kwargs = Dict(map(ATMOS_MODEL_GROUPS) do (_, group_field)
        group_field => Dict{Symbol, Any}()
    end)

    # Sort kwargs into a hierarchy of dicts matching the AtmosModel struct
    atmos_model_kwargs = Dict{Symbol, Any}()
    unknown_args = Symbol[]

    for (key, value) in pairs(kwargs)
        if haskey(GROUPED_PROPERTY_MAP, key)
            group_field = GROUPED_PROPERTY_MAP[key]
            group_kwargs[group_field][key] = value
        elseif key in fieldnames(AtmosModel)
            atmos_model_kwargs[key] = value
        else
            push!(unknown_args, key)
        end
    end

    # Throw error for all unknown arguments at once
    if !isempty(unknown_args)
        _throw_unknown_atmos_model_argument_error(unknown_args)
    end

    return group_kwargs, atmos_model_kwargs
end

"""
    _throw_unknown_atmos_model_argument_error(unknown_args)

Throw a helpful error message for unknown AtmosModel constructor arguments.
"""
function _throw_unknown_atmos_model_argument_error(unknown_args)
    n_unknown = length(unknown_args)
    plural = n_unknown > 1 ? "s" : ""

    # All valid arguments: forwarded properties + direct AtmosModel fields
    available_forwarded = sort(collect(keys(GROUPED_PROPERTY_MAP)))
    available_direct = sort(collect(fieldnames(AtmosModel)))
    available_all = sort(unique([available_forwarded; available_direct]))

    error(
        "Unknown AtmosModel argument$plural: $(join(unknown_args, ", ")). " *
        "Available arguments:\n  " *
        join(available_all, "\n  "),
    )
end

# Convenience constructors for common configurations

"""
    DryAtmosModel(; kwargs...)

Create a dry atmospheric model with sensible defaults for dry simulations.

# Example
```julia
model = DryAtmosModel(;
    radiation_mode = HeldSuarezForcing(),
    hyperdiff = Hyperdiffusion(; ν₄_vorticity_coeff = 1e15, divergence_damping_factor = 1.0, prandtl_number = 1.0)
)
```
"""
function DryAtmosModel(; kwargs...)
    defaults = (microphysics_model = DryModel(),)
    return AtmosModel(; defaults..., kwargs...)
end

"""
    EquilMoistAtmosModel(; kwargs...)

Create an equilibrium moist atmospheric model with sensible defaults for moist simulations.
"""
function EquilMoistAtmosModel(; kwargs...)
    defaults = (
        microphysics_model = EquilibriumMicrophysics0M(),
        cloud_model = GridScaleCloud(),
        surface_model = PrescribedSST(),
        sfc_temperature = ZonallySymmetricSST(),
        insolation = IdealizedInsolation(),
    )
    return AtmosModel(; defaults..., kwargs...)
end

"""
    NonEquilMoistAtmosModel(; kwargs...)

Create a non-equilibrium moist atmospheric model with sensible defaults.
"""
function NonEquilMoistAtmosModel(; kwargs...)
    defaults = (
        microphysics_model = NonEquilibriumMicrophysics1M(),
        microphysics_tendency_timestepping = Explicit(),
    )
    return AtmosModel(; defaults..., kwargs...)
end


abstract type AbstractCallbackFrequency end
struct EveryNSteps <: AbstractCallbackFrequency
    n::Int
end
struct EveryΔt{FT} <: AbstractCallbackFrequency
    Δt::FT
end
struct AtmosCallback{F, CBF <: AbstractCallbackFrequency, VI <: Vector{Int}}
    f!::F
    cbf::CBF
    n_measured_calls::VI
end
callback_frequency(cb::AtmosCallback) = cb.cbf
prescribed_every_n_steps(x::EveryNSteps) = x.n
prescribed_every_n_steps(cb::AtmosCallback) = prescribed_every_n_steps(cb.cbf)

prescribed_every_Δt_steps(x::EveryΔt) = x.Δt
prescribed_every_Δt_steps(cb::AtmosCallback) = prescribed_every_Δt_steps(cb.cbf)

# TODO: improve accuracy
n_expected_calls(cbf::EveryΔt, dt, tspan) = (tspan[2] - tspan[1]) / cbf.Δt
n_expected_calls(cbf::EveryNSteps, dt, tspan) =
    ((tspan[2] - tspan[1]) / dt) / cbf.n
n_expected_calls(cb::AtmosCallback, dt, tspan) =
    n_expected_calls(cb.cbf, dt, tspan)

AtmosCallback(f!, cbf) = AtmosCallback(f!, cbf, Int[0])
function (cb::AtmosCallback)(integrator)
    cb.f!(integrator)
    cb.n_measured_calls[] += 1
    return nothing
end
n_measured_calls(cb::AtmosCallback) = cb.n_measured_calls[]

struct AtmosConfig{FT, TD, PA, C, CF}
    toml_dict::TD
    parsed_args::PA
    comms_ctx::C
    config_files::CF
    job_id::String
end

Base.eltype(::AtmosConfig{FT}) where {FT} = FT

TupleOrVector(T) = Union{Tuple{<:T, Vararg{T}}, Vector{<:T}}

# Use short, relative paths, if possible.
function normrelpath(file)
    rfile = normpath(relpath(file, dirname(config_path)))
    return if isfile(rfile) && samefile(rfile, file)
        rfile
    else
        file
    end
end

function maybe_add_default(config_files, default_config_file)
    return if any(x -> samefile(x, default_config_file), config_files)
        config_files
    else
        (default_config_file, config_files...)
    end
end

"""
    AtmosConfig(
        config_file::String = default_config_file;
        job_id = config_id_from_config_file(config_file),
        comms_ctx = nothing,
    )
    AtmosConfig(
        config_files::Union{NTuple{<:Any, String} ,Vector{String}};
        job_id = config_id_from_config_files(config_files),
        comms_ctx = nothing,
    )

Helper function for the AtmosConfig constructor. Reads a YAML file into a Dict
and passes it to the AtmosConfig constructor.
"""
AtmosConfig(
    config_file::String = default_config_file;
    job_id = config_id_from_config_file(config_file),
    comms_ctx = nothing,
) = AtmosConfig((config_file,); job_id, comms_ctx)

function AtmosConfig(
    config_files::TupleOrVector(String);
    job_id = config_id_from_config_files(config_files),
    comms_ctx = nothing,
)

    all_config_files =
        normrelpath.(maybe_add_default(config_files, default_config_file))
    configs = map(all_config_files) do config_file
        strip_help_messages(load_yaml_file(config_file))
    end
    return AtmosConfig(
        configs;
        comms_ctx,
        config_files = all_config_files,
        job_id,
    )
end

"""
    AtmosConfig(
        configs::Union{NTuple{<:Any, Dict} ,Vector{Dict}};
        comms_ctx = nothing,
        config_files,
        job_id
    )

Constructs the AtmosConfig from the Dicts passed in. This Dict overrides all of
the default configurations set in `default_config_dict()`.
"""
AtmosConfig(configs::AbstractDict; kwargs...) =
    AtmosConfig((configs,); kwargs...)
function AtmosConfig(
    configs::TupleOrVector(AbstractDict);
    comms_ctx = nothing,
    config_files = [default_config_file],
    job_id = "",
)
    config_files = map(x -> normrelpath(x), config_files)

    # using config_files = [default_config_file] as a default
    # relies on the fact that override_default_config uses
    # default_config_file.
    config = merge(configs...)
    comms_ctx = isnothing(comms_ctx) ? get_comms_context(config) : comms_ctx
    config = override_default_config(config)

    FT = config["FLOAT_TYPE"] == "Float64" ? Float64 : Float32
    toml_dict = CP.create_toml_dict(
        FT;
        override_file = CP.merge_toml_files(config["toml"]),
    )
    config = config_with_resolved_and_acquired_artifacts(config, comms_ctx)

    isempty(job_id) &&
        @warn "`job_id` is empty and likely not passed to AtmosConfig"

    @info "Making AtmosConfig with config files: $(sprint(config_summary, config_files))"

    C = typeof(comms_ctx)
    TD = typeof(toml_dict)
    PA = typeof(config)
    CF = typeof(config_files)
    return AtmosConfig{FT, TD, PA, C, CF}(
        toml_dict,
        config,
        comms_ctx,
        config_files,
        job_id,
    )
end

"""
    maybe_resolve_and_acquire_artifacts(input_str::AbstractString, context)

When given a string of the form `artifact"name"/something/else`, resolve the
artifact path and download it (if not already available).

In all the other cases, return the input unchanged.
"""
function maybe_resolve_and_acquire_artifacts(
    input_str::AbstractString,
    context,
)
    matched = match(r"artifact\"([a-zA-Z0-9_]+)\"(\/.*)?", input_str)
    if isnothing(matched)
        return input_str
    else
        artifact_name, other_path = matched
        return joinpath(
            @clima_artifact(string(artifact_name), context),
            lstrip(other_path, '/'),
        )
    end
end

function maybe_resolve_and_acquire_artifacts(
    input,
    _,
)
    return input
end

"""
    config_with_resolved_and_acquired_artifacts(input_str::AbstractString, context)

Substitute strings of the form `artifact"name"/something/else` with the actual
artifact path.
"""
function config_with_resolved_and_acquired_artifacts(
    config::AbstractDict,
    context,
)
    return Dict(
        k => maybe_resolve_and_acquire_artifacts(v, context) for
        (k, v) in config
    )
end

function config_summary(io::IO, config_files)
    print(io, '\n')
    for x in config_files
        println(io, "   $x")
    end
end
