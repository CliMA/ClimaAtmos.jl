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
struct NonEquilibriumMicrophysics1M <: AbstractMicrophysicsModel
    n_substeps::Int  # number of susteps for time averaging tendencies
    function NonEquilibriumMicrophysics1M(; n_substeps = 1)
        return new(n_substeps)
    end
end
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
- `ρq_lcl`, `ρq_icl`, `ρq_rai`, `ρq_sno`,
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
 but may include e.g. `ρq_tot`, `ρq_lcl`, `ρq_icl`, ...
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
 `χ ∈ {q_lclʲ, q_raiʲ, q_iclʲ, q_snoʲ}` (depending on the microphysical model).
 `α_χ` is a damping coefficient for each variable, and `ζ` is the damping function

 ```math
 ζ(z) = sin^2(π(z-zd)/(zmax-zd)/2)
 ```

 with `zd` the lower damping height and `zmax` the domain top height.

 Separate damping coefficients are used:
 - `α_uₕ`: horizontal velocity, `uₕ`;
 - `α_w`: vertical velocity, `u₃`, `u₃ʲ`;
 - `α_sgs_tracer`: subgrid-scale tracer variables, `ρtke`, `mseʲ`, `q_totʲ`,
    `q_lclʲ`, `q_raiʲ`, `q_iclʲ`, `q_snoʲ`.

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
    # TODO: Get these values from the setup instead of hardcoding:
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
Base.broadcastable(x::AbstractTimesteppingMode) = tuple(x)

struct QuasiMonotoneLimiter end # For dispatching to use the ClimaCore QuasiMonotoneLimiter.

abstract type AbstractScaleBlendingMethod end
struct SmoothMinimumBlending <: AbstractScaleBlendingMethod end
struct HardMinimumBlending <: AbstractScaleBlendingMethod end
Base.broadcastable(x::AbstractScaleBlendingMethod) = tuple(x)

"""
    AtmosNumerics(; kwargs...)

Group struct holding numerical-method choices: per-equation upwinding schemes,
diffusion timestepping mode, hyperdiffusion, limiter, and debug toggles.
Accessed on `AtmosModel` via `model.numerics` or directly via `model.<field>`.

# Fields

## `energy_q_tot_upwinding` / `tracer_upwinding` / `edmfx_mse_q_tot_upwinding` / `edmfx_sgsflux_upwinding` / `edmfx_tracer_upwinding`
Upwinding scheme for each tracer-advection family. Each accepts a `Val{Symbol}`
(string/symbol values are converted to `Val` at construction).

| Value | YAML | Description |
|---|---|---|
| `Val(:none)` | `"none"` | No upwinding (centered). |
| `Val(:first_order)` | `"first_order"` | First-order upwind. |
| `Val(:third_order)` | `"third_order"` | Third-order upwind. |
| `Val(:boris_book)` | `"boris_book"` | Boris-Book limiter. |
| `Val(:zalesak)` | `"zalesak"` | Zalesak limiter. |
| `Val(:vanleer_limiter)` | `"vanleer_limiter"` | Van Leer limiter (default for grid-mean tracers). |

The matching YAML keys are `energy_q_tot_upwinding`, `tracer_upwinding`,
`edmfx_mse_q_tot_upwinding`, `edmfx_sgsflux_upwinding`, `edmfx_tracer_upwinding`.

## `test_dycore_consistency`
Inject NaNs into selected equations for debugging. Default: `nothing`.

| Value | YAML `test_dycore_consistency` | Description |
|---|---|---|
| `nothing` | `false` | Disabled. |
| [`TestDycoreConsistency`](@ref) | `true` | Enable consistency check. |

## `reproducible_restart`
Force exact bit-reproducibility when restarting from a checkpoint.
Default: `nothing`.

| Value | YAML `reproducible_restart` | Description |
|---|---|---|
| `nothing` | `false` | Standard restart. |
| [`ReproducibleRestart`](@ref) | `true` | Bit-reproducible restart. |

## `limiter`
SEM quasi-monotone limiter on tracer transport. Default: `nothing`.

| Value | YAML `apply_sem_quasimonotone_limiter` | Description |
|---|---|---|
| `nothing` | `false` | Disabled. |
| [`QuasiMonotoneLimiter`](@ref) | `true` | Apply ClimaCore's SEM quasi-monotone limiter. |

## `diff_mode`
Timestepping mode for vertical diffusion. Default: `Explicit()`.

| Value | YAML `implicit_diffusion` | Description |
|---|---|---|
| [`Explicit`](@ref) | `false` | Explicit RHS treatment. |
| [`Implicit`](@ref) | `true` | Included in the implicit Jacobian. |

## `hyperdiff`
Fourth-order hyperdiffusion. Default: a `Hyperdiffusion{Float32}` with
ClimaAtmos defaults.

| Value | YAML `hyperdiff` | Description |
|---|---|---|
| `nothing` | `false` / `nothing` | Disabled. |
| [`Hyperdiffusion`](@ref) | `"Hyperdiffusion"` | Custom hyperdiffusion with coefficients from YAML/params. |
| [`Hyperdiffusion`](@ref) (CAM_SE preset) | `"CAM_SE"` | CAM_SE hyperdiffusion preset; rejects custom coefficients. |

# See also

[`AtmosModel`](@ref), [`Hyperdiffusion`](@ref).
"""
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
    SCMSetup(; kwargs...)

Group struct holding single-column-model (SCM) and large-eddy-simulation (LES)
specific forcings: subsidence, external forcing, large-scale advection, and
SCM Coriolis. These are primarily used for testing, calibration, and idealized
research. Accessed on `AtmosModel` via `model.scm_setup` or directly.

# Fields

## `subsidence`
Vertical subsidence profile. Default: `nothing`.

| Value | YAML `subsidence` | Description |
|---|---|---|
| `nothing` | `~` | No subsidence. |
| [`Subsidence`](@ref) | `"Bomex"` / `"Rico"` / `"DYCOMS"` | Case-specific subsidence profile. |

`"DYCOMS"` requires `radiation_mode isa RadiationDYCOMS` so the divergence
rate is consistent.

## `external_forcing`
External (e.g. GCM-driven or reanalysis-driven) forcing. Default: `nothing`.

| Value | YAML `external_forcing` | Description |
|---|---|---|
| `nothing` | `~` | No external forcing. |
| [`GCMForcing`](@ref) | `"GCM"` | GCM-driven SCM forcing read from a NetCDF cfsite file. |
| [`ExternalDrivenTVForcing`](@ref) | `"ReanalysisTimeVarying"` / `"ReanalysisMonthlyAveragedDiurnal"` | Time-varying ERA5-driven forcing. Column-only. |
| [`ISDACForcing`](@ref) | `"ISDAC"` | ISDAC case forcing. |

## `ls_adv`
Large-scale advection profile. Default: `nothing`.

| Value | YAML `ls_adv` | Description |
|---|---|---|
| `nothing` | `~` | No large-scale advection. |
| [`LargeScaleAdvection`](@ref) | `"Bomex"` / `"Rico"` / `"ARM_SGP"` / etc. | Case-specific advection of T and qt. |

## `advection_test`
Whether to enable the dycore advection test (passive scalar advection check).
Default: `false`.

## `scm_coriolis`
SCM Coriolis forcing (geostrophic-wind specification). Default: `nothing`.

| Value | YAML `scm_coriolis` | Description |
|---|---|---|
| `nothing` | `~` | No SCM Coriolis. |
| `NamedTuple{(:prof_ug, :prof_vg, :coriolis_param)}` | `"Bomex"` / `"Rico"` / `"DYCOMS_RF01"` / `"DYCOMS_RF02"` / `"GABLS"` | Case-specific geostrophic profiles. |

# See also

[`AtmosModel`](@ref), [`AbstractForcing`](@ref).
"""
@kwdef struct SCMSetup{S, EF, LA, AT, SC}
    subsidence::S = nothing
    external_forcing::EF = nothing
    ls_adv::LA = nothing
    advection_test::AT = false
    scm_coriolis::SC = nothing
end

"""
    AtmosWater(; kwargs...)

Group struct holding moisture, microphysics, and cloud configuration. Accessed
on `AtmosModel` via `model.water` or directly via `model.<field>` (the
property-forwarding macro flattens these names).

# Fields

## `microphysics_model`
Microphysics scheme used by the moisture-equation tendency.
Default: `DryModel()`.

| Value | YAML | Description |
|---|---|---|
| [`DryModel`](@ref) | `"dry"` | No moisture tracers; tendencies skipped. |
| [`EquilibriumMicrophysics0M`](@ref) | `"0M"` | Bulk equilibrium 0-moment scheme. Requires `sgs_quadrature ≠ nothing`. |
| [`NonEquilibriumMicrophysics1M`](@ref) | `"1M"` | 1-moment microphysics (cloud liquid, ice, rain, snow). |
| [`NonEquilibriumMicrophysics2M`](@ref) | `"2M"` | 2-moment microphysics (number + mass concentrations). |
| [`NonEquilibriumMicrophysics2MP3`](@ref) | `"2MP3"` | 2-moment warm rain + P3 ice scheme. |

## `cloud_model`
Cloud-fraction calculation used by radiation and microphysics.
Default: `QuadratureCloud()`.

| Value | YAML | Description |
|---|---|---|
| [`GridScaleCloud`](@ref) | `"grid_scale"` | Cloud fraction = 0 or 1 from grid-scale state. |
| [`QuadratureCloud`](@ref) | `"quadrature"` | SGS quadrature over the moisture-distribution PDF. |
| [`MLCloud`](@ref) | `"MLCloud"` | Neural-network cloud fraction. Requires the `cloud_fraction_nn` artifact. |

## `microphysics_tendency_timestepping`
Whether microphysics tendencies are integrated implicitly or explicitly.
Default: `nothing` (set by `AtmosWater(config, params)` based on `implicit_microphysics`).

| Value | YAML `implicit_microphysics` | Description |
|---|---|---|
| [`Explicit`](@ref) | `false` | Microphysics treated as an explicit RHS term. |
| [`Implicit`](@ref) | `true` | Microphysics included in the implicit Jacobian. |

## `tracer_nonnegativity_method`
Method for enforcing tracer ≥ 0. Default: `nothing` (no enforcement).

| Value | YAML | Description |
|---|---|---|
| `nothing` | `~` | No nonnegativity enforcement. |
| [`TracerNonnegativityElementConstraint`](@ref) | `"elementwise_constraint"`[`_qtot`] | Redistribute tracer mass within each element. |
| [`TracerNonnegativityVaporConstraint`](@ref) | `"vapor_constraint"`[`_qtot`] | Redistribute mass between tracer and vapor. |
| [`TracerNonnegativityVaporTendency`](@ref) | `"vapor_tendency"` | Apply a tendency exchanging mass with vapor. |
| [`TracerNonnegativityVerticalWaterBorrowing`](@ref) | `"vertical_water_borrowing"` | Vertical mass borrowing limiter. |

The `_qtot` suffix on the YAML name (e.g. `"vapor_constraint_qtot"`) extends
the constraint to the total water tracer; only `elementwise_constraint` and
`vapor_constraint` accept it.

## `sgs_quadrature`
Subgrid-scale moisture-distribution quadrature. Default: `nothing` (no SGS quadrature).

| Value | YAML `use_sgs_quadrature` | Description |
|---|---|---|
| `nothing` | `false` | No SGS quadrature; grid-mean tendencies. |
| [`SGSQuadrature`](@ref) | `true` | Gauss-Hermite quadrature over the moisture PDF. |

## `terminal_velocity_mode`
Terminal-velocity treatment for precipitating species.
Default: `DiagnosticTerminalVelocity()`.

| Value | YAML `fixed_terminal_velocity` | Description |
|---|---|---|
| [`DiagnosticTerminalVelocity`](@ref) | `false` | Diagnostic velocity from microphysics scheme. |
| [`FixedTerminalVelocity`](@ref) | `true` | Constant velocity from `params.fixed_*_terminal_velocity`. |

# Cross-field constraints

- `EquilibriumMicrophysics0M` requires `sgs_quadrature ≠ nothing`. Validated at
  construction (`AtmosWater(config, params)` errors when violated).
- `MLCloud` requires the `cloud_fraction_nn` artifact to be downloadable.

# See also

[`AtmosModel`](@ref), [`AtmosRadiation`](@ref) (for the cloud-in-radiation toggle),
[`AbstractMicrophysicsModel`](@ref).
"""
@kwdef struct AtmosWater{MM, CM, MTTS, TNM, SQ, TVM}
    microphysics_model::MM = DryModel()
    cloud_model::CM = QuadratureCloud()
    microphysics_tendency_timestepping::MTTS = nothing
    tracer_nonnegativity_method::TNM = nothing
    sgs_quadrature::SQ = nothing
    terminal_velocity_mode::TVM = DiagnosticTerminalVelocity()
end

"""
    AtmosRadiation(; kwargs...)

Group struct holding radiation and insolation configuration. Accessed on
`AtmosModel` via `model.radiation` or directly via `model.radiation_mode` /
`model.insolation`.

# Fields

## `radiation_mode`
Radiation parameterization. Default: `nothing` (no radiation).

| Value | YAML `rad` | Description |
|---|---|---|
| `nothing` | `nothing` / `"nothing"` | No radiation. |
| [`RRTMGPI.GrayRadiation`](@ref) | `"gray"` | RRTMGP gray-atmosphere shortwave + longwave. |
| [`RRTMGPI.ClearSkyRadiation`](@ref) | `"clearsky"` | RRTMGP clear-sky radiation. |
| [`RRTMGPI.AllSkyRadiation`](@ref) | `"allsky"` | RRTMGP with cloud optics. |
| [`RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics`](@ref) | `"allskywithclear"` | All-sky with clear-sky diagnostic outputs. |
| [`HeldSuarezForcing`](@ref) | `"held_suarez"` | Held-Suarez idealized thermal forcing. |
| [`RadiationDYCOMS`](@ref) | (setup-driven) | DYCOMS-RF case longwave parameterization. |
| [`RadiationISDAC`](@ref) | (setup-driven) | ISDAC case radiation. |
| [`RadiationTRMM_LBA`](@ref) | (setup-driven) | TRMM-LBA case radiation. |

When `radiation_mode isa HeldSuarezForcing`, the YAML driver also forces
`disable_momentum_vertical_diffusion = true` on `VerticalDiffusion`.

## `insolation`
Insolation source. Default: `IdealizedInsolation()`.

| Value | YAML `insolation` | Description |
|---|---|---|
| [`IdealizedInsolation`](@ref) | `"idealized"` | Diurnally varying insolation, fixed orbit. |
| [`TimeVaryingInsolation`](@ref) | `"timevarying"` | Realistic time-varying solar declination. |
| [`RCEMIPIIInsolation`](@ref) | `"rcemipii"` | RCEMIP-II prescribed solar geometry. |
| [`GCMDrivenInsolation`](@ref) | `"gcmdriven"` | Insolation read from GCM forcing file. |
| [`ExternalTVInsolation`](@ref) | `"externaldriventv"` | Externally driven time-varying insolation. |

# Cross-field constraints

- All-sky and clear-sky modes interact with `AtmosWater.cloud_model` and the
  `prescribe_clouds_in_radiation` YAML knob. Setting `idealized_clouds: true`
  with `prescribe_clouds_in_radiation` is rejected at construction.

# See also

[`AtmosModel`](@ref), [`AtmosWater`](@ref) (cloud model coupling).
"""
@kwdef struct AtmosRadiation{RM, IN}
    radiation_mode::RM = nothing
    insolation::IN = IdealizedInsolation()
end

"""
    AtmosTurbconv(; kwargs...)

Group struct holding turbulence-and-convection configuration: EDMF mass-flux
modeling, sub-grid-scale (SGS) timestepping modes, and large-eddy-simulation
(LES) closures. Accessed on `AtmosModel` via `model.turbconv` or directly via
`model.<field>`.

# Fields

## `edmfx_model`
EDMF (eddy-diffusivity / mass-flux) model configuration. Default: `nothing`.

Pass `nothing` to disable EDMF, or an [`EDMFXModel`](@ref) instance whose
fields select the entrainment/detrainment closure, mass-flux toggles, and
scale-blending method. See [`EDMFXModel`](@ref) for its options.

## `turbconv_model`
Turbulent-convection scheme. Default: `nothing`.

| Value | YAML `turbconv` | Description |
|---|---|---|
| `nothing` | `~` | No turbconv scheme. |
| [`EDOnlyEDMFX`](@ref) | `"edonly_edmfx"` | Eddy-diffusivity only (no mass flux). |
| [`PrognosticEDMFX`](@ref) | `"prognostic_edmfx"` | Prognostic EDMF with prognostic updrafts. |
| [`DiagnosticEDMFX`](@ref) | `"diagnostic_edmfx"` | Diagnostic EDMF with diagnostic updrafts. |

## `sgs_adv_mode` / `sgs_entr_detr_mode` / `sgs_nh_pressure_mode` / `sgs_vertdiff_mode` / `sgs_mf_mode`
Timestepping mode for the corresponding SGS term. Each accepts:

| Value | YAML | Description |
|---|---|---|
| [`Explicit`](@ref) | `false` | Treat the term as an explicit RHS contribution. |
| [`Implicit`](@ref) | `true` | Include the term in the implicit Jacobian. |

The matching YAML keys are `implicit_sgs_advection`, `implicit_sgs_entr_detr`,
`implicit_sgs_nh_pressure`, `implicit_sgs_vertdiff`, `implicit_sgs_mass_flux`.
Default: `Explicit()` for all five.

## `smagorinsky_lilly`
Smagorinsky-Lilly LES eddy viscosity. Default: `nothing`.

| Value | YAML `smagorinsky_lilly` | Description |
|---|---|---|
| `nothing` | `~` | Disabled. |
| [`SmagorinskyLilly`](@ref) | `"UVW"` / `"UV"` / `"W"` / `"UV_W"` | Smagorinsky-Lilly with the chosen axis combination. |

## `amd_les`
Anisotropic Minimum Dissipation LES closure. Default: `nothing`.

| Value | YAML `amd_les` | Description |
|---|---|---|
| `nothing` | `false` | Disabled. |
| [`AnisotropicMinimumDissipation`](@ref) | `true` | AMD closure with coefficient from `params.c_amd`. |

## `constant_horizontal_diffusion`
Constant horizontal diffusion. Default: `nothing`.

| Value | YAML `constant_horizontal_diffusion` | Description |
|---|---|---|
| `nothing` | `false` | Disabled. |
| [`ConstantHorizontalDiffusion`](@ref) | `true` | Constant `D` from `params.constant_horizontal_diffusion_D`. |

# See also

[`AtmosModel`](@ref), [`EDMFXModel`](@ref), [`AbstractEDMF`](@ref).
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
    AtmosGravityWave(; kwargs...)

Group struct holding non-orographic and orographic gravity-wave drag
configuration. Accessed on `AtmosModel` via `model.gravity_wave` or directly
via `model.non_orographic_gravity_wave` / `model.orographic_gravity_wave`.

# Fields

## `non_orographic_gravity_wave`
Non-orographic gravity-wave-drag parameterization. Default: `nothing`.

| Value | YAML `non_orographic_gravity_wave` | Description |
|---|---|---|
| `nothing` | `false` | Disabled. |
| [`NonOrographicGravityWave`](@ref) | `true` | NGW spectral drag (Alexander-Dunkerton 1999 style). |

## `orographic_gravity_wave`
Orographic gravity-wave-drag parameterization. Default: `nothing`.

| Value | YAML `orographic_gravity_wave` | Description |
|---|---|---|
| `nothing` | `~` | Disabled. |
| [`LinearOrographicGravityWave`](@ref) | `"linear"` | Linear-mountain wave parameterization. |
| [`FullOrographicGravityWave`](@ref) | `"raw_topo"` | Full nonlinear scheme using preprocessed topography drag tensor (computed at runtime). |
| [`FullOrographicGravityWave`](@ref) | `"gfdl_restart"` | Full nonlinear scheme using GFDL-style precomputed drag tensor (loaded from artifact). |

# See also

[`AtmosModel`](@ref), [`OrographicGravityWave`](@ref), [`AbstractGravityWave`](@ref).
"""
@kwdef struct AtmosGravityWave{NOGW, OGW}
    non_orographic_gravity_wave::NOGW = nothing
    orographic_gravity_wave::OGW = nothing
end

"""
    AtmosSponge(; kwargs...)

Group struct holding viscous and Rayleigh sponge configuration. Sponges damp
upward-propagating waves near the model top. Accessed on `AtmosModel` via
`model.sponge` or directly via `model.viscous_sponge` / `model.rayleigh_sponge`.

# Fields

## `viscous_sponge`
Viscous sponge (Laplacian damping above `zd`). Default: `nothing`.

| Value | YAML `viscous_sponge` | Description |
|---|---|---|
| `nothing` | `false` / `"none"` | Disabled. |
| [`ViscousSponge`](@ref) | `true` / `"ViscousSponge"` | Damping with coefficients from `params.zd_viscous` and `params.kappa_2_sponge`. |

## `rayleigh_sponge`
Rayleigh sponge (linear-relaxation damping above `zd`). Default: `nothing`.

| Value | YAML `rayleigh_sponge` | Description |
|---|---|---|
| `nothing` | `false` | Disabled. |
| [`RayleighSponge`](@ref) | `true` / `"RayleighSponge"` | Damping with coefficients from `params.zd_rayleigh`, `alpha_rayleigh_*`. |

# See also

[`AtmosModel`](@ref), [`SpongeModel`](@ref).
"""
@kwdef struct AtmosSponge{VS, RS}
    viscous_sponge::VS = nothing
    rayleigh_sponge::RS = nothing
end

"""
    AtmosSurface(; kwargs...)

Group struct holding surface-temperature, surface-model, and albedo
configuration. Accessed on `AtmosModel` via `model.surface` or directly via
`model.sfc_temperature` / `model.surface_model` / `model.surface_albedo`.

# Fields

## `sfc_temperature`
Surface-temperature distribution (set by the chosen `Setups.*` setup type, not
by a YAML knob). Default: `ZonallySymmetricSST()`.

| Value | Description |
|---|---|
| [`ZonallySymmetricSST`](@ref) | Latitude-dependent zonally-symmetric SST. |
| [`RCEMIPIISST`](@ref) | RCEMIP-II prescribed SST distribution. |
| [`ExternalTVColumnSST`](@ref) | Time-varying SST from external forcing column. |

## `surface_model`
Surface-model treatment (prognostic vs. prescribed).
Default: `PrescribedSST()`.

| Value | YAML `prognostic_surface` | Description |
|---|---|---|
| [`PrescribedSST`](@ref) | `false` / `"PrescribedSST"` | Surface temperature is prescribed by `sfc_temperature`. |
| [`SlabOceanSST`](@ref) | `true` / `"SlabOceanSST"` | Slab-ocean prognostic SST. |

## `surface_albedo`
Surface albedo model. Default: `ConstantAlbedo{Float32}(; α = 0.07)` (low value
suitable for ocean surfaces).

| Value | YAML `albedo_model` | Description |
|---|---|---|
| [`ConstantAlbedo`](@ref) | `"ConstantAlbedo"` | Constant albedo from `params.idealized_ocean_albedo`. |
| [`RegressionFunctionAlbedo`](@ref) | `"RegressionFunctionAlbedo"` | Refractive-index based regression. Requires `radiation_mode ≠ nothing`. |
| [`CouplerAlbedo`](@ref) | `"CouplerAlbedo"` | Albedo set externally by the coupler. |

# See also

[`AtmosModel`](@ref), [`AbstractSST`](@ref), [`AbstractSurfaceTemperature`](@ref).
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

Create an `AtmosModel` with sensible defaults. Configuration is grouped by
subsystem; every kwarg either targets a group struct directly or is forwarded
to its owning group via the `@generated` `getproperty` flattening.

# Subsystem groups

Each group's docstring lists the available options for its fields:

- [`AtmosWater`](@ref) — moisture, microphysics, clouds
- [`AtmosRadiation`](@ref) — radiation mode, insolation
- [`AtmosTurbconv`](@ref) — EDMF, turbconv scheme, SGS timestepping, LES closures
- [`AtmosGravityWave`](@ref) — non-orographic and orographic gravity-wave drag
- [`AtmosSponge`](@ref) — viscous and Rayleigh sponges
- [`AtmosSurface`](@ref) — surface temperature, surface model, albedo
- [`AtmosNumerics`](@ref) — upwinding, hyperdiffusion, limiter, debug toggles
- [`SCMSetup`](@ref) — single-column-model forcings (subsidence, ls_adv, etc.)

Two top-level fields are not grouped:

- `vertical_diffusion`: `nothing`, [`VerticalDiffusion`](@ref), or [`DecayWithHeightDiffusion`](@ref)
- `disable_surface_flux_tendency::Bool`: skip surface-flux momentum tendency

The `prescribed_flow` field also lives at the top level; see
[`ShipwayHill2012VelocityProfile`](@ref).

# Flat vs. grouped access

Kwargs accept either flat field names or group structs:

```julia
# Flat field name — forwarded to the right group automatically
model = AtmosModel(; microphysics_model = EquilibriumMicrophysics0M())

# Or pass a complete group struct
model = AtmosModel(; water = AtmosWater(; microphysics_model = EquilibriumMicrophysics0M()))

# Property access works both ways
model.microphysics_model        # forwarded
model.water.microphysics_model  # explicit
```

# Examples

```julia
# Minimal: dry atmosphere with defaults
model = AtmosModel()

# Held-Suarez idealized forcing
model = AtmosModel(; radiation_mode = HeldSuarezForcing())

# Moist global model with all-sky radiation
model = AtmosModel(;
    microphysics_model = EquilibriumMicrophysics0M(),
    radiation_mode = RRTMGPI.AllSkyRadiation(),
)
```

# Convenience constructors

For common scientifically-meaningful configurations, see the
[`Presets`](@ref ClimaAtmos.Presets) module.
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
    atmos_toml = map(config["toml"]) do file
        isfile(file) ? file :
        isfile(joinpath(pkgdir(@__MODULE__), file)) ?
        joinpath(pkgdir(@__MODULE__), file) : error("Parameter file $file not found.")
    end
    toml_dict = CP.create_toml_dict(
        FT;
        override_file = CP.merge_toml_files(atmos_toml),
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
