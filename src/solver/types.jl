import ClimaCore.Quadratures.GaussQuadrature as GQ
import StaticArrays as SA
import Thermodynamics as TD
import Dates

import ClimaUtilities.ClimaArtifacts: @clima_artifact
import LazyArtifacts

abstract type AbstractMoistureModel end
abstract type AbstractMoistModel <: AbstractMoistureModel end
struct DryModel <: AbstractMoistureModel end
struct EquilMoistModel <: AbstractMoistModel end
struct NonEquilMoistModel <: AbstractMoistModel end

abstract type AbstractPrecipitationModel end
struct NoPrecipitation <: AbstractPrecipitationModel end
struct Microphysics0Moment <: AbstractPrecipitationModel end
struct Microphysics1Moment <: AbstractPrecipitationModel end
struct Microphysics2Moment <: AbstractPrecipitationModel end

"""
    Microphysics2MomentP3 <: AbstractPrecipitationModel

Struct used for dispatch to the 2-moment warm rain + P3 ice microphysics parameterizations
"""
struct Microphysics2MomentP3 <: AbstractPrecipitationModel end

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
    SGSQuadrature

Compute the mean as a weighted sum of the Gauss-Hermite quadrature points.
"""
struct SGSQuadrature{N, A, W} <: AbstractSGSamplingType
    a::A  # values
    w::W  # weights
    function SGSQuadrature(::Type{FT}; quadrature_order = 3) where {FT}
        N = quadrature_order
        # TODO: double check this python-> julia translation
        # a, w = np.polynomial.hermite.hermgauss(N)
        a, w = GQ.hermite(FT, N)
        a, w = SA.SVector{N, FT}(a), SA.SVector{N, FT}(w)
        return new{N, typeof(a), typeof(w)}(a, w)
    end
end
quadrature_order(::SGSQuadrature{N}) where {N} = N

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

Compute the cloud fraction by sampling over the quadrature points, but without
the EDMF sub-grid scale model.
"""
struct QuadratureCloud{SGQ <: AbstractSGSamplingType} <: AbstractCloudModel
    SG_quad::SGQ
end

"""
    SGSQuadratureCloud

Compute the cloud fraction as a sum of the EDMF environment and updraft
contributions. The EDMF environment cloud fraction is computed by sampling over
the quadrature points.
"""
struct SGSQuadratureCloud{SGQ <: AbstractSGSamplingType} <: AbstractCloudModel
    SG_quad::SGQ
end

abstract type AbstractSST end
struct ZonallySymmetricSST <: AbstractSST end
struct ZonallyAsymmetricSST <: AbstractSST end
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
    AbstractOzone

Describe how ozone concentration should be set.
"""
abstract type AbstractOzone end

"""
    IdealizedOzone

Implement a static (not varying in time) idealized ozone profile as described by
`idealized_ozone`.
"""
struct IdealizedOzone <: AbstractOzone end

"""
    PrescribedOzone

Implement a time-varying ozone profile as read from disk.

The CMIP6 forcing dataset is used. For production runs, you should acquire the
high-resolution, multi-year `ozone_concentrations` artifact. If this is not available, a low
resolution, single-year version will be used.

Refer to ClimaArtifacts for more information on how to obtain the artifact.
"""
struct PrescribedOzone <: AbstractOzone end

"""
    AbstractCO2

Describe how CO2 concentration should be set.
"""
abstract type AbstractCO2 end

"""
    FixedCO2

Implement a static CO2 profile as read from disk.

The data used is the one distributed with `RRTMGP.jl`.

By default, this is 397.547 parts per million.

This is the volume mixing ratio.
"""
struct FixedCO2{FT} <: AbstractCO2
    value::FT

    function FixedCO2(; FT = Float64, value = FT(397.547e-6))
        return new{FT}(value)
    end
end

"""
    MuanaLoaCO2

Implement a time-varying CO2 profile as read from disk.

The data from the Mauna Loa CO2 measurements is used. It is a assumed that the
concentration is constant.
"""
struct MaunaLoaCO2 <: AbstractCO2 end

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
Base.@kwdef struct SlabOceanSST{FT} <: AbstractSurfaceTemperature
    # optional slab ocean parameters:
    depth_ocean::FT = 40 # ocean mixed layer depth [m]
    ρ_ocean::FT = 1020 # ocean density [kg / m³]
    cp_ocean::FT = 4184 # ocean heat capacity [J/(kg * K)]
    q_flux::Bool = false # use Q-flux (parameterization of horizontal ocean mixing of energy)
    Q₀::FT = -20 # Q-flux maximum mplitude [W/m²]
    ϕ₀::FT = 16 # Q-flux meridional scale [deg]
end

abstract type AbstractHyperdiffusion end
Base.@kwdef struct ClimaHyperdiffusion{FT} <: AbstractHyperdiffusion
    ν₄_vorticity_coeff::FT
    ν₄_scalar_coeff::FT
    divergence_damping_factor::FT
end

abstract type AbstractVerticalDiffusion end
Base.@kwdef struct VerticalDiffusion{DM, FT} <: AbstractVerticalDiffusion
    C_E::FT
end
disable_momentum_vertical_diffusion(::VerticalDiffusion{DM}) where {DM} = DM
Base.@kwdef struct DecayWithHeightDiffusion{DM, FT} <: AbstractVerticalDiffusion
    H::FT
    D₀::FT
end
disable_momentum_vertical_diffusion(::DecayWithHeightDiffusion{DM}) where {DM} =
    DM
disable_momentum_vertical_diffusion(::Nothing) = false

struct SurfaceFlux end

abstract type AbstractSponge end
Base.Broadcast.broadcastable(x::AbstractSponge) = tuple(x)
Base.@kwdef struct ViscousSponge{FT} <: AbstractSponge
    zd::FT
    κ₂::FT
end

abstract type AbstractEddyViscosityModel end
struct SmagorinskyLilly <: AbstractEddyViscosityModel end

Base.@kwdef struct RayleighSponge{FT} <: AbstractSponge
    zd::FT
    α_uₕ::FT
    α_w::FT
    α_sgs_tracer::FT
end

abstract type AbstractGravityWave end
Base.@kwdef struct NonOrographicGravityWave{FT} <: AbstractGravityWave
    source_pressure::FT = 31500
    damp_pressure::FT = 85
    source_height::FT = 15000
    Bw::FT = 1.0
    Bn::FT = 1.0
    dc::FT = 0.8
    cmax::FT = 99.6
    c0::FT = 0
    nk::FT = 1
    cw::FT = 40.0
    cw_tropics::FT = 40.0
    cn::FT = 40.0
    Bt_0::FT = 0.0003
    Bt_n::FT = 0.0003
    Bt_s::FT = 0.0003
    Bt_eq::FT = 0.0003
    ϕ0_n::FT = 30
    ϕ0_s::FT = -30
    dϕ_n::FT = 5
    dϕ_s::FT = -5
end

Base.@kwdef struct OrographicGravityWave{FT, S} <: AbstractGravityWave
    γ::FT = 0.4
    ϵ::FT = 0.0
    β::FT = 0.5
    h_frac::FT = 0.1
    ρscale::FT = 1.2
    L0::FT = 80e3
    a0::FT = 0.9
    a1::FT = 3.0
    Fr_crit::FT = 0.7
    topo_info::S = "gfdl_restart"
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
Base.@kwdef struct EnvBuoyGradVars{FT, TS}
    ts::TS
    ∂θv∂z_unsat::FT
    ∂qt∂z_sat::FT
    ∂θli∂z_sat::FT
end

function EnvBuoyGradVars(
    ts::TD.ThermodynamicState,
    ∂θv∂z_unsat_∂qt∂z_sat_∂θli∂z_sat,
)
    (; ∂θv∂z_unsat, ∂qt∂z_sat, ∂θli∂z_sat) = ∂θv∂z_unsat_∂qt∂z_sat_∂θli∂z_sat
    return EnvBuoyGradVars(ts, ∂θv∂z_unsat, ∂qt∂z_sat, ∂θli∂z_sat)
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

struct DiagnosticEDMFX{N, TKE, FT} <: AbstractEDMF
    a_half::FT # WARNING: this should never be used outside of `specific`
end
DiagnosticEDMFX{N, TKE}(a_half::FT) where {N, TKE, FT} =
    DiagnosticEDMFX{N, TKE, FT}(a_half)

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

abstract type AbstractSurfaceThermoState end
struct GCMSurfaceThermoState <: AbstractSurfaceThermoState end

abstract type AbstractTendencyModel end
struct UseAllTendency <: AbstractTendencyModel end
struct NoGridScaleTendency <: AbstractTendencyModel end
struct NoSubgridScaleTendency <: AbstractTendencyModel end

# Define broadcasting for types
Base.broadcastable(x::AbstractSurfaceThermoState) = tuple(x)
Base.broadcastable(x::AbstractMoistureModel) = tuple(x)
Base.broadcastable(x::AbstractPrecipitationModel) = tuple(x)
Base.broadcastable(x::AbstractForcing) = tuple(x)
Base.broadcastable(x::EDOnlyEDMFX) = tuple(x)
Base.broadcastable(x::PrognosticEDMFX) = tuple(x)
Base.broadcastable(x::DiagnosticEDMFX) = tuple(x)
Base.broadcastable(x::AbstractEntrainmentModel) = tuple(x)
Base.broadcastable(x::AbstractDetrainmentModel) = tuple(x)
Base.broadcastable(x::AbstractSGSamplingType) = tuple(x)
Base.broadcastable(x::AbstractTendencyModel) = tuple(x)

Base.@kwdef struct RadiationDYCOMS{FT}
    "Large-scale divergence"
    divergence::FT = 3.75e-6
    alpha_z::FT = 1.0
    kappa::FT = 85.0
    F0::FT = 70.0
    F1::FT = 22.0
end

Base.@kwdef struct RadiationISDAC{FT}
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

struct TestDycoreConsistency end

abstract type AbstractTimesteppingMode end
struct Explicit <: AbstractTimesteppingMode end
struct Implicit <: AbstractTimesteppingMode end

struct QuasiMonotoneLimiter end # For dispatching to use the ClimaCore QuasiMonotoneLimiter.

abstract type AbstractScaleBlendingMethod end
struct SmoothMinimumBlending <: AbstractScaleBlendingMethod end
struct HardMinimumBlending <: AbstractScaleBlendingMethod end
Base.broadcastable(x::AbstractScaleBlendingMethod) = tuple(x)

Base.@kwdef struct AtmosNumerics{EN_UP, TR_UP, ED_UP, SG_UP, TDC, LIM, DM, HD}

    """Enable specific upwinding schemes for specific equations"""
    energy_upwinding::EN_UP
    tracer_upwinding::TR_UP
    edmfx_upwinding::ED_UP
    edmfx_sgsflux_upwinding::SG_UP

    """Add NaNs to certain equations to track down problems"""
    test_dycore_consistency::TDC

    limiter::LIM

    """Timestepping mode for diffusion: Explicit() or Implicit()"""
    diff_mode::DM = nothing

    """Hyperdiffusion model: nothing or ClimaHyperdiffusion()"""
    hyperdiff::HD = nothing
end
Base.broadcastable(x::AtmosNumerics) = tuple(x)

function Base.summary(io::IO, numerics::AtmosNumerics)
    pns = string.(propertynames(numerics))
    buf = maximum(length.(pns))
    keys = propertynames(numerics)
    vals = repeat.(" ", map(s -> buf - length(s) + 2, pns))
    bufs = (; zip(keys, vals)...)
    print(io, '\n')
    for pn in propertynames(numerics)
        prop = getproperty(numerics, pn)
        s = string(
            "  ", # needed for some reason
            getproperty(bufs, pn),
            '`',
            string(pn),
            '`',
            "::",
            '`',
            typeof(prop),
            '`',
            '\n',
        )
        print(io, s)
    end
end

const ValTF = Union{Val{true}, Val{false}}

Base.@kwdef struct EDMFXModel{
    EEM,
    EDM,
    ESMF <: ValTF,
    ESDF <: ValTF,
    ENP <: ValTF,
    EVD <: ValTF,
    EF <: ValTF,
    SBM <: AbstractScaleBlendingMethod,
}
    entr_model::EEM = nothing
    detr_model::EDM = nothing
    sgs_mass_flux::ESMF = Val(false)
    sgs_diffusive_flux::ESDF = Val(false)
    nh_pressure::ENP = Val(false)
    vertical_diffusion::EVD = Val(false)
    filter::EF = Val(false)
    scale_blending_method::SBM
end

# Grouped structs to reduce AtmosModel type parameters

"""
    SCMSetup

Groups Single-Column Model and Large-Eddy Simulation specific forcing, advection, and setup models.

These components are primarily used internally for testing, calibration, and research purposes
with single-column model setups. Most external users will not need these components.
"""
Base.@kwdef struct SCMSetup{S, EF, LA, AT, SC}
    subsidence::S = nothing
    external_forcing::EF = nothing
    ls_adv::LA = nothing
    advection_test::AT = nothing
    scm_coriolis::SC = nothing
end

"""
    AtmosWater

Groups moisture-related models and types.
"""
Base.@kwdef struct AtmosWater{MM, PM, CM, NCFM, CCDPS}
    moisture_model::MM = nothing
    microphysics_model::PM = nothing
    cloud_model::CM = nothing
    noneq_cloud_formation_mode::NCFM = nothing
    call_cloud_diagnostics_per_stage::CCDPS = nothing
end

"""
    AtmosRadiation

Groups radiation-related models and types.
"""
Base.@kwdef struct AtmosRadiation{RM, OZ, CO2, IN}
    radiation_mode::RM = nothing
    ozone::OZ = nothing
    co2::CO2 = nothing
    insolation::IN = nothing
end

"""
    AtmosTurbconv

Groups turbulence convection-related models and types.
"""
Base.@kwdef struct AtmosTurbconv{EDMFX, TCM, SAM, SEDM, SNPM, SVM, SMM, SH, SV}
    edmfx_model::EDMFX = nothing
    turbconv_model::TCM = nothing
    sgs_adv_mode::SAM = nothing
    sgs_entr_detr_mode::SEDM = nothing
    sgs_nh_pressure_mode::SNPM = nothing
    sgs_vertdiff_mode::SVM = nothing
    sgs_mf_mode::SMM = nothing
    smagorinsky_horizontal::SH = nothing
    smagorinsky_vertical::SV = nothing
end

"""
    AtmosGravityWave

Groups gravity wave-related models and types.
"""
Base.@kwdef struct AtmosGravityWave{NOGW, OGW}
    non_orographic_gravity_wave::NOGW = nothing
    orographic_gravity_wave::OGW = nothing
end

"""
    AtmosSponge

Groups sponge-related models and types.
"""
Base.@kwdef struct AtmosSponge{VS, RS}
    viscous_sponge::VS = nothing
    rayleigh_sponge::RS = nothing
end

"""
    AtmosSurface

Groups surface-related models and types.
"""
Base.@kwdef struct AtmosSurface{ST, SM, SA}
    sfc_temperature::ST = nothing
    surface_model::SM = nothing
    surface_albedo::SA = nothing
end

# Add broadcastable for the new grouped types
Base.broadcastable(x::SCMSetup) = tuple(x)
Base.broadcastable(x::AtmosWater) = tuple(x)
Base.broadcastable(x::AtmosRadiation) = tuple(x)
Base.broadcastable(x::AtmosTurbconv) = tuple(x)
Base.broadcastable(x::AtmosGravityWave) = tuple(x)
Base.broadcastable(x::AtmosSponge) = tuple(x)
Base.broadcastable(x::AtmosSurface) = tuple(x)

struct AtmosModel{W, SCM, R, TC, GW, VD, SP, SU, NU}
    water::W
    scm_setup::SCM
    radiation::R
    turbconv::TC
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

# Forward property access: atmos.moisture_model → atmos.moisture.moisture_model
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

function Base.summary(io::IO, atmos::AtmosModel)
    pns = string.(propertynames(atmos))
    buf = maximum(length.(pns))
    keys = propertynames(atmos)
    vals = repeat.(" ", map(s -> buf - length(s) + 2, pns))
    bufs = (; zip(keys, vals)...)
    print(io, '\n')
    for pn in propertynames(atmos)
        prop = getproperty(atmos, pn)
        # Skip some data:
        prop isa Bool && continue
        prop isa NTuple && continue
        prop isa Int && continue
        prop isa Float64 && continue
        prop isa Float32 && continue
        s = string(
            "  ", # needed for some reason
            getproperty(bufs, pn),
            '`',
            string(pn),
            '`',
            "::",
            '`',
            typeof(prop),
            '`',
            '\n',
        )
        print(io, s)
    end
end

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
model = AtmosModel(; moisture_model = EquilMoistModel())
model.moisture_model        # Direct access
model.water.moisture_model  # Grouped access
```

# Example: Minimal model (uses defaults)
```julia
model = AtmosModel()  # Creates a basic dry atmospheric model
```

# Example: Dry model with Held-Suarez forcing and hyperdiffusion
```julia
model = AtmosModel(;
    radiation_mode = HeldSuarezForcing(),
    hyperdiff = ClimaHyperdiffusion(; 
        ν₄_vorticity_coeff = 1e15, 
        ν₄_scalar_coeff = 1e15, 
        divergence_damping_factor = 1.0
    )
)
```

# Example: Moist model with full radiation
```julia
model = AtmosModel(;
    moisture_model = EquilMoistModel(),
    microphysics_model = Microphysics0Moment(),
    radiation_mode = RRTMGPI.AllSkyRadiation(),
    ozone = IdealizedOzone(),
    co2 = FixedCO2()
)
```

# Default Configuration
The default AtmosModel provides:
- **Dry atmosphere**: DryModel() with NoPrecipitation()
- **Basic surface**: PrescribedSST() with ZonallySymmetricSST()
- **Simple clouds**: GridScaleCloud()
- **Idealized insolation**: IdealizedInsolation()
- **Conservative numerics**: First-order upwinding with Explicit() timestepping
- **No advanced physics**: No radiation, turbulence, or forcing by default

# Available Structs

## AtmosWater
- `moisture_model`: DryModel(), EquilMoistModel(), NonEquilMoistModel()
- `microphysics_model`: NoPrecipitation(), Microphysics0Moment(), Microphysics1Moment(), Microphysics2Moment()
- `cloud_model`: GridScaleCloud(), QuadratureCloud(), SGSQuadratureCloud()
- `noneq_cloud_formation_mode`: Explicit(), Implicit()
- `call_cloud_diagnostics_per_stage`: nothing or CallCloudDiagnosticsPerStage()

## SCMSetup (Single-Column Model & LES specific - accessed via model.subsidence, model.external_forcing, etc.)
Internal testing and calibration components for single-column setups:
- `subsidence`: nothing or Bomex_subsidence, Rico_subsidence, DYCOMS_subsidence, etc
- `external_forcing`: nothing or external forcing objects (GCMForcing, ExternalDrivenTVForcing, ISDACForcing)
- `ls_adv`: nothing or LargeScaleAdvection()
- `advection_test`: nothing or boolean
- `scm_coriolis`: nothing or SCMCoriolis()

## AtmosRadiation
- `radiation_mode`: Radiation and atmospheric forcing modes
  - Global radiation: RRTMGPI.ClearSkyRadiation(), RRTMGPI.AllSkyRadiation()
  - Atmospheric forcing: HeldSuarezForcing() (for idealized dynamics)
  - SCM-specific: RadiationDYCOMS(), RadiationISDAC(), RadiationTRMM_LBA()
- `ozone`: IdealizedOzone(), PrescribedOzone()
- `co2`: FixedCO2(), MaunaLoaCO2()
- `insolation`: IdealizedInsolation(), TimeVaryingInsolation(), etc.

## AtmosTurbconv
- `edmfx_model`: EDMFXModel()
- `turbconv_model`: nothing, PrognosticEDMFX(), DiagnosticEDMFX(), EDOnlyEDMFX()
- `sgs_adv_mode`, `sgs_entr_detr_mode`, `sgs_nh_pressure_mode`, `sgs_vertdiff_mode`, `sgs_mf_mode`: Explicit(), Implicit()
- `smagorinsky_lilly`: nothing or SmagorinskyLilly()

## AtmosGravityWave
- `non_orographic_gravity_wave`: nothing or NonOrographicGravityWave()  
- `orographic_gravity_wave`: nothing or OrographicGravityWave()

## AtmosSponge
- `viscous_sponge`: nothing or ViscousSponge()
- `rayleigh_sponge`: nothing or RayleighSponge()

## AtmosSurface
- `sfc_temperature`: ZonallySymmetricSST(), ZonallyAsymmetricSST(), RCEMIPIISST(), ExternalTVColumnSST()
- `surface_model`: PrescribedSST(), SlabOceanSST()
- `surface_albedo`: ConstantAlbedo(), RegressionFunctionAlbedo(), CouplerAlbedo()

## AtmosNumerics
- `energy_upwinding`, `tracer_upwinding`, `edmfx_upwinding`, `edmfx_sgsflux_upwinding`: Val() upwinding schemes
- `test_dycore_consistency`: nothing or TestDycoreConsistency() for debugging
- `limiter`: nothing or QuasiMonotoneLimiter()
- `diff_mode`: Explicit(), Implicit() timestepping mode for diffusion
- `hyperdiff`: nothing or ClimaHyperdiffusion()

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

    return AtmosModel{
        typeof(water),
        typeof(scm_setup),
        typeof(radiation),
        typeof(turbconv),
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

const _DEFAULT_ATMOS_MODEL_KWARGS = (
    moisture_model = DryModel(),
    microphysics_model = NoPrecipitation(),
    cloud_model = GridScaleCloud(),
    surface_model = PrescribedSST(),
    sfc_temperature = ZonallySymmetricSST(),
    insolation = IdealizedInsolation(),

    # AtmosNumerics defaults
    energy_upwinding = Val(:first_order),
    tracer_upwinding = Val(:first_order),
    edmfx_upwinding = Val(:first_order),
    edmfx_sgsflux_upwinding = Val(:none),
    test_dycore_consistency = nothing,
    limiter = nothing,
    diff_mode = Explicit(),
    hyperdiff = nothing,

    # Top-level
    disable_surface_flux_tendency = false,
)

"""
    _partition_atmos_model_kwargs(kwargs)

Partition the given kwargs into grouped and direct kwargs matching the AtmosModel struct.

Helper function for the AtmosModel constructor.
"""
function _partition_atmos_model_kwargs(kwargs)

    # Merge default minimal model arguments with given kwargs
    all_kwargs = merge(_DEFAULT_ATMOS_MODEL_KWARGS, kwargs)

    # group_kwargs contains a Dict for each group in ATMOS_MODEL_GROUPS
    group_kwargs = Dict(map(ATMOS_MODEL_GROUPS) do (_, group_field)
        group_field => Dict{Symbol, Any}()
    end)

    # Sort kwargs into a hierarchy of dicts matching the AtmosModel struct
    atmos_model_kwargs = Dict{Symbol, Any}()
    unknown_args = Symbol[]

    for (key, value) in pairs(all_kwargs)
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
    hyperdiff = ClimaHyperdiffusion(; ν₄_vorticity_coeff = 1e15, ν₄_scalar_coeff = 1e15, divergence_damping_factor = 1.0)
)
```
"""
function DryAtmosModel(; kwargs...)
    defaults = (
        moisture_model = DryModel(),
        microphysics_model = NoPrecipitation(),
        cloud_model = GridScaleCloud(),
        surface_model = PrescribedSST(),
        sfc_temperature = ZonallySymmetricSST(),
        insolation = IdealizedInsolation(),
    )
    return AtmosModel(; defaults..., kwargs...)
end

"""
    EquilMoistAtmosModel(; kwargs...)

Create an equilibrium moist atmospheric model with sensible defaults for moist simulations.
"""
function EquilMoistAtmosModel(; kwargs...)
    defaults = (
        moisture_model = EquilMoistModel(),
        microphysics_model = Microphysics0Moment(),
        cloud_model = GridScaleCloud(),
        surface_model = PrescribedSST(),
        sfc_temperature = ZonallySymmetricSST(),
        insolation = IdealizedInsolation(),
        ozone = IdealizedOzone(),
        co2 = FixedCO2(),
    )
    return AtmosModel(; defaults..., kwargs...)
end

"""
    NonEquilMoistAtmosModel(; kwargs...)

Create a non-equilibrium moist atmospheric model with sensible defaults.
"""
function NonEquilMoistAtmosModel(; kwargs...)
    defaults = (
        moisture_model = NonEquilMoistModel(),
        microphysics_model = Microphysics1Moment(),
        cloud_model = GridScaleCloud(),
        noneq_cloud_formation_mode = Explicit(),
        surface_model = PrescribedSST(),
        sfc_temperature = ZonallySymmetricSST(),
        insolation = IdealizedInsolation(),
        ozone = IdealizedOzone(),
        co2 = FixedCO2(),
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
    maybe_resolve_and_acquire_artifacts(input_str::AbstractString, context::ClimaComms.AbstractCommsContext)

When given a string of the form `artifact"name"/something/else`, resolve the
artifact path and download it (if not already available).

In all the other cases, return the input unchanged.
"""
function maybe_resolve_and_acquire_artifacts(
    input_str::AbstractString,
    context::ClimaComms.AbstractCommsContext,
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
    _::ClimaComms.AbstractCommsContext,
)
    return input
end

"""
    config_with_resolved_and_acquired_artifacts(input_str::AbstractString, context::ClimaComms.AbstractCommsContext)

Substitute strings of the form `artifact"name"/something/else` with the actual
artifact path.
"""
function config_with_resolved_and_acquired_artifacts(
    config::AbstractDict,
    context::ClimaComms.AbstractCommsContext,
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
