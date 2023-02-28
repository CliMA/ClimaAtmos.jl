abstract type AbstractMoistureModel end
struct DryModel <: AbstractMoistureModel end
struct EquilMoistModel <: AbstractMoistureModel end
struct NonEquilMoistModel <: AbstractMoistureModel end

abstract type AbstractEnergyFormulation end
struct PotentialTemperature <: AbstractEnergyFormulation end
struct TotalEnergy <: AbstractEnergyFormulation end
struct InternalEnergy <: AbstractEnergyFormulation end

abstract type AbstractPrecipitationModel end
struct NoPrecipitation <: AbstractPrecipitationModel end
struct Microphysics0Moment <: AbstractPrecipitationModel end
struct Microphysics1Moment <: AbstractPrecipitationModel end

abstract type AbstractModelConfig end
struct SingleColumnModel <: AbstractModelConfig end
struct SphericalModel <: AbstractModelConfig end
struct BoxModel <: AbstractModelConfig end

abstract type AbstractCoupling end
struct Coupled <: AbstractCoupling end
struct Decoupled <: AbstractCoupling end

abstract type AbstractHyperdiffusion end
Base.@kwdef struct ClimaHyperdiffusion{B, FT} <: AbstractHyperdiffusion
    κ₄::FT
    divergence_damping_factor::FT
end
Base.@kwdef struct TempestHyperdiffusion{B, FT} <: AbstractHyperdiffusion
    κ₄::FT
    divergence_damping_factor::FT
end

q_tot_hyperdiffusion_enabled(::ClimaHyperdiffusion{B}) where {B} = B
q_tot_hyperdiffusion_enabled(::TempestHyperdiffusion{B}) where {B} = B

abstract type AbstractVerticalDiffusion end
Base.@kwdef struct VerticalDiffusion{DM, FT} <: AbstractVerticalDiffusion
    C_E::FT
end
diffuse_momentum(::VerticalDiffusion{DM}) where {DM} = DM
diffuse_momentum(::Nothing) = false

abstract type AbstractSponge end
Base.@kwdef struct ViscousSponge{FT} <: AbstractSponge
    zd::FT
    κ₂::FT
end

Base.@kwdef struct RayleighSponge{FT} <: AbstractSponge
    zd::FT
    α_uₕ::FT
    α_w::FT
end

abstract type AbstractGravityWave end
Base.@kwdef struct NonOrographyGravityWave{FT} <: AbstractGravityWave
    source_pressure::FT = 31500
    damp_pressure::FT = 85
    source_height::FT = 15000
    Bw::FT = 1.0
    Bn::FT = 1.0
    dc::FT = 0.6
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

Base.@kwdef struct OrographicGravityWave{FT} <: AbstractGravityWave
    γ::FT = 0.4
    ϵ::FT = 0.0
    β::FT = 0.5
    ρscale::FT = 1.2
    L0::FT = 80e3
    a0::FT = 0.9
    a1::FT = 3.0
    Fr_crit::FT = 0.7
end

abstract type AbstractForcing end
struct HeldSuarezForcing <: AbstractForcing end
struct Subsidence{T} <: AbstractForcing
    prof::T
end
# TODO: is this a forcing?
struct LargeScaleAdvection{PT, PQ}
    prof_dTdt::PT # Set large-scale cooling
    prof_dqtdt::PQ # Set large-scale drying
end

struct EDMFCoriolis{U, V, FT}
    prof_ug::U
    prof_vg::V
    coriolis_param::FT
end

abstract type AbstractSurfaceThermoState end
struct GCMSurfaceThermoState <: AbstractSurfaceThermoState end

abstract type AbstractSurfaceScheme end
struct BulkSurfaceScheme{T} <: AbstractSurfaceScheme
    sfc_thermo_state_type::T
end
struct MoninObukhovSurface{T} <: AbstractSurfaceScheme
    sfc_thermo_state_type::T
end # TODO: unify with MoninObukhovSurface in TC

# Define broadcasting for types
Base.broadcastable(x::AbstractSurfaceThermoState) = Ref(x)
Base.broadcastable(x::AbstractSurfaceScheme) = Ref(x)
Base.broadcastable(x::AbstractMoistureModel) = Ref(x)
Base.broadcastable(x::AbstractEnergyFormulation) = Ref(x)
Base.broadcastable(x::AbstractPrecipitationModel) = Ref(x)
Base.broadcastable(x::AbstractForcing) = Ref(x)

Base.@kwdef struct RadiationDYCOMS_RF01{FT}
    "Large-scale divergence"
    divergence::FT = 3.75e-6
    alpha_z::FT = 1.0
    kappa::FT = 85.0
    F0::FT = 70.0
    F1::FT = 22.0
end
import AtmosphericProfilesLibrary as APL

struct RadiationTRMM_LBA{R}
    rad_profile::R
    function RadiationTRMM_LBA(::Type{FT}) where {FT}
        rad_profile = APL.TRMM_LBA_radiation(FT)
        return new{typeof(rad_profile)}(rad_profile)
    end
end

"""
    ThermoDispatcher

A dispatching type for selecting the
precise thermodynamics method call to
be used.
"""
Base.@kwdef struct ThermoDispatcher{EF, MM}
    energy_form::EF
    moisture_model::MM
end
Base.broadcastable(x::ThermoDispatcher) = Ref(x)


# TODO: remove AbstractPerformanceMode and all subtypes
# This is temporarily needed to investigate performance of
# our handling of tracers.
abstract type AbstractPerformanceMode end
struct PerfExperimental <: AbstractPerformanceMode end
struct PerfStandard <: AbstractPerformanceMode end
Base.broadcastable(x::AbstractPerformanceMode) = Ref(x)

Base.@kwdef struct AtmosModel{
    MC,
    C,
    PEM,
    MM,
    EF,
    PM,
    F,
    S,
    RM,
    LA,
    EC,
    TCM,
    SS,
    NOGW,
    OGW,
    HD,
    VD,
    VS,
    RS,
}
    model_config::MC = nothing
    coupling::C = nothing
    perf_mode::PEM = nothing
    moisture_model::MM = nothing
    energy_form::EF = nothing
    precip_model::PM = nothing
    forcing_type::F = nothing
    subsidence::S = nothing
    radiation_mode::RM = nothing
    ls_adv::LA = nothing
    edmf_coriolis::EC = nothing
    turbconv_model::TCM = nothing
    surface_scheme::SS = nothing
    non_orographic_gravity_wave::NOGW = nothing
    orographic_gravity_wave::OGW = nothing
    hyperdiff::HD = nothing
    vert_diff::VD = nothing
    viscous_sponge::VS = nothing
    rayleigh_sponge::RS = nothing
end

Base.broadcastable(x::AtmosModel) = Ref(x)

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
