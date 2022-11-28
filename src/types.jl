abstract type AbstractMoistureModel end
struct DryModel <: AbstractMoistureModel end
struct EquilMoistModel <: AbstractMoistureModel end
struct NonEquilMoistModel <: AbstractMoistureModel end

abstract type AbstractCompressibilityModel end
struct CompressibleFluid <: AbstractCompressibilityModel end
struct AnelasticFluid <: AbstractCompressibilityModel end

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
Base.@kwdef struct ThermoDispatcher{EF, MM, CM}
    energy_form::EF
    moisture_model::MM
    compressibility_model::CM
end
Base.broadcastable(x::ThermoDispatcher) = Ref(x)


Base.@kwdef struct AtmosModel{
    MC,
    C,
    MM,
    EF,
    PM,
    F,
    S,
    RM,
    LA,
    EC,
    TCM,
    CM,
    SS,
    GW,
}
    model_config::MC = nothing
    coupling::C = nothing
    moisture_model::MM = nothing
    energy_form::EF = nothing
    precip_model::PM = nothing
    forcing_type::F = nothing
    subsidence::S = nothing
    radiation_mode::RM = nothing
    ls_adv::LA = nothing
    edmf_coriolis::EC = nothing
    turbconv_model::TCM = nothing
    compressibility_model::CM = nothing
    surface_scheme::SS = nothing
    non_orographic_gravity_wave::GW = nothing
end

Base.broadcastable(x::AtmosModel) = Ref(x)

is_compressible(atmos::AtmosModel) =
    atmos.compressibility_model isa CompressibleFluid
is_anelastic(atmos::AtmosModel) = atmos.compressibility_model isa AnelasticFluid
is_column(atmos::AtmosModel) = atmos.model_config isa SingleColumnModel
is_anelastic_column(atmos::AtmosModel) = is_anelastic(atmos) && is_column(atmos)

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
