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

abstract type AbstractMicrophysicsModel end
struct Microphysics0Moment <: AbstractMicrophysicsModel end

abstract type AbstractForcing end
struct HeldSuarezForcing <: AbstractForcing end
struct Subsidence{T} <: AbstractForcing
    prof::T
end

abstract type AbstractSurfaceScheme end
struct BulkSurfaceScheme <: AbstractSurfaceScheme end
struct MoninObukhovSurface <: AbstractSurfaceScheme end # TODO: unify with MoninObukhovSurface in TC

# Define broadcasting for types
Base.broadcastable(x::AbstractSurfaceScheme) = Ref(x)
Base.broadcastable(x::AbstractMoistureModel) = Ref(x)
Base.broadcastable(x::AbstractEnergyFormulation) = Ref(x)
Base.broadcastable(x::AbstractMicrophysicsModel) = Ref(x)
Base.broadcastable(x::AbstractForcing) = Ref(x)

Base.@kwdef struct RadiationDYCOMS_RF01{FT}
    "Large-scale divergence (same as in ForcingBase)"
    # TODO: make sure in sync with large_scale_divergence
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
