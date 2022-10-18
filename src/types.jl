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

abstract type AbstractSurfaceScheme end
struct BulkSurfaceScheme <: AbstractSurfaceScheme end
struct MoninObukhovSurface <: AbstractSurfaceScheme end # TODO: unify with MoninObukhovSurface in TC

# Define broadcasting for types
Base.broadcastable(x::AbstractSurfaceScheme) = Ref(x)
Base.broadcastable(x::AbstractMoistureModel) = Ref(x)
Base.broadcastable(x::AbstractEnergyFormulation) = Ref(x)
Base.broadcastable(x::AbstractMicrophysicsModel) = Ref(x)
Base.broadcastable(x::AbstractForcing) = Ref(x)

struct RadiationNone end
struct RadiationDYCOMS_RF01 end
struct RadiationTRMM_LBA end
