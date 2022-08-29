
"""
Supertype for all model thermodynamics styles.
"""
abstract type AbstractThermodynamicsStyle <: AbstractModelStyle end

struct PotentialTemperature <: AbstractThermodynamicsStyle end
struct TotalEnergy <: AbstractThermodynamicsStyle end
struct InternalEnergy <: AbstractThermodynamicsStyle end

Models.variable_names(::PotentialTemperature) = (:ρθ,)
Models.variable_names(::TotalEnergy) = (:ρe_tot,)
Models.variable_names(::InternalEnergy) = (:ρe_int,)

Models.variable_types(::PotentialTemperature, ::AbstractModel, FT) = (ρθ = FT,)
Models.variable_types(::TotalEnergy, ::AbstractModel, FT) = (ρe_tot = FT,)
Models.variable_types(::InternalEnergy, ::AbstractModel, FT) = (ρe_int = FT,)

Models.variable_spaces(::PotentialTemperature, ::AbstractSingleColumnModel) =
    (ρθ = Spaces.FiniteDifferenceSpace{Spaces.CellCenter},)
Models.variable_spaces(::TotalEnergy, ::AbstractSingleColumnModel) =
    (ρe_tot = Spaces.FiniteDifferenceSpace{Spaces.CellCenter},)
Models.variable_spaces(::InternalEnergy, ::AbstractSingleColumnModel) =
    (ρe_int = Spaces.FiniteDifferenceSpace{Spaces.CellCenter},)
Models.variable_spaces(
    ::PotentialTemperature,
    ::Union{AbstractNonhydrostatic2DModel, AbstractNonhydrostatic3DModel},
) = (ρθ = Spaces.ExtrudedFiniteDifferenceSpace{Spaces.CellCenter},)
Models.variable_spaces(
    ::TotalEnergy,
    ::Union{AbstractNonhydrostatic2DModel, AbstractNonhydrostatic3DModel},
) = (ρe_tot = Spaces.ExtrudedFiniteDifferenceSpace{Spaces.CellCenter},)
Models.variable_spaces(
    ::InternalEnergy,
    ::Union{AbstractNonhydrostatic2DModel, AbstractNonhydrostatic3DModel},
) = (ρe_int = Spaces.ExtrudedFiniteDifferenceSpace{Spaces.CellCenter},)
