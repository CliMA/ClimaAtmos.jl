
"""
Supertype for all model thermodynamics styles.
"""
abstract type AbstractThermodynamicsStyle <: AbstractModelStyle end

struct PotentialTemperature <: AbstractThermodynamicsStyle end
struct TotalEnergy <: AbstractThermodynamicsStyle end

Models.variable_names(::PotentialTemperature) = (:ρθ,)
Models.variable_names(::TotalEnergy) = (:ρe_tot,)

Models.variable_types(::PotentialTemperature, ::AbstractModel, FT) = (ρθ = FT,)
Models.variable_types(::TotalEnergy, ::AbstractModel, FT) = (ρe_tot = FT,)

Models.variable_spaces(::PotentialTemperature) =
    (ρθ = Spaces.ExtrudedFiniteDifferenceSpace{Spaces.CellCenter},)
Models.variable_spaces(::TotalEnergy) =
    (ρe_tot = Spaces.ExtrudedFiniteDifferenceSpace{Spaces.CellCenter},)
