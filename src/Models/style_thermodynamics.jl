
"""
Supertype for all model thermodynamics styles.
"""
abstract type AbstractThermodynamicsStyle <: AbstractModelStyle end

struct PotentialTemperature <: AbstractThermodynamicsStyle end
struct TotalEnergy <: AbstractThermodynamicsStyle end

Models.state_variable_names(::PotentialTemperature) = (:ρθ,)
Models.state_variable_names(::TotalEnergy) = (:ρe_tot,)
