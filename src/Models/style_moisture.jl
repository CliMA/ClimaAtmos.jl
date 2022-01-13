
"""
Supertype for all model moisture styles.
"""
abstract type AbstractMoistureStyle <: AbstractModelStyle end

struct Dry <: AbstractMoistureStyle end
struct EquilibriumMoisture <: AbstractMoistureStyle end
struct NonEquilibriumMoisture <: AbstractMoistureStyle end

Models.state_variable_names(::Dry) = nothing
Models.state_variable_names(::EquilibriumMoisture) = (:ρq_tot,)
Models.state_variable_names(::NonEquilibriumMoisture) =
    (:ρq_tot, :ρq_liq, :ρq_ice)
