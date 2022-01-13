
"""
Supertype for all base model styles.
"""
abstract type AbstractBaseModelStyle <: AbstractModelStyle end

struct AdvectiveForm <: AbstractBaseModelStyle end
struct ConservativeForm <: AbstractBaseModelStyle end

Models.state_variable_names(::AdvectiveForm) = (:ρ, :uh, :w)
Models.state_variable_names(::ConservativeForm) = (:ρ, :ρuh, :ρw)
