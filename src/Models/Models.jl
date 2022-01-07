module Models

export AbstractModel,
    default_initial_conditions, make_ode_function, state_variable_names

export AbstractModelStyle,
    AbstractBaseModelStyle,
    AbstractThermodynamicsStyle,
    AbstractMoistureStyle,
    AdvectiveForm,
    ConservativeForm,
    PotentialTemperature,
    TotalEnergy,
    Dry,
    EquilibriumMoisture,
    NonEquilibriumMoisture

"""
Supertype for all models.
"""
abstract type AbstractModel end

"""
Supertype for all model styles.
"""
abstract type AbstractModelStyle end

"""
    default_initial_conditions(model)

Construct the initial conditions for `model`.
"""
default_initial_conditions(model::AbstractModel) =
    error("default_initial_conditions not implemented for given model type")

"""
    subcomponents(model::AbstractModel)

Return the subcomponents of `model`, e.g. `base`, `thermodynamics`, `moisture`, etc.
"""
subcomponents(model::AbstractModel) =
    error("subcomponents not implemented for given model type")

"""
    state_variable_names(model::AbstractModel)

Construct the state variable names for `model`.
"""
state_variable_names(model::AbstractModel) =
    error("state_variable_names not implemented for given model type")

"""
    state_variable_names(style::AbstractModelStyle)

Construct the state variable names for `style`.
# Example
```jldoctest; setup = :(using ClimaAtmos.Models)
julia> Models.state_variable_names(EquilibriumMoisture())
(:ρq_tot,)
```
"""
state_variable_names(style::AbstractModelStyle) =
    error("state_variable_names not implemented for given model style")

"""
    make_ode_function(model)

Construct the ordinary differential equations for `model`.
"""
make_ode_function(model::AbstractModel) =
    error("make_ode_function not implemented for given model type")

# model styles
include("style_base_model.jl")
include("style_thermodynamics.jl")
include("style_moisture.jl")

# models
include("SingleColumnModels/SingleColumnModels.jl")
include("Nonhydrostatic2DModels/Nonhydrostatic2DModels.jl")
include("Nonhydrostatic3DModels/Nonhydrostatic3DModels.jl")

end # module
