module Models

export AbstractModel, default_initial_conditions, make_ode_function

"""
Supertype for all models.
"""
abstract type AbstractModel end

"""
    default_initial_conditions(model)

Construct the initial conditions for `model`.
"""
default_initial_conditions(model::AbstractModel) =
    error("default_initial_conditions not implemented for given model type")

"""
    make_ode_function(model)

Construct the ordinary differential equations for `model`.
"""
make_ode_function(model::AbstractModel) =
    error("make_ode_function not implemented for given model type")

include("ShallowWaterModels/ShallowWaterModels.jl")
include("SingleColumnModels/SingleColumnModels.jl")
include("Nonhydrostatic2DModels/Nonhydrostatic2DModels.jl")

end # module
