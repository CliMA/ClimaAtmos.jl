module Models

"""
    AbstractModel
"""
abstract type AbstractModel end

include("model_interface.jl")

export AbstractModel

export make_initial_conditions
export make_ode_function
export get_boundary_flux

end # module
