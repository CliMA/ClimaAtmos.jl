module Models

"""
    AbstractModel
"""
abstract type AbstractModel end

include("model_interface.jl")
include("ShallowWaterModels/ShallowWaterModels.jl")
include("SingleColumnModels/SingleColumnModels.jl")

export AbstractModel

export default_initial_conditions
export make_ode_function
export get_boundary_flux

end # module
