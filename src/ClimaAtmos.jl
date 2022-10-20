module ClimaAtmos

include("Parameters.jl")
import .Parameters

include("types.jl")
include("utilities.jl")

include("RRTMGPInterface.jl")
import .RRTMGPInterface as RRTMGPI

include("TurbulenceConvection/TurbulenceConvection.jl")
import .TurbulenceConvection as TC

include(joinpath("tendencies", "viscous_sponge.jl"))
include(joinpath("tendencies", "advection.jl"))

include("model_getters.jl") # high-level (using parsed_args) model getters

end # module
