module ClimaAtmos

include("Parameters.jl")
import .Parameters

include("types.jl")

include("RRTMGPInterface.jl")
import .RRTMGPInterface as RRTMGPI

include("TurbulenceConvection/TurbulenceConvection.jl")
import .TurbulenceConvection as TC

include("model_getters.jl") # high-level (using parsed_args) model getters

end # module
