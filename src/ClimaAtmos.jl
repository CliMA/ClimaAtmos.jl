module ClimaAtmos

include("Parameters.jl")
import .Parameters as CAP

include("types.jl")
include("utilities.jl")

include("RRTMGPInterface.jl")
import .RRTMGPInterface as RRTMGPI

include("TurbulenceConvection/TurbulenceConvection.jl")
import .TurbulenceConvection as TC

include("thermo_state.jl")

include(joinpath("tendencies", "held_suarez.jl"))
include(joinpath("tendencies", "rayleigh_sponge.jl"))
include(joinpath("tendencies", "viscous_sponge.jl"))
include(joinpath("tendencies", "advection.jl"))

include("model_getters.jl") # high-level (using parsed_args) model getters

end # module
