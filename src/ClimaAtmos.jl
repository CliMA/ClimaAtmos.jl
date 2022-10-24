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

include(joinpath("tendencies", "forcing", "subsidence.jl"))
include(joinpath("tendencies", "forcing", "held_suarez.jl"))

include(joinpath("tendencies", "microphysics.jl"))
include(joinpath("tendencies", "vertical_diffusion_boundary_layer.jl"))
include(joinpath("tendencies", "rayleigh_sponge.jl"))
include(joinpath("tendencies", "viscous_sponge.jl"))
include(joinpath("tendencies", "advection.jl"))

include(joinpath("tendencies", "implicit_tendency.jl"))

include("model_getters.jl") # high-level (using parsed_args) model getters

end # module
