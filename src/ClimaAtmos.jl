module ClimaAtmos

include("Parameters.jl")
import .Parameters as CAP

include("types.jl")
include("utilities.jl")

include("RRTMGPInterface.jl")
import .RRTMGPInterface as RRTMGPI

include("TurbulenceConvection/TurbulenceConvection.jl")
import .TurbulenceConvection as TC

include("ref_state.jl")
include("thermo_state.jl")
include("precomputed_quantities.jl")

include(joinpath("tendencies", "forcing", "large_scale_advection.jl")) # TODO: should this be in tendencies/?
include(joinpath("tendencies", "forcing", "subsidence.jl"))
include(joinpath("tendencies", "forcing", "held_suarez.jl"))

include(joinpath("tendencies", "hyperdiffusion.jl"))
include(joinpath("tendencies", "edmf_coriolis.jl"))
include(joinpath("tendencies", "precipitation.jl"))
include(joinpath("tendencies", "vertical_diffusion_boundary_layer.jl"))
include(joinpath("tendencies", "rayleigh_sponge.jl"))
include(joinpath("tendencies", "viscous_sponge.jl"))
include(joinpath("tendencies", "advection.jl"))

include(joinpath("tendencies", "implicit_tendency.jl"))

include("model_getters.jl") # high-level (using parsed_args) model getters

end # module
