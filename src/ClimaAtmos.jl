module ClimaAtmos

using NVTX, Colors

include("Parameters.jl")
import .Parameters as CAP

include(joinpath("utils", "abbreviations.jl"))
include(joinpath("utils", "common_spaces.jl"))
include(joinpath("utils", "types.jl"))
include(joinpath("utils", "utilities.jl"))
include(joinpath("utils", "debug_utils.jl"))
include(joinpath("utils", "classify_case.jl"))
include(joinpath("utils", "topography_helper.jl"))
include(joinpath("utils", "variable_manipulations.jl"))

include(joinpath("parameterizations", "radiation", "radiation_utilities.jl"))
include(joinpath("parameterizations", "radiation", "RRTMGPInterface.jl"))
import .RRTMGPInterface as RRTMGPI
include(joinpath("parameterizations", "radiation", "radiation.jl"))

include("TurbulenceConvection/TurbulenceConvection.jl")
import .TurbulenceConvection as TC

include("precomputed_quantities.jl")

include(joinpath("InitialConditions", "InitialConditions.jl"))
include(joinpath("utils", "discrete_hydrostatic_balance.jl"))

include(joinpath("tendencies", "implicit", "wfact.jl"))
include(joinpath("tendencies", "implicit", "schur_complement_W.jl"))
include(joinpath("tendencies", "implicit", "implicit_tendency.jl"))

include(joinpath("tendencies", "forcing", "large_scale_advection.jl")) # TODO: should this be in tendencies/?
include(joinpath("tendencies", "forcing", "subsidence.jl"))
include(joinpath("parameterizations", "held_suarez", "held_suarez.jl"))

include(
    joinpath(
        "parameterizations",
        "gravity_wave_drag",
        "non_orographic_gravity_wave.jl",
    ),
)
include(
    joinpath(
        "parameterizations",
        "gravity_wave_drag",
        "orographic_gravity_wave_helper.jl",
    ),
)
include(
    joinpath(
        "parameterizations",
        "gravity_wave_drag",
        "orographic_gravity_wave.jl",
    ),
)
include(joinpath("tendencies", "hyperdiffusion.jl"))
include(joinpath("tendencies", "edmf_coriolis.jl"))
include(joinpath("parameterizations", "microphysics", "precipitation.jl"))
include(joinpath("tendencies", "vertical_diffusion_boundary_layer.jl"))
include(joinpath("parameterizations", "sponge", "rayleigh_sponge.jl"))
include(joinpath("parameterizations", "sponge", "viscous_sponge.jl"))
include(joinpath("tendencies", "advection.jl"))
include(joinpath("dycore_equations", "sgs_flux_tendencies.jl"))

include("staggered_nonhydrostatic_model.jl")

include(joinpath("utils", "surface.jl"))

include(joinpath("callbacks", "callbacks.jl"))

include(joinpath("utils", "model_getters.jl")) # high-level (using parsed_args) model getters
include(joinpath("utils", "type_getters.jl"))
include(joinpath("utils", "yaml_helper.jl"))

end # module
