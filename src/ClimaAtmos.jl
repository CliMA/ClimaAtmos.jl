module ClimaAtmos

using NVTX, Colors

include("Parameters.jl")
import .Parameters as CAP

include(joinpath("utils", "abbreviations.jl"))
include(joinpath("utils", "common_spaces.jl"))
include(joinpath("utils", "types.jl"))
include(joinpath("utils", "nvtx.jl"))
include(joinpath("utils", "cli_options.jl"))
include(joinpath("utils", "utilities.jl"))
include(joinpath("utils", "debug_utils.jl"))
include(joinpath("utils", "classify_case.jl"))
include(joinpath("utils", "topography_helper.jl"))
include(joinpath("utils", "variable_manipulations.jl"))

include(
    joinpath("parameterized_tendencies", "radiation", "radiation_utilities.jl"),
)
include(joinpath("parameterized_tendencies", "radiation", "RRTMGPInterface.jl"))
import .RRTMGPInterface as RRTMGPI
include(joinpath("parameterized_tendencies", "radiation", "radiation.jl"))

include("TurbulenceConvection/TurbulenceConvection.jl")
import .TurbulenceConvection as TC

include("precomputed_quantities.jl")

include(joinpath("InitialConditions", "InitialConditions.jl"))
include(
    joinpath(
        "parameterized_tendencies",
        "TurbulenceConvection",
        "tc_functions.jl",
    ),
)
include(joinpath("SurfaceStates", "SurfaceStates.jl"))
include(joinpath("utils", "discrete_hydrostatic_balance.jl"))

include(joinpath("prognostic_equations", "pressure_work.jl"))
include(joinpath("prognostic_equations", "zero_velocity.jl"))

include(joinpath("prognostic_equations", "implicit", "wfact.jl"))
include(joinpath("prognostic_equations", "implicit", "schur_complement_W.jl"))
include(joinpath("prognostic_equations", "implicit", "implicit_tendency.jl"))

include(joinpath("prognostic_equations", "remaining_tendency.jl"))
include(joinpath("prognostic_equations", "forcing", "large_scale_advection.jl")) # TODO: should this be in tendencies/?
include(joinpath("prognostic_equations", "forcing", "subsidence.jl"))
include(joinpath("parameterized_tendencies", "held_suarez", "held_suarez.jl"))

include(
    joinpath(
        "parameterized_tendencies",
        "gravity_wave_drag",
        "non_orographic_gravity_wave.jl",
    ),
)
include(
    joinpath(
        "parameterized_tendencies",
        "gravity_wave_drag",
        "orographic_gravity_wave_helper.jl",
    ),
)
include(
    joinpath(
        "parameterized_tendencies",
        "gravity_wave_drag",
        "orographic_gravity_wave.jl",
    ),
)
include(joinpath("prognostic_equations", "hyperdiffusion.jl"))
include(joinpath("prognostic_equations", "edmf_coriolis.jl"))
include(joinpath("prognostic_equations", "edmfx_closures.jl"))
include(
    joinpath("parameterized_tendencies", "microphysics", "precipitation.jl"),
)
include(
    joinpath("prognostic_equations", "vertical_diffusion_boundary_layer.jl"),
)
include(joinpath("parameterized_tendencies", "sponge", "rayleigh_sponge.jl"))
include(joinpath("parameterized_tendencies", "sponge", "viscous_sponge.jl"))
include(joinpath("prognostic_equations", "advection.jl"))
include(joinpath("dycore_equations", "sgs_flux_tendencies.jl"))

include("staggered_nonhydrostatic_model.jl")

include(joinpath("callbacks", "callbacks.jl"))

include(joinpath("utils", "model_getters.jl")) # high-level (using parsed_args) model getters
include(joinpath("utils", "type_getters.jl"))
include(joinpath("utils", "yaml_helper.jl"))
include(joinpath("solve.jl"))

end # module
