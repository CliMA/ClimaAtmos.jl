module ClimaAtmos

using NVTX
import DiffEqBase # TODO: Get rid of this dependency.
import LinearAlgebra
import Thermodynamics as TD

include(joinpath("parameters", "Parameters.jl"))
import .Parameters as CAP

include(joinpath("utils", "abbreviations.jl"))
include(joinpath("utils", "gpu_compat.jl"))
include(joinpath("utils", "common_spaces.jl"))
include(joinpath("solver", "types.jl"))
include(joinpath("solver", "cli_options.jl"))
include(joinpath("utils", "utilities.jl"))
include(joinpath("utils", "debug_utils.jl"))
include(joinpath("topography", "topography.jl"))
include(joinpath("utils", "variable_manipulations.jl"))
include(joinpath("utils", "read_gcm_driven_scm_data.jl"))

include(joinpath("utils", "AtmosArtifacts.jl"))
import .AtmosArtifacts as AA

include(
    joinpath("parameterized_tendencies", "radiation", "radiation_utilities.jl"),
)
include(joinpath("parameterized_tendencies", "radiation", "RRTMGPInterface.jl"))
import .RRTMGPInterface as RRTMGPI
include(joinpath("parameterized_tendencies", "radiation", "radiation.jl"))

include(joinpath("cache", "prognostic_edmf_precomputed_quantities.jl"))
include(joinpath("cache", "diagnostic_edmf_precomputed_quantities.jl"))
include(joinpath("cache", "precipitation_precomputed_quantities.jl"))
include(joinpath("cache", "precomputed_quantities.jl"))
include(joinpath("cache", "cloud_fraction.jl"))
include(joinpath("cache", "surface_albedo.jl"))

include(joinpath("initial_conditions", "InitialConditions.jl"))
include(
    joinpath(
        "parameterized_tendencies",
        "turbulence_convection",
        "tc_functions.jl",
    ),
)
include(joinpath("surface_conditions", "SurfaceConditions.jl"))
include(joinpath("utils", "discrete_hydrostatic_balance.jl"))

include(joinpath("prognostic_equations", "pressure_work.jl"))
include(joinpath("prognostic_equations", "zero_velocity.jl"))
include(joinpath("prognostic_equations", "zero_tendency.jl"))

include(joinpath("prognostic_equations", "implicit", "implicit_tendency.jl"))
include(joinpath("prognostic_equations", "implicit", "implicit_solver.jl"))
include(joinpath("prognostic_equations", "implicit", "approx_jacobian.jl"))
include(joinpath("prognostic_equations", "implicit", "exact_jacobian.jl"))
include(joinpath("prognostic_equations", "implicit", "debug_jacobian.jl"))
include(joinpath("prognostic_equations", "implicit", "dual_fixes.jl"))

include(joinpath("prognostic_equations", "remaining_tendency.jl"))
include(joinpath("prognostic_equations", "forcing", "large_scale_advection.jl")) # TODO: should this be in tendencies/?
include(joinpath("prognostic_equations", "forcing", "subsidence.jl"))
include(joinpath("prognostic_equations", "forcing", "external_forcing.jl"))

include(joinpath("prognostic_equations", "surface_temp.jl"))

include(joinpath("parameterized_tendencies", "radiation", "held_suarez.jl"))

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
include(joinpath("prognostic_equations", "gm_sgs_closures.jl"))
include(joinpath("prognostic_equations", "edmf_coriolis.jl"))
include(joinpath("prognostic_equations", "buoyancy_gradients.jl"))
include(joinpath("prognostic_equations", "edmfx_closures.jl"))
include(joinpath("prognostic_equations", "edmfx_entr_detr.jl"))
include(joinpath("prognostic_equations", "edmfx_tke.jl"))
include(joinpath("prognostic_equations", "edmfx_sgs_flux.jl"))
include(joinpath("prognostic_equations", "edmfx_boundary_condition.jl"))
include(joinpath("prognostic_equations", "edmfx_precipitation.jl"))
include(
    joinpath("parameterized_tendencies", "microphysics", "precipitation.jl"),
)
include(
    joinpath("parameterized_tendencies", "microphysics", "cloud_condensate.jl"),
)
include(
    joinpath(
        "parameterized_tendencies",
        "microphysics",
        "microphysics_wrappers.jl",
    ),
)
include(
    joinpath("prognostic_equations", "vertical_diffusion_boundary_layer.jl"),
)
include(joinpath("parameterized_tendencies", "sponge", "rayleigh_sponge.jl"))
include(joinpath("parameterized_tendencies", "sponge", "viscous_sponge.jl"))
include(joinpath("prognostic_equations", "advection.jl"))

include(joinpath("cache", "temporary_quantities.jl"))
include(joinpath("cache", "tracer_cache.jl"))
include(joinpath("cache", "cache.jl"))
include(joinpath("prognostic_equations", "dss.jl"))
include(joinpath("prognostic_equations", "limited_tendencies.jl"))

include(joinpath("callbacks", "callbacks.jl"))

include(joinpath("diagnostics", "Diagnostics.jl"))
import .Diagnostics as CAD

include(joinpath("callbacks", "get_callbacks.jl"))

include(joinpath("simulation", "AtmosSimulations.jl"))

include(joinpath("solver", "model_getters.jl")) # high-level (using parsed_args) model getters
include(joinpath("solver", "type_getters.jl"))
include(joinpath("solver", "yaml_helper.jl"))
include(joinpath("solver", "solve.jl"))

include(joinpath("parameters", "create_parameters.jl"))

end # module
