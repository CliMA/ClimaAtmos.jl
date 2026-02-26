module ClimaAtmos

using NVTX
import Adapt
import LinearAlgebra
import NullBroadcasts: NullBroadcasted
import LazyBroadcast
import LazyBroadcast: lazy
import Thermodynamics as TD
import Thermodynamics
import ClimaCore.MatrixFields: @name


include("compat.jl")
include(joinpath("parameters", "Parameters.jl"))
import .Parameters as CAP

include(joinpath("utils", "abbreviations.jl"))
include(joinpath("utils", "gpu_compat.jl"))
include(joinpath("solver", "types.jl"))
include(joinpath("solver", "cli_options.jl"))
include(joinpath("utils", "utilities.jl"))
include(joinpath("utils", "debug_utils.jl"))
include(joinpath("utils", "variable_manipulations.jl"))
include(joinpath("utils", "read_gcm_driven_scm_data.jl"))
include(joinpath("utils", "era5_observations_to_forcing_file.jl"))
include(joinpath("utils", "weather_model.jl"))

include(joinpath("utils", "AtmosArtifacts.jl"))
import .AtmosArtifacts as AA

include(joinpath("topography", "topography.jl"))
include(joinpath("topography", "steady_state_solutions.jl"))

include(joinpath("parameterized_tendencies", "radiation", "RRTMGPInterface.jl"))
import .RRTMGPInterface as RRTMGPI
include(joinpath("parameterized_tendencies", "radiation", "radiation.jl"))

include(joinpath("cache", "prognostic_edmf_precomputed_quantities.jl"))
include(joinpath("cache", "diagnostic_edmf_precomputed_quantities.jl"))
include(joinpath("cache", "microphysics_cache.jl"))
include(joinpath("cache", "precomputed_quantities.jl"))
include(joinpath("cache", "surface_albedo.jl"))

# Microphysics module (SGS quadrature, cloud fraction, tendency limiters, wrappers)
include(joinpath("parameterized_tendencies", "microphysics", "microphysics.jl"))

include(joinpath("initial_conditions", "InitialConditions.jl"))
include(joinpath("surface_conditions", "SurfaceConditions.jl"))
include(joinpath("utils", "refstate_thermodynamics.jl"))

include(joinpath("prognostic_equations", "pressure_work.jl"))
include(joinpath("prognostic_equations", "zero_velocity.jl"))

include(joinpath("prognostic_equations", "implicit", "implicit_tendency.jl"))
include(joinpath("prognostic_equations", "implicit", "jacobian.jl"))
include(
    joinpath("prognostic_equations", "implicit", "manual_sparse_jacobian.jl"),
)
include(joinpath("prognostic_equations", "implicit", "auto_dense_jacobian.jl"))
include(joinpath("prognostic_equations", "implicit", "auto_sparse_jacobian.jl"))
include(joinpath("prognostic_equations", "implicit", "autodiff_utils.jl"))

include(joinpath("prognostic_equations", "water_advection.jl"))
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
include(joinpath("prognostic_equations", "scm_coriolis.jl"))
include(joinpath("prognostic_equations", "eddy_diffusion_closures.jl"))
include(joinpath("prognostic_equations", "mass_flux_closures.jl"))
include(joinpath("prognostic_equations", "edmfx_entr_detr.jl"))
include(joinpath("prognostic_equations", "edmfx_tke.jl"))
include(joinpath("prognostic_equations", "edmfx_sgs_flux.jl"))
include(joinpath("prognostic_equations", "edmfx_boundary_condition.jl"))
include(joinpath("prognostic_equations", "edmfx_microphysics.jl"))
include(
    joinpath("prognostic_equations", "vertical_diffusion_boundary_layer.jl"),
)
include(joinpath("prognostic_equations", "surface_flux.jl"))
include(joinpath("parameterized_tendencies", "sponge", "rayleigh_sponge.jl"))
include(joinpath("parameterized_tendencies", "sponge", "viscous_sponge.jl"))
include(
    joinpath(
        "parameterized_tendencies",
        "les_sgs_models",
        "smagorinsky_lilly.jl",
    ),
)
include(
    joinpath(
        "parameterized_tendencies",
        "les_sgs_models",
        "anisotropic_minimum_dissipation.jl",
    ),
)
include(
    joinpath(
        "parameterized_tendencies",
        "les_sgs_models",
        "constant_horizontal_diffusion.jl",
    ),
)
include(joinpath("prognostic_equations", "advection.jl"))

include(joinpath("cache", "temporary_quantities.jl"))
include(joinpath("cache", "tracer_cache.jl"))
include(joinpath("cache", "cache.jl"))
include(joinpath("cache", "eddy_diffusivity_coefficient.jl"))
include(joinpath("prognostic_equations", "constrain_state.jl"))
include(joinpath("prognostic_equations", "limited_tendencies.jl"))

include(joinpath("callbacks", "callbacks.jl"))

include(joinpath("diagnostics", "Diagnostics.jl"))
import .Diagnostics as CAD

include(joinpath("callbacks", "get_callbacks.jl"))

include(joinpath("parameters", "create_parameters.jl"))

include(joinpath("simulation", "grids.jl"))
include(joinpath("simulation", "AtmosSimulations.jl"))

include(joinpath("solver", "model_getters.jl")) # high-level (using parsed_args) model getters
include(joinpath("solver", "type_getters.jl"))
include(joinpath("solver", "yaml_helper.jl"))
include(joinpath("solver", "solve.jl"))

include(joinpath("utils", "show.jl"))

end # module
