module ClimaAtmos

using NVTX
import Adapt
import LinearAlgebra
import NullBroadcasts: NullBroadcasted
import LazyBroadcast
import LazyBroadcast: lazy
import Thermodynamics as TD
import Thermodynamics
import ClimaCore
import ClimaCore.MatrixFields: @name

import ClimaUtilities
import ClimaCore: Domains, Spaces, Topologies
import ClimaDiagnostics
import RRTMGP

include("parameters/Parameters.jl")
import .Parameters as CAP

include("utils/abbreviations.jl")
include("utils/gpu_compat.jl")
include("types.jl")
include("config/atmos_config.jl")
include("config/cli_options.jl")
include("utils/utilities.jl")
include("utils/debug_utils.jl")
include("utils/variable_manipulations.jl")
include("utils/read_gcm_driven_scm_data.jl")
include("config/era5_observations_to_forcing_file.jl")
include("utils/weather_model.jl")

include("utils/AtmosArtifacts.jl")
import .AtmosArtifacts as AA

include("topography/topography.jl")
include("topography/steady_state_solutions.jl")

include("parameterized_tendencies/radiation/RRTMGPInterface.jl")
import .RRTMGPInterface as RRTMGPI
include("parameterized_tendencies/radiation/radiation.jl")

include("cache/prognostic_edmf_precomputed_quantities.jl")
include("cache/diagnostic_edmf_precomputed_quantities.jl")
include("cache/microphysics_cache.jl")
include("cache/precomputed_quantities.jl")
include("cache/surface_albedo.jl")

# Microphysics module (SGS quadrature, cloud fraction, tendency limiters, wrappers)
include("parameterized_tendencies/microphysics/microphysics.jl")

include("surface_conditions/SurfaceConditions.jl")
include("setups/Setups.jl")
include("utils/refstate_thermodynamics.jl")

include("prognostic_equations/pressure_work.jl")
include("prognostic_equations/zero_velocity.jl")

include("prognostic_equations/implicit/implicit_tendency.jl")
include("prognostic_equations/implicit/jacobian.jl")
include("prognostic_equations/implicit/manual_sparse_jacobian.jl")
include("prognostic_equations/implicit/initialize_implicit_problem.jl")
include("prognostic_equations/implicit/auto_dense_jacobian.jl")
include("prognostic_equations/implicit/auto_sparse_jacobian.jl")
include("prognostic_equations/implicit/autodiff_utils.jl")

include("prognostic_equations/water_advection.jl")
include("prognostic_equations/remaining_tendency.jl")
include("prognostic_equations/forcing/large_scale_advection.jl") # TODO: should this be in tendencies/?
include("prognostic_equations/forcing/subsidence.jl")
include("prognostic_equations/forcing/external_forcing.jl")

include("prognostic_equations/surface_temp.jl")

include("parameterized_tendencies/radiation/held_suarez.jl")

include("parameterized_tendencies/gravity_wave_drag/non_orographic_gravity_wave.jl")
include("parameterized_tendencies/gravity_wave_drag/orographic_gravity_wave_helper.jl")
include("parameterized_tendencies/gravity_wave_drag/orographic_gravity_wave.jl")
include("prognostic_equations/hyperdiffusion.jl")
include("prognostic_equations/gm_sgs_closures.jl")
include("prognostic_equations/scm_coriolis.jl")
include("prognostic_equations/eddy_diffusion_closures.jl")
include("prognostic_equations/mass_flux_closures.jl")
include("prognostic_equations/edmfx_entr_detr.jl")
include("prognostic_equations/edmfx_tke.jl")
include("prognostic_equations/edmfx_sgs_flux.jl")
include("prognostic_equations/edmfx_boundary_condition.jl")
include("prognostic_equations/vertical_diffusion_boundary_layer.jl")
include("prognostic_equations/surface_flux.jl")
include("parameterized_tendencies/sponge/rayleigh_sponge.jl")
include("parameterized_tendencies/sponge/viscous_sponge.jl")
include("parameterized_tendencies/les_sgs_models/smagorinsky_lilly.jl")
include("parameterized_tendencies/les_sgs_models/anisotropic_minimum_dissipation.jl")
include("parameterized_tendencies/les_sgs_models/constant_horizontal_diffusion.jl")
include("prognostic_equations/advection.jl")

include("cache/temporary_quantities.jl")
include("cache/tracer_cache.jl")
include("cache/cache.jl")
include("cache/eddy_diffusivity_coefficient.jl")
include("prognostic_equations/constrain_state.jl")
include("prognostic_equations/limited_tendencies.jl")

include("callbacks/callbacks.jl")

include("diagnostics/Diagnostics.jl")
import .Diagnostics as CAD
import .Diagnostics: DiagnosticsConfig

include("callbacks/get_callbacks.jl")

include("parameters/create_parameters.jl")

include("simulation/grids.jl")
include("simulation/restart.jl")
include("simulation/integrator.jl")
include("simulation/AtmosSimulations.jl")
include("simulation/solve.jl")

include("presets.jl")

include("config/model_getters.jl")
include("config/type_getters.jl")
include("config/yaml_helper.jl")

include("utils/show.jl")

end # module
