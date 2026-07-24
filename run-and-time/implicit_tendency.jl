import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import Random
Random.seed!(1234)

using BenchmarkTools

# ---------------------------------------------------------------------------
# Minimal setup to call `implicit_tendency!` at the bottom of this file.
#
# `implicit_tendency!` only needs `Y` (state), `p` (cache), and `t`, so we
# skip `CA.get_simulation` (integrator, Jacobian, callbacks, diagnostics,
# output writers) entirely and build just `Y`/`p`/`t` — this mirrors the
# `Y`/`p`/`t`-producing subset of `AtmosSimulation{FT}(...)` (see
# src/simulation/AtmosSimulations.jl). Restart handling is skipped (this
# config starts from an initial condition, not a restart file).
#
# Beyond that, `implicit_tendency!` exercises only the vertical-advection,
# EDMFX, microphysics, implicit-diffusion, and Rayleigh-sponge code paths. A
# call-tree trace confirms it never reads the radiation, aerosol, trace-gas,
# gravity-wave, tracer, external-forcing, or hyperdiffusion parts of `p`.
# Those are also the *expensive* part of model setup (RRTMGP compilation +
# lookup tables, ETOPO topography load + spectral smoothing, aerosol/trace-gas
# data ingestion, Beres gravity-wave tables), so we strip them from the config
# below *before* building anything. `ClimaAtmosParameters`/`get_atmos` then
# never load their parameter sets, `build_cache` constructs trivial stubs in
# their place, and `set_precomputed_quantities!` skips their precomputed
# quantities.
#
# What is KEPT (because `implicit_tendency!` dispatches on / reads it):
#   turbconv = prognostic_edmfx, microphysics_model = 1M, implicit_diffusion,
#   rayleigh_sponge, and DefaultMoninObukhov surface fluxes (needed for the
#   implicit-diffusion path's diffusivities).
# ---------------------------------------------------------------------------
config_path = "config"

config_file = [
    joinpath(pkgdir(CA), config_path, "common_configs", "numerics_sphere_he16ze63.yml"),
    joinpath(
        pkgdir(CA),
        config_path,
        "longrun_configs",
        "amip_target.yml",
    ),
]
config = CA.AtmosConfig(config_file; job_id = nothing)

# Stub out the components `implicit_tendency!` does not read. Mutating
# `parsed_args` here (before params/model/grid are built) means the un-needed
# parameter sets, data files, and caches are never constructed in the first
# place. None of these knobs affect which branches `implicit_tendency!` takes.
pa = config.parsed_args
pa["rad"] = nothing                        # no RRTMGP model / lookup tables
pa["aerosol_radiation"] = false
pa["prescribed_aerosols"] = String[]       # no aerosol data ingestion
pa["time_varying_trace_gases"] = String[]  # no trace-gas data ingestion
pa["insolation"] = "idealized"             # unused once `rad` is off
pa["non_orographic_gravity_wave"] = false  # no Beres source tables
pa["orographic_gravity_wave"] = nothing
pa["topography"] = "NoWarp"                # skip ETOPO load + spectral smoothing
pa["topo_smoothing"] = false

FT = eltype(config)

params = CA.ClimaAtmosParameters(config)
setup = CA.get_setup_type(pa, CA.CAP.thermodynamics_params(params))
model = CA.get_atmos(config, params; setup_type = setup)
grid = CA.get_grid(pa, params, config.comms_ctx)

# Time arguments (dt is what the tendency actually reads via `p.dt`).
dt, t_start, t_end = CA.convert_time_args(
    pa["dt"], pa["t_start"], pa["t_end"], CA.parse_date(pa["start_date"]),
)

# Build the state `Y` from the initial condition.
spaces = CA.get_spaces(grid)
Y = CA.Setups.initial_state(
    setup, params, model, spaces.center_space, spaces.face_space,
)
CA.Setups.overwrite_initial_state!(setup, Y, params.thermodynamics_params)

# Build the cache `p`. `build_cache` also calls `set_precomputed_quantities!`,
# so the precomputed quantities the tendency reads are populated here. The
# aerosol / trace-gas tuples are now empty, so the tracer cache is a stub.
steady_state_velocity = CA.steady_state_velocity_from_config(config, params)
resolved_steady_state_velocity =
    steady_state_velocity isa Function ? steady_state_velocity(Y, params) :
    steady_state_velocity
p = CA.build_cache(
    Y,
    model,
    params,
    dt,
    CA.parse_date(pa["start_date"]),
    Tuple(pa["prescribed_aerosols"]),
    Tuple(pa["time_varying_trace_gases"]),
    resolved_steady_state_velocity,
    CA.vertical_water_borrowing_species_from_config(config),
)

t = t_start

Yₜ = similar(Y);
CA.implicit_tendency!(Yₜ, Y, p, t)
BenchmarkTools.@benchmark CUDA.@sync CA.implicit_tendency!($Yₜ, $Y, $p, $t)
