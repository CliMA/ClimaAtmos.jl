import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import Random
Random.seed!(1234)

# ---------------------------------------------------------------------------
# Minimal setup to time the final 1M/2M step of
# `set_covariance_cache_and_cloud_fraction!` — the cloud env state
# (`cloud_env_state`), `set_sgs_moments_and_cloud_fraction!`, and
# `set_precip_fraction!` — exposed as the `case_setup`/`case_run` functions
# expected by the timing harness.
#
# It only needs `Y` (state) and `p` (cache), so we skip `CA.get_simulation`
# (integrator, Jacobian, callbacks, diagnostics, output writers) entirely and
# build just `Y`/`p`/`t` — this mirrors the `Y`/`p`/`t`-producing subset of
# `AtmosSimulation{FT}(...)` (see src/simulation/AtmosSimulations.jl). Restart
# handling is skipped (this config starts from an initial condition, not a
# restart file).
#
# These are steps `set_precomputed_quantities!` (invoked by `build_cache`)
# already runs, so building the cache in `case_setup` exercises them once. They
# read only the thermodynamic / SGS-covariance parts of `p.precomputed`, and
# never read the radiation, aerosol, trace-gas, gravity-wave, tracer, or
# external-forcing parts of `p`. Those are also the *expensive* part of model
# setup (RRTMGP compilation + lookup tables, ETOPO topography load + spectral
# smoothing, aerosol/trace-gas data ingestion, Beres gravity-wave tables), so
# we strip them from the config below *before* building anything.
# `ClimaAtmosParameters`/`get_atmos` then never load their parameter sets,
# `build_cache` constructs trivial stubs in their place, and
# `set_precomputed_quantities!` skips their precomputed quantities.
#
# What is KEPT (because it determines the SGS/cloud-fraction code path):
#   microphysics_model = 1M, cloud_model = QuadratureCloud, and
#   turbconv = prognostic_edmfx.
# ---------------------------------------------------------------------------

# Defines: get_base_atmos_config
include("atmos_config.jl")

"""
    case_setup()

Build the minimal `Y`/`p`/`t` needed to run `case_run`
and return them as a `NamedTuple` with fields `Y`, `p`, `t`. Mirrors the
`Y`/`p`/`t`-producing subset of `AtmosSimulation{FT}(...)`; skips the
integrator, Jacobian, callbacks, diagnostics, and output writers.
"""
function case_setup()
    config = get_base_atmos_config()

    # Stub out the components the timed routines do not read.
    # Mutating `parsed_args` here (before params/model/grid are built) means the
    # un-needed parameter sets, data files, and caches are never constructed in
    # the first place. None of these knobs affect which branch the routines take.
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

    params = CA.ClimaAtmosParameters(config)
    setup_type = CA.get_setup_type(pa, CA.CAP.thermodynamics_params(params))
    grid = CA.get_grid(pa, params, config.comms_ctx)
    model = CA.get_atmos(config, params, grid; setup_type = setup_type)

    # Time arguments (dt is what the cache reads via `p.dt`).
    dt, t_start, _ = CA.convert_time_args(
        pa["dt"], pa["t_start"], pa["t_end"], CA.parse_date(pa["start_date"]),
    )

    # Build the state `Y` from the initial condition.
    spaces = CA.get_spaces(grid)
    Y = CA.Setups.initial_state(
        setup_type, params, model, spaces.center_space, spaces.face_space,
    )
    CA.Setups.overwrite_initial_state!(setup_type, Y, params.thermodynamics_params)

    # Build the cache `p`. `build_cache` also calls `set_precomputed_quantities!`,
    # so the precomputed quantities the routine reads are populated here. The
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
        resolved_steady_state_velocity,
    )

    t = t_start

    return (; Y, p, t)
end

"""
    case_run(state)

Run the final 1M/2M step of `set_covariance_cache_and_cloud_fraction!` under
measurement: build the cloud env state (materialized under `PrognosticEDMFX`),
then call `set_sgs_moments_and_cloud_fraction!` and `set_precip_fraction!`. May
be called multiple times on the same state.
"""
function case_run(state)
    (; Y, p) = state
    thermo_params = CA.CAP.thermodynamics_params(p.params)
    materialize = p.atmos.turbconv_model isa CA.PrognosticEDMFX
    env_state = CA.cloud_env_state(Y, p, thermo_params; materialize)
    CA.set_sgs_moments_and_cloud_fraction!(Y, p, env_state)
    CA.set_precip_fraction!(Y, p, env_state.ᶜq_precip)
    return nothing
end
