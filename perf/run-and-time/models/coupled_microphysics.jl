import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import Random
Random.seed!(1234)

# ---------------------------------------------------------------------------
# Minimal setup to call `step_coupled_microphysics_sedimentation!`, exposed as
# the `case_setup`/`case_run` functions expected by the timing harness.
#
# It only needs `Y` (state), `p` (cache), and `dt`, so we skip
# `CA.get_simulation` (integrator, Jacobian, callbacks, diagnostics, output
# writers) and build just `Y`/`p`/`dt` — this mirrors the `Y`/`p`/`t`-producing
# subset of `AtmosSimulation{FT}(...)` (see src/simulation/AtmosSimulations.jl).
#
# `step_coupled_microphysics_sedimentation!` reads the thermodynamic,
# microphysics, and SGS parts of `p.precomputed` and ignores radiation, aerosol,
# trace-gas, gravity-wave, tracer, and external-forcing caches. We strip those
# components from the config before building anything so setup stays fast.
#
# What is KEPT (because it determines the microphysics sweep code path):
#   microphysics_model = 1M and turbconv = prognostic_edmfx.
# ---------------------------------------------------------------------------

# Defines: get_base_atmos_config
include("atmos_config.jl")

"""
    case_setup()

Build the minimal `Y`/`p`/`dt` needed to call
`step_coupled_microphysics_sedimentation!` and return them as a `NamedTuple`
with fields `Y`, `p`, `dt`. Mirrors the `Y`/`p`/`t`-producing subset of
`AtmosSimulation{FT}(...)`; skips the integrator, Jacobian, callbacks,
diagnostics, and output writers.
"""
function case_setup()
    config = get_base_atmos_config()

    # Stub out components `step_coupled_microphysics_sedimentation!` does not
    # read before building params, grid, and caches.
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

    dt, _, _ = CA.convert_time_args(
        pa["dt"],
        pa["t_start"],
        pa["t_end"],
        CA.parse_date(pa["start_date"]),
    )

    spaces = CA.get_spaces(grid)
    Y = CA.Setups.initial_state(
        setup_type,
        params,
        model,
        spaces.center_space,
        spaces.face_space,
    )
    CA.Setups.overwrite_initial_state!(setup_type, Y, params.thermodynamics_params)

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

    return (; Y, p, dt)
end

"""
    case_run(state)

Run `step_coupled_microphysics_sedimentation!` under measurement. May be called
multiple times on the same state.
"""
function case_run(state)
    CA.step_coupled_microphysics_sedimentation!(state.Y, state.p, state.dt)
    return nothing
end
