import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import Random
Random.seed!(1234)

# ---------------------------------------------------------------------------
# Minimal setup to call `hyperdiffusion_tendency!`, exposed as the
# `case_setup`/`case_run` functions expected by the timing harness.
#
# `hyperdiffusion_tendency!` only needs `Y` (state), `p` (cache), and `t`, so
# we skip `CA.get_simulation` (integrator, Jacobian, callbacks, diagnostics,
# output writers) entirely and build just `Y`/`p`/`t` — this mirrors the
# `Y`/`p`/`t`-producing subset of `AtmosSimulation{FT}(...)` (see
# src/simulation/AtmosSimulations.jl). Restart handling is skipped (this
# config starts from an initial condition, not a restart file).
#
# `hyperdiffusion_tendency!` reads only `p.atmos.{hyperdiff, turbconv_model}`,
# `p.core.ᶜΦ`, `p.precomputed.ᶜT`/`ᶜTʲs`, `p.hyperdiff.*`, and `p.scratch`. It
# never reads the radiation, aerosol, trace-gas, gravity-wave, tracer, or
# external-forcing parts of `p`. Those are also the *expensive* part of model
# setup (RRTMGP compilation + lookup tables, ETOPO topography load + spectral
# smoothing, aerosol/trace-gas data ingestion, Beres gravity-wave tables), so
# we strip them from the config below *before* building anything.
# `ClimaAtmosParameters`/`get_atmos` then never load their parameter sets,
# `build_cache` constructs trivial stubs in their place, and
# `set_precomputed_quantities!` skips their precomputed quantities.
#
# What is KEPT (because `hyperdiffusion_tendency!` dispatches on / reads it):
#   hyperdiff = Hyperdiffusion and turbconv = prognostic_edmfx.
# ---------------------------------------------------------------------------

# Defines: get_base_atmos_config
include("atmos_config.jl")

"""
    case_setup()

Build the minimal `Y`/`p`/`t` needed to call `hyperdiffusion_tendency!` and
return them (plus the tendency buffers) as a `NamedTuple` with fields `Y`,
`Yₜ`, `Yₜ_lim`, `p`, `t`. Mirrors the `Y`/`p`/`t`-producing subset of
`AtmosSimulation{FT}(...)`; skips the integrator, Jacobian, callbacks,
diagnostics, and output writers.
"""
function case_setup()
    config = get_base_atmos_config()

    # Stub out the components `hyperdiffusion_tendency!` does not read.
    # Mutating `parsed_args` here (before params/model/grid are built) means
    # the un-needed parameter sets, data files, and caches are never
    # constructed in the first place. None of these knobs affect which
    # branches the tendency takes.
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
    setup_type = CA.get_setup_type(pa, CA.CAP.thermodynamics_params(params))
    grid = CA.get_grid(pa, params, config.comms_ctx)
    model = CA.get_atmos(config, params, grid; setup_type = setup_type)

    # Time arguments (dt is what the cache reads via `p.dt`).
    dt, t_start, t_end = CA.convert_time_args(
        pa["dt"], pa["t_start"], pa["t_end"], CA.parse_date(pa["start_date"]),
    )

    # Build the state `Y` from the initial condition.
    spaces = CA.get_spaces(grid)
    Y = CA.Setups.initial_state(
        setup_type, params, model, spaces.center_space, spaces.face_space,
    )
    CA.Setups.overwrite_initial_state!(setup_type, Y, params.thermodynamics_params)

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

    Yₜ = similar(Y)
    Yₜ_lim = similar(Y)

    return (; Y, Yₜ, Yₜ_lim, p, t)
end

"""
    case_run(state)

Run `hyperdiffusion_tendency!` under measurement. May be called multiple
times on the same state.
"""
function case_run(state)
    CA.hyperdiffusion_tendency!(state.Yₜ, state.Yₜ_lim, state.Y, state.p, state.t)
    return nothing
end
