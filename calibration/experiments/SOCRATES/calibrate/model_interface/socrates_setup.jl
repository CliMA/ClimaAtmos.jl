"""
In-memory SOCRATES setup + external forcing (no ClimaColumn NetCDF).

`SocratesSetup` supplies IC profiles, surface temperature, insolation, and
`SocratesForcing`. Cache/tendency methods extend ClimaAtmos like ISDAC/GCM,
sampling SSCF arrays onto the model column each step.
"""

using Interpolations: Interpolations
using ClimaAtmos: ClimaAtmos as CA
using Dates: Dates



"""Payload of SSCF-sampled column/surface arrays for one case."""
struct SocratesForcing{Z, T, C, S}
    z::Z                 # forcing vertical coordinate [m]
    times_sec::T         # simulation times [s] from start_date epoch
    column_vars::C       # Dict/NamedTuple of (nz, nt) matrices (canonical names)
    surface_vars::S      # Dict/NamedTuple of length-nt surface series
    lat::Float64
    lon::Float64
end

"""SCM setup: IC + forcing + ExternalTemperature / ExternalTVInsolation."""
struct SocratesSetup{P, F}
    profiles::P          # ClimaAtmos.Setups.ColumnProfiles
    forcing::F           # SocratesForcing
    lat::Float64
    lon::Float64
end

"""Build `SocratesSetup` from `generate_socrates_forcing` NamedTuple."""
function SocratesSetup(forcing_nt::NamedTuple)
    z = forcing_nt.z
    cv = forcing_nt.column_vars
    # IC = first time sample on the forcing vertical grid
    profiles = ClimaAtmos.Setups.ColumnProfiles(
        z,
        vec(cv["ta"][:, 1]),
        vec(cv["ua"][:, 1]),
        vec(cv["va"][:, 1]),
        vec(cv["hus"][:, 1]),
        vec(cv["rho"][:, 1]),
    )
    times_sec = Float64[
        Dates.value(Dates.Second(td - Dates.DateTime(1970, 1, 1))) for
        td in forcing_nt.times_dt
    ]
    forcing = SocratesForcing(
        z,
        times_sec,
        cv,
        forcing_nt.surface_vars,
        Float64(forcing_nt.lat),
        Float64(forcing_nt.lon),
    )
    return SocratesSetup(profiles, forcing, forcing.lat, forcing.lon)
end

# --- Setups interface (injected via get_simulation_with_setup) ---

ClimaAtmos.Setups.center_initial_condition(setup::SocratesSetup, local_geometry, params) =
    ClimaAtmos.Setups.column_profiles_ic(setup.profiles, local_geometry)

ClimaAtmos.Setups.external_forcing(setup::SocratesSetup, ::Type) = setup.forcing

ClimaAtmos.Setups.insolation_model(::SocratesSetup) =
    ClimaAtmos.ExternalTVInsolation() # Does this mean the SSCF LES dTdt_rad over online radiation or is insolation different?

ClimaAtmos.Setups.surface_temperature_model(::SocratesSetup) =
    ClimaAtmos.SurfaceConditions.ExternalTemperature()

function ClimaAtmos.Setups.surface_condition(setup::SocratesSetup, params)
    FT = eltype(params)
    return (;
        flux_scheme = ClimaAtmos.SurfaceConditions.MoninObukhov(; z0 = FT(1e-4)),
        temperature = nothing,
        overrides = nothing,
    )
end

# --- Helpers: (z, t) interpolants and Field fill ---

function _zt_interpolant(z, times_sec, mat)
    # mat is (nz, nt)
    return Interpolations.extrapolate(
        Interpolations.interpolate(
            (z, times_sec),
            mat,
            (Interpolations.Gridded(Interpolations.Linear()), Interpolations.Gridded(Interpolations.Linear())),
        ),
        (Interpolations.Flat(), Interpolations.Flat()),
    )
end

function _fill_zt!(field, itp, t)
    ᶜz = CA.CC.Fields.coordinate_field(axes(field)).z
    zvals = vec(Array(parent(ᶜz)))
    parent(field) .= itp.(zvals, Ref(t))
    return nothing
end

# --- external_forcing_cache / tendency! ---

function ClimaAtmos.external_forcing_cache(
    Y,
    forcing::SocratesForcing,
    params,
    start_date,
)
    FT = CA.CC.Spaces.undertype(axes(Y.c))
    ᶜz = CA.CC.Fields.coordinate_field(Y.c).z
    z = forcing.z
    ts = forcing.times_sec
    cv = forcing.column_vars

    itps = (;
        tntha = _zt_interpolant(z, ts, cv["tntha"]),
        tnhusha = _zt_interpolant(z, ts, cv["tnhusha"]),
        tntva = _zt_interpolant(z, ts, cv["tntva"]),
        tnhusva = _zt_interpolant(z, ts, cv["tnhusva"]),
        ta = _zt_interpolant(z, ts, cv["ta"]),
        hus = _zt_interpolant(z, ts, cv["hus"]),
        ua = _zt_interpolant(z, ts, cv["ua"]),
        va = _zt_interpolant(z, ts, cv["va"]),
        wa = _zt_interpolant(z, ts, cv["wa"]),
    )

    ᶜdTdt_hadv = similar(Y.c, FT)
    ᶜdqtdt_hadv = similar(Y.c, FT)
    ᶜdTdt_fluc = similar(Y.c, FT)
    ᶜdqtdt_fluc = similar(Y.c, FT)
    ᶜT_nudge = similar(Y.c, FT)
    ᶜqt_nudge = similar(Y.c, FT)
    ᶜu_nudge = similar(Y.c, FT)
    ᶜv_nudge = similar(Y.c, FT)
    ᶜls_subsidence = similar(Y.c, FT)
    ᶜinv_τ_scalar = similar(Y.c, FT)
    ᶜinv_τ_wind = similar(Y.c, FT)
    @. ᶜinv_τ_scalar = ClimaAtmos.compute_gcm_driven_scalar_inv_τ(ᶜz, params)
    @. ᶜinv_τ_wind = ClimaAtmos.compute_gcm_driven_momentum_inv_τ(ᶜz, params)

    # Surface TVIs (same contract as ExternalDrivenTVForcing)
    sv = forcing.surface_vars
    surface_vars = (:ts, :coszen, :rsdt)
    surface_target_space =
        axes(CA.CC.Fields.level(Y.f.u₃, CA.CC.Utilities.half))
    method = CA.ClimaUtilities.TimeVaryingInputs.LinearInterpolation()
    surface_timevaryinginputs = (;
        ts = CA.ClimaUtilities.TimeVaryingInputs.TimeVaryingInput(FT.(ts), FT.(sv["ts"]); method),
        coszen = CA.ClimaUtilities.TimeVaryingInputs.TimeVaryingInput(FT.(ts), FT.(sv["coszen"]); method),
        rsdt = CA.ClimaUtilities.TimeVaryingInputs.TimeVaryingInput(FT.(ts), FT.(sv["rsdt"]); method),
    )
    surface_fields = similar(
        CA.CC.Fields.level(Y.f.u₃, CA.CC.Utilities.half),
        NamedTuple{surface_vars, NTuple{length(surface_vars), FT}},
    )

    return (;
        itps,
        ᶜdTdt_hadv,
        ᶜdqtdt_hadv,
        ᶜdTdt_fluc,
        ᶜdqtdt_fluc,
        ᶜT_nudge,
        ᶜqt_nudge,
        ᶜu_nudge,
        ᶜv_nudge,
        ᶜls_subsidence,
        ᶜinv_τ_scalar,
        ᶜinv_τ_wind,
        surface_fields,
        surface_timevaryinginputs,
    )
end

function ClimaAtmos.external_forcing_tendency!(
    Yₜ,
    Y,
    p,
    t,
    ::SocratesForcing,
)
    (;
        itps,
        ᶜdTdt_hadv,
        ᶜdqtdt_hadv,
        ᶜdTdt_fluc,
        ᶜdqtdt_fluc,
        ᶜT_nudge,
        ᶜqt_nudge,
        ᶜu_nudge,
        ᶜv_nudge,
        ᶜls_subsidence,
        ᶜinv_τ_scalar,
        ᶜinv_τ_wind,
    ) = p.external_forcing

    t_sec = float(t)
    _fill_zt!(ᶜdTdt_hadv, itps.tntha, t_sec)
    _fill_zt!(ᶜdqtdt_hadv, itps.tnhusha, t_sec)
    _fill_zt!(ᶜdTdt_fluc, itps.tntva, t_sec)
    _fill_zt!(ᶜdqtdt_fluc, itps.tnhusva, t_sec)
    _fill_zt!(ᶜT_nudge, itps.ta, t_sec)
    _fill_zt!(ᶜqt_nudge, itps.hus, t_sec)
    _fill_zt!(ᶜu_nudge, itps.ua, t_sec)
    _fill_zt!(ᶜv_nudge, itps.va, t_sec)
    _fill_zt!(ᶜls_subsidence, itps.wa, t_sec)

    ClimaAtmos.nudge_uv!(Yₜ, Y, p, ᶜu_nudge, ᶜv_nudge, ᶜinv_τ_wind)

    ᶜdTdt_sum = p.scratch.ᶜtemp_scalar
    ᶜdqtdt_sum = p.scratch.ᶜtemp_scalar_2
    ClimaAtmos.nudge_Tq!(
        ᶜdTdt_sum,
        ᶜdqtdt_sum,
        Y,
        p,
        ᶜT_nudge,
        ᶜqt_nudge,
        ᶜinv_τ_scalar,
    )
    @. ᶜdTdt_sum = ᶜdTdt_hadv + ᶜdTdt_sum + ᶜdTdt_fluc
    @. ᶜdqtdt_sum = ᶜdqtdt_hadv + ᶜdqtdt_sum + ᶜdqtdt_fluc

    ClimaAtmos.apply_Tq_forcing!(Yₜ, Y, p, ᶜdTdt_sum, ᶜdqtdt_sum)
    ClimaAtmos.apply_subsidence_forcing!(Yₜ, Y, p, ᶜls_subsidence)
    return nothing
end

"""
    get_simulation_with_setup(config, setup)

Same as `ClimaAtmos.get_simulation` but uses the given `setup` instead of
`get_setup_type` from the YAML `initial_condition` string.
"""
function get_simulation_with_setup(config::ClimaAtmos.AtmosConfig, setup)
    pa = config.parsed_args
    FT = eltype(config)
    job_id = config.job_id
    params = ClimaAtmos.ClimaAtmosParameters(config)
    model = ClimaAtmos.get_atmos(config, params; setup_type = setup)
    grid = ClimaAtmos.get_grid(pa, params, config.comms_ctx)

    ClimaAtmos.log_context(config.comms_ctx)

    sim = ClimaAtmos.AtmosSimulation{FT}(;
        model,
        params,
        context = config.comms_ctx,
        grid,
        setup,
        steady_state_velocity = ClimaAtmos.steady_state_velocity_from_config(
            config,
            params,
        ),
        dt = pa["dt"],
        start_date = ClimaAtmos.parse_date(pa["start_date"]),
        t_start = pa["t_start"],
        t_end = pa["t_end"],
        ode_config = ClimaAtmos.ode_configuration(FT, pa),
        jacobian = ClimaAtmos.jacobian_from_parsed_args(pa),
        debug_jacobian = pa["debug_jacobian"],
        update_cache_every = pa["update_cache_every"],
        update_constrain_state_every = pa["update_constrain_state_every"],
        aerosol_names = Tuple(pa["prescribed_aerosols"]),
        time_varying_trace_gases = Tuple(pa["time_varying_trace_gases"]),
        vertical_water_borrowing_species = ClimaAtmos.vertical_water_borrowing_species_from_config(
            config,
        ),
        job_id,
        output_dir = pa["output_dir"],
        output_dir_style = pa["output_dir_style"],
        restart_file = pa["restart_file"],
        detect_restart_file = pa["detect_restart_file"],
        callback_kwargs = ClimaAtmos.callback_kwargs_from_config(config),
        diagnostics = ClimaAtmos.diagnostics_config_from_config(config),
        checkpoint_frequency = pa["dt_save_state_to_disk"],
        log_to_file = pa["log_to_file"],
        verbose = true,
    )

    @info "Simulation info" job_id = sim.job_id output_dir = sim.output_dir
    ClimaAtmos.log_yaml_and_toml_manifests(config, sim.output_dir, sim.job_id)
    return sim
end
