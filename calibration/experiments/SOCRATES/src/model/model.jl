"""
The SOCRATES column model and simulation.

`socrates_simulation` assembles an `AtmosSimulation` from typed ClimaAtmos objects — no YAML, no
`AtmosConfig`. `AtmosSimulation{FT}` already accepts a `setup` object, so nothing here needs to
reimplement `get_simulation`.
"""

using ClimaAtmos: ClimaAtmos as CA
using Dates: Dates

"""
    socrates_model(FT, params; external_forcing, water, turbconv, radiation, surface, numerics, ...)

The SOCRATES column `AtmosModel`: prognostic EDMFX with one updraft, 1-moment non-equilibrium
microphysics, quadrature cloud, all-sky RRTMGP with clear-sky diagnostics, prescribed surface
temperature, and Monin–Obukhov fluxes.

Every component is a keyword argument holding the ClimaAtmos object itself, so a caller swaps a
component by passing a different object.

!!! warning "`AtmosWater`'s struct defaults are not runnable"
    `microphysics_tendency_timestepping` and `sgs_quadrature` both default to `nothing` on
    `AtmosWater`, which silently drops the microphysics tendency and the SGS integration — the
    model then produces no condensate at all. `AtmosWater(::AtmosConfig, …)` in
    `src/config/model_getters.jl` always resolves both, so the `water` default below sets all six
    fields, using ClimaAtmos's own documented defaults (`implicit_microphysics`,
    `use_sgs_quadrature`, `fixed_terminal_velocity` all `true`; `quadrature_order` 3; Gaussian SGS;
    1M substeps 3 bulk / 2 quadrature).

`params` supplies the fixed terminal velocities and SGS quadrature bounds, so it must be the same
parameter set the simulation is built with.

Radiation is computed online by RRTMGP rather than prescribed from the LES `dTdt_rad` (which SSCF
can also supply); see the README for that choice.
"""
function socrates_model(
    ::Type{FT},
    params;
    external_forcing,
    area_fraction::Real = 1.0e-3,
    z0::Real = DEFAULT_Z0,
    surface_albedo::Real = 0.07,
    edmfx = CA.EDMFXModel(;
        entr_model = CA.PiGroupsEntrainment(),
        detr_model = CA.BuoyancyVelocityDetrainment(),
        sgs_mass_flux = true,
        sgs_diffusive_flux = true,
        nh_pressure = true,
        vertical_diffusion = true,
        filter = true,
        scale_blending_method = CA.SmoothMinimumBlending(),
    ),
    turbconv = CA.AtmosTurbconv(;
        edmfx_model = edmfx,
        turbconv_model = CA.PrognosticEDMFX(;
            n_updrafts = 1,
            prognostic_tke = true,
            area_fraction = FT(area_fraction),
        ),
    ),
    water = CA.AtmosWater(;
        microphysics_model = CA.NonEquilibriumMicrophysics1M(;
            n_substeps = 3,
            n_substeps_quad = 2,
        ),
        cloud_model = CA.QuadratureCloud(),
        microphysics_tendency_timestepping = CA.Implicit(),
        tracer_nonnegativity_method = nothing,
        sgs_quadrature = CA.SGSQuadrature(
            FT;
            quadrature_order = 3,
            distribution = CA.GaussianSGS(),
            T_min = FT(CA.Parameters.T_min_sgs(params)),
            q_max = FT(CA.Parameters.q_max_sgs(params)),
        ),
        terminal_velocity_mode = CA.FixedTerminalVelocity{FT}(
            CA.Parameters.fixed_cloud_liquid_terminal_velocity(params),
            CA.Parameters.fixed_cloud_ice_terminal_velocity(params),
            CA.Parameters.fixed_rain_terminal_velocity(params),
            CA.Parameters.fixed_snow_terminal_velocity(params),
        ),
    ),
    radiation = CA.AtmosRadiation(;
        radiation_mode = CA.RRTMGPInterface.AllSkyRadiationWithClearSkyDiagnostics(),
        insolation = CA.ExternalTVInsolation(),
    ),
    surface = CA.AtmosSurface(;
        flux_scheme = CA.SurfaceConditions.MoninObukhov(; z0 = FT(z0)),
        temperature = CA.SurfaceConditions.ExternalTemperature(),
        surface_albedo = CA.ConstantAlbedo{FT}(; α = FT(surface_albedo)),
    ),
    numerics = CA.AtmosNumerics(;
        diff_mode = CA.Implicit(),
        hyperdiff = nothing,
        edmfx_sgsflux_upwinding = :first_order,
    ),
    sponge = CA.AtmosSponge(),
) where {FT <: AbstractFloat}

    return CA.AtmosModel(;
        water,
        scm_setup = CA.SCMSetup(; external_forcing),
        radiation,
        turbconv,
        surface,
        numerics,
        sponge,
    )
end

"""
    socrates_ode_config(FT; ode_algo, update_jacobian_every, max_newton_iters)

The IMEX time-stepping configuration. Spelled with named locals because
`CA.ode_configuration` takes eleven positional arguments, several of which are only meaningful
when a Krylov method is enabled.
"""
function socrates_ode_config(
    ::Type{FT};
    ode_algo::AbstractString = "ARS222",
    update_jacobian_every::AbstractString = "stage",
    max_newton_iters::Int = 1,
    use_krylov_method = false,
    use_dynamic_krylov_rtol = false,
    eisenstat_walker_forcing_alpha = 2.0,
    krylov_rtol = 0.1,
    use_newton_rtol = false,
    newton_rtol = 1.0e-5,
    jvp_step_adjustment = 1.0,
) where {FT <: AbstractFloat}

    return CA.ode_configuration(
        FT,
        ode_algo,
        update_jacobian_every,
        max_newton_iters,
        use_krylov_method,
        use_dynamic_krylov_rtol,
        eisenstat_walker_forcing_alpha,
        krylov_rtol,
        use_newton_rtol,
        newton_rtol,
        jvp_step_adjustment,
    )
end

"""
    socrates_simulation(FT, case; kwargs...)

An `AtmosSimulation{FT}` for `case`, ready to hand to `ClimaAtmos.solve_atmos!`. Returned without
being solved, so it can also be inspected or given extra callbacks.

# Keyword arguments

  - `params`: parameter override sources — a TOML path, a `Dict`, or a vector mixing both, applied
    in order over the case defaults. See [`socrates_params`](@ref).
  - `grid`: the column grid, and the only way to set the vertical resolution. Defaults to the Atlas
    LES's own levels. Build a different one with [`socrates_grid`](@ref) — `socrates_grid(FT, case;
    dz_min = 200)` to merge LES cells, or `faces` for an arbitrary column — or supply any ClimaAtmos
    grid, e.g. `CA.ColumnGrid(FT; z_elem = 60, z_max = 6000, z_stretch = true, dz_bottom = 30)`. The
    forcing is sampled onto whatever levels the grid has and the diagnostics are written on them, so
    one choice propagates everywhere.
  - `t_end`: run length [s]; defaults to the Atlas LES length for this case.
  - `dt`: timestep [s].
  - `output_dir`: where diagnostics are written.
  - `diagnostics`: a `DiagnosticsConfig`; defaults to the scored variables at 600 s.
  - `forcing_dt`: spacing [s] of the sampled forcing time axis.
  - `job_id`, `output_dir_style`, `verbose`: passed through to `AtmosSimulation`.
"""
function socrates_simulation(
    ::Type{FT},
    case::SocratesCase;
    params = nothing,
    grid = socrates_grid(FT, case),
    t_end::Real = SocratesModel.t_end(case),
    t_start::Real = 0,
    dt::Real = 10,
    output_dir::AbstractString,
    diagnostics = nothing,
    forcing_dt::Real = DEFAULT_FORCING_DT,
    forcing_terms = default_socrates_forcing_terms(case),
    ode_config = socrates_ode_config(FT),
    jacobian = CA.ManualSparseJacobian(; approximate_solve_iters = 2),
    checkpoint_frequency = Inf,
    area_fraction::Real = 1.0e-3,
    z0::Real = DEFAULT_Z0,
    model_kwargs = (;),
    job_id::AbstractString = case_name(case),
    output_dir_style::AbstractString = "activelink",
    verbose::Bool = true,
    refresh_forcing::Bool = false,
) where {FT <: AbstractFloat}
    validate(case)
    start_date = simulation_start_date(case)
    z = socrates_z(grid)
    setup = socrates_setup(
        FT,
        case;
        z,
        dt_sec = forcing_dt,
        start_date,
        forcing_terms,
        z0,
        refresh = refresh_forcing,
    )
    atmos_params = socrates_params(FT, case; params)
    return CA.AtmosSimulation{FT}(;
        model = socrates_model(
            FT,
            atmos_params;
            external_forcing = CA.Setups.external_forcing(setup, FT),
            area_fraction,
            z0,
            model_kwargs...,
        ),
        grid,
        setup,
        params = atmos_params,
        dt,
        t_start,
        t_end,
        start_date,
        ode_config,
        jacobian,
        diagnostics = something(diagnostics, socrates_diagnostics(; n_levels = length(z))),
        job_id,
        output_dir,
        output_dir_style,
        checkpoint_frequency,
        verbose,
    )
end