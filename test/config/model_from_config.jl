using ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import Logging
import Dates

@testset "Hyperdiffusion config" begin
    @info "CAM_SE (Special case of Hyperdiffusion)"

    # Test that CAM_SE uses the correct hardcoded coefficients
    # These match the values from https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2017MS001257
    config = CA.AtmosConfig(
        Dict(
            "hyperdiff" => "CAM_SE",
            "vorticity_hyperdiffusion_coefficient" => 0.1857,
            "hyperdiffusion_prandtl_number" => 0.2,
            "divergence_damping_factor" => 5.0,
        ),
        job_id = "test_hyperdiff_cam_se_args",
    )

    parsed_args = config.parsed_args
    FT = eltype(config)
    hyperdiff_model = CA.get_hyperdiffusion_model(parsed_args, FT)

    # Test that CAM_SE returns the correct coefficient values
    cam_se_ν₄_vort = FT(0.150 * 1.238)
    cam_se_prandtl = FT(0.2)
    cam_se_ν₄_dd = FT(5)
    @test hyperdiff_model isa CA.Hyperdiffusion
    @test hyperdiff_model.ν₄_vorticity_coeff == cam_se_ν₄_vort
    @test hyperdiff_model.prandtl_number == cam_se_prandtl
    @test hyperdiff_model.divergence_damping_factor == cam_se_ν₄_dd

    @info "Test CAM_SE coefficient validation"
    # Test that CAM_SE throws an error when user tries to set different coefficients
    config_wrong = CA.AtmosConfig(
        Dict(
            "hyperdiff" => "CAM_SE",
            "vorticity_hyperdiffusion_coefficient" => 0.18571, # deliberately wrong value
            "hyperdiffusion_prandtl_number" => 0.2,
            "divergence_damping_factor" => 5.0,
        ),
        job_id = "test_hyperdiff_cam_se_wrong_args",
    )
    @test_throws AssertionError CA.get_simulation(config_wrong)

    @info "Test unrecognized Hyperdiffusion scheme"
    config_unknown = CA.AtmosConfig(
        Dict("hyperdiff" => "UnknownHyperdiffusion"),
        job_id = "test_unknown_hyperdiff",
    )
    parsed_args_unknown = config_unknown.parsed_args
    FT = eltype(config_unknown)
    @test_throws ErrorException CA.get_hyperdiffusion_model(
        parsed_args_unknown,
        FT,
    )
end

@testset "Setup components applied to the model via get_atmos" begin
    config = CA.AtmosConfig(
        Dict("config" => "column", "initial_condition" => "Bomex"),
        job_id = "test_setup_components_bomex",
    )
    params = CA.ClimaAtmosParameters(config)
    setup = CA.get_setup_type(
        config.parsed_args,
        CA.Parameters.thermodynamics_params(params),
    )
    grid = CA.get_grid(config.parsed_args, params, config.comms_ctx)
    atmos = CA.get_atmos(config, params, grid; setup_type = setup)

    @test atmos.subsidence isa CA.LargeScaleSubsidence
    @test atmos.ls_adv isa CA.LargeScaleAdvection
    @test !isnothing(atmos.scm_coriolis)
    @test atmos.surface.flux_scheme isa CA.SurfaceConditions.MoninObukhov
    @test atmos.surface.temperature isa
          CA.SurfaceConditions.AnalyticTemperature
end

@testset "Smagorinsky-Lilly axes options" begin
    # The constructor accepts exactly the four documented axes symbols and
    # rejects anything else.
    @test_throws AssertionError CA.SmagorinskyLilly(; axes = :XYZ)

    # Each axis symbol classifies the closure along the horizontal and vertical
    # axes; only `:UVW` couples all axes isotropically.
    for (axes, horizontal, vertical, uvw_coupled) in (
        (:UVW, true, true, true),
        (:UV, true, false, false),
        (:W, false, true, false),
        (:UV_W, true, true, false),
    )
        model = CA.SmagorinskyLilly(; axes)
        @test CA.is_smagorinsky_horizontal(model) == horizontal
        @test CA.is_smagorinsky_vertical(model) == vertical
        @test CA.is_smagorinsky_UVW_coupled(model) == uvw_coupled
    end

    # `check_case_consistency` pairs implicit vertical diffusion only with a
    # vertically-acting closure. The horizontal-only `UV` closure has no
    # vertical Jacobian block, so it must be rejected, while `UVW`, `W`, and
    # `UV_W` are accepted.
    consistency(smag) = CA.check_case_consistency(
        CA.AtmosConfig(
            Dict(
                "config" => "box",
                "smagorinsky_lilly" => smag,
                "implicit_diffusion" => true,
                "turbconv" => nothing,
                "vert_diff" => nothing,
            );
            job_id = "smag_$(smag)_imp_diff",
        ).parsed_args,
    )
    @test_throws AssertionError consistency("UV")
    for smag in ("UVW", "W", "UV_W")
        @test consistency(smag) === nothing
    end
end

@testset "Orographic gravity wave config" begin
    FT = Float64
    ogw_params = CA.OrographicGravityWaveParameters(FT)
    # `get_orographic_gravity_wave_model` only reads `orographic_gravity_wave_params`.
    params = (; orographic_gravity_wave_params = ogw_params)

    for name in ("raw_topo", "raw_topo_online", "gfdl_restart")
        parsed_args =
            Dict("orographic_gravity_wave" => name, "topography" => "Earth")
        ogw = CA.get_orographic_gravity_wave_model(parsed_args, params, FT)
        @test ogw isa CA.FullOrographicGravityWave
        @test ogw.topo_info == Val(Symbol(name))
        @test ogw.α_smoothing == ogw_params.α_smoothing
    end

    @test CA.get_orographic_gravity_wave_model(
        Dict("orographic_gravity_wave" => nothing),
        params,
        FT,
    ) === nothing

    @test_throws ErrorException CA.get_orographic_gravity_wave_model(
        Dict("orographic_gravity_wave" => "bogus", "topography" => "Earth"),
        params,
        FT,
    )
end

@testset "raw_topo stale-artifact warning" begin
    FT = Float64
    mk(; γ, h_frac, α, topography = Val(:Earth)) =
        CA.FullOrographicGravityWave{FT, Val{:raw_topo}, typeof(topography)}(;
            γ,
            ϵ = 0.0,
            β = 0.5,
            h_frac,
            ρscale = 1.2,
            L0 = 8.0e4,
            a0 = 0.9,
            a1 = 3.0,
            Fr_crit = 0.7,
            α_smoothing = α,
            topo_info = Val(:raw_topo),
            topography,
        )

    # Defaults match the shipped artifact: no warning.
    @test_logs min_level = Logging.Warn CA.warn_if_stale_raw_topo_artifact(
        mk(; γ = 0.4, h_frac = 0.1, α = 0.15),
    )
    # Analytical topography loads no artifact: no warning even when overridden.
    @test_logs min_level = Logging.Warn CA.warn_if_stale_raw_topo_artifact(
        mk(; γ = 0.5, h_frac = 0.2, α = 0.3, topography = Val(:Schar)),
    )
    # Overridden shape parameter on Earth: warn.
    @test_logs (:warn,) CA.warn_if_stale_raw_topo_artifact(
        mk(; γ = 0.5, h_frac = 0.1, α = 0.15),
    )
end

@testset "ERA5 relaxation config" begin
    FT = Float64

    # Disabled by default.
    @test CA.get_era5_relaxation_model(
        Dict("era5_relaxation" => nothing),
        FT,
    ) === nothing

    parsed_args = Dict(
        "era5_relaxation" => "/some/era5/dir",
        "era5_relaxation_window_hours" => 12.0,
        "era5_relaxation_tau_temperature_hours" => 6.0,
        "era5_relaxation_tau_humidity_hours" => 4.0,
        "era5_relaxation_tau_wind_hours" => 3.0,
        "era5_relaxation_taper_begin_frac" => 0.5,
        "era5_relaxation_taper_end_frac" => 1.0,
    )
    er = CA.get_era5_relaxation_model(parsed_args, FT)
    @test er isa CA.ERA5Relaxation
    @test er.data_dir == "/some/era5/dir"
    @test er.window == 12 * 3600
    @test er.τ_temperature == 6 * 3600
    @test er.τ_humidity == 4 * 3600
    @test er.τ_wind == 3 * 3600
    @test er.taper_begin == 6 * 3600   # 0.5 * window
    @test er.taper_end == 12 * 3600    # 1.0 * window

    # Taper: 1 before taper_begin, linear ramp to 0, then 0.
    @test CA.era5_relaxation_taper(0.0, er) == 1
    @test CA.era5_relaxation_taper(er.taper_begin, er) == 1
    @test CA.era5_relaxation_taper(er.taper_end, er) == 0
    @test CA.era5_relaxation_taper(2 * er.taper_end, er) == 0
    midpoint = (er.taper_begin + er.taper_end) / 2
    @test CA.era5_relaxation_taper(midpoint, er) ≈ 0.5

    # Invalid taper fractions are rejected.
    bad = merge(parsed_args, Dict("era5_relaxation_taper_begin_frac" => 1.5))
    @test_throws ErrorException CA.get_era5_relaxation_model(bad, FT)

    # Snapshot dates bracket the window on the 6-hourly grid.
    start_date = Dates.DateTime(2016, 12, 31, 3)
    dates = CA.era5_relaxation_snapshot_dates(start_date, 12 * 3600)
    @test first(dates) <= start_date
    @test last(dates) >= start_date + Dates.Hour(12)
    @test all(diff(Dates.value.(dates)) .> 0)
end
