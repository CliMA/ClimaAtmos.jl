using ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import Logging

@testset "Hyperdiffusion config" begin
    @info "CAM_SE (Special case of Hyperdiffusion)"

    # Test that CAM_SE uses the correct hardcoded coefficients
    # These match the values from https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2017MS001257
    config = CA.AtmosConfig(
        Dict(
            "z_stretch" => false,
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
            "z_stretch" => false,
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
        Dict(
            "config" => "column",
            "z_stretch" => false,
            "initial_condition" => "Bomex",
        ),
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

    config_mpace = CA.AtmosConfig(
        Dict(
            "config" => "column",
            "z_stretch" => false,
            "initial_condition" => "MPACE",
        ),
        job_id = "test_setup_components_mpace",
    )
    params_mpace = CA.ClimaAtmosParameters(config_mpace)
    thermo_params_mpace = CA.Parameters.thermodynamics_params(params_mpace)
    FT_mpace = eltype(thermo_params_mpace)
    setup_mpace =
        CA.get_setup_type(config_mpace.parsed_args, thermo_params_mpace)
    grid_mpace =
        CA.get_grid(config_mpace.parsed_args, params_mpace, config_mpace.comms_ctx)
    atmos_mpace =
        CA.get_atmos(config_mpace, params_mpace, grid_mpace; setup_type = setup_mpace)

    @test setup_mpace isa CA.Setups.MPACE
    @test atmos_mpace.subsidence isa CA.LargeScaleSubsidence
    @test atmos_mpace.ls_adv isa CA.LargeScaleAdvection
    @test isnothing(atmos_mpace.scm_coriolis)
    @test atmos_mpace.external_forcing isa CA.MPACEForcing
    @test atmos_mpace.external_forcing.u_nudge == FT_mpace(-13)
    @test atmos_mpace.external_forcing.v_nudge == FT_mpace(-3)
    @test atmos_mpace.external_forcing.τ_wind == FT_mpace(3600)
    @test atmos_mpace.insolation isa CA.TimeVaryingInsolation
    @test atmos_mpace.surface.flux_scheme isa CA.SurfaceConditions.MoninObukhov
    @test atmos_mpace.surface.temperature isa
          CA.SurfaceConditions.AnalyticTemperature
    @test setup_mpace.profiles.p(FT_mpace(0)) ≈ FT_mpace(101000) rtol = 1e-5
    @test setup_mpace.profiles.θ(FT_mpace(0)) ≈ FT_mpace(269.2) rtol = 1e-5
    @test setup_mpace.profiles.q_tot(FT_mpace(0)) ≈
          FT_mpace(1.95e-3 / (1 + 1.95e-3)) rtol = 1e-5

    # Verify M-PACE subsidence, large-scale advection, and stratospheric extension
    grav_mpace = CA.Parameters.grav(params_mpace)
    @test setup_mpace.profiles.w_sub(FT_mpace(0)) ≈ FT_mpace(0) atol = 1e-10
    @test setup_mpace.profiles.w_sub(FT_mpace(500)) < FT_mpace(0)
    # Above 850 hPa (z ≈ 1.4 km) and below the upper-tropospheric taper (p ≥ 550 hPa),
    # ω = -ρ g w_sub = D (p_s - p_inv) = 5.8e-6 * (101000 - 85000) Pa/s is constant.
    ω_cap = FT_mpace(5.8e-6 * (101000 - 85000))
    for z_ft in (FT_mpace(2000), FT_mpace(3500))
        p_ft = setup_mpace.profiles.p(z_ft)
        θ_ft = setup_mpace.profiles.θ(z_ft)
        q_ft = setup_mpace.profiles.q_tot(z_ft)
        T_ft = CA.TD.air_temperature(
            thermo_params_mpace,
            CA.TD.pθ_li(),
            p_ft,
            θ_ft,
            q_ft,
        )
        ρ_ft = CA.TD.air_density(
            thermo_params_mpace,
            T_ft,
            p_ft,
            q_ft,
            FT_mpace(0),
            FT_mpace(0),
        )
        @test -ρ_ft * grav_mpace * setup_mpace.profiles.w_sub(z_ft) ≈ ω_cap rtol =
            5e-3
    end
    @test setup_mpace.profiles.w_sub(FT_mpace(10000)) == FT_mpace(0)

    # Large-scale temperature and moisture tendencies (Eqs. 4 & 5)
    @test setup_mpace.profiles.prof_dTdt(FT_mpace(0), FT_mpace(0)) ≈
          FT_mpace(-15 / 86400) rtol = 1e-4
    @test FT_mpace(-15 / 86400) <
          setup_mpace.profiles.prof_dTdt(FT_mpace(0), FT_mpace(1000)) <
          FT_mpace(-4 / 86400)
    @test setup_mpace.profiles.prof_dTdt(FT_mpace(0), FT_mpace(3500)) ≈
          FT_mpace(-4 / 86400) rtol = 1e-4
    @test setup_mpace.profiles.prof_dTdt(FT_mpace(0), FT_mpace(10000)) ==
          FT_mpace(0)

    @test setup_mpace.profiles.prof_dqtdt(FT_mpace(0)) ≈
          FT_mpace(-3e-3 / 86400) rtol = 1e-4
    @test FT_mpace(-3e-3 / 86400) <
          setup_mpace.profiles.prof_dqtdt(FT_mpace(1000)) <
          FT_mpace(-0.164e-3 / 86400)
    @test setup_mpace.profiles.prof_dqtdt(FT_mpace(3500)) ≈
          FT_mpace(-0.164e-3 / 86400) rtol = 1e-4
    @test setup_mpace.profiles.prof_dqtdt(FT_mpace(10000)) == FT_mpace(0)

    # Stratospheric humidity floor (3 ppmw) and isothermal temperature above 300 hPa
    @test setup_mpace.profiles.q_tot(FT_mpace(10000)) ≈
          FT_mpace(3e-6 / (1 + 3e-6)) rtol = 1e-5
    T_strat_12km = CA.TD.air_temperature(
        thermo_params_mpace,
        CA.TD.pθ_li(),
        setup_mpace.profiles.p(FT_mpace(12000)),
        setup_mpace.profiles.θ(FT_mpace(12000)),
        setup_mpace.profiles.q_tot(FT_mpace(12000)),
    )
    T_strat_20km = CA.TD.air_temperature(
        thermo_params_mpace,
        CA.TD.pθ_li(),
        setup_mpace.profiles.p(FT_mpace(20000)),
        setup_mpace.profiles.θ(FT_mpace(20000)),
        setup_mpace.profiles.q_tot(FT_mpace(20000)),
    )
    @test T_strat_12km ≈ T_strat_20km rtol = 1e-5
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

@testset "vert_diff is rejected alongside PROPHET or a vertical LES closure" begin
    # `job_id` has to stay unique across the cases below.
    consistency(; extra...) = CA.check_case_consistency(
        CA.AtmosConfig(
            Dict{String, Any}(
                "config" => "column",
                "vert_diff" => "VerticalDiffusion",
                (String(k) => v for (k, v) in extra)...,
            );
            job_id = "vert_diff_" * join((string(v) for v in values(extra)), "_"),
        ).parsed_args,
    )
    @test consistency() === nothing
    @test_throws "cannot be combined" consistency(; turbconv = "prognostic_edmfx")
    @test_throws "cannot be combined" consistency(; smagorinsky_lilly = "UVW")
    # AMD acts on both axes, so it conflicts whatever its configuration.
    @test_throws "cannot be combined" consistency(; amd_les = true)
    # A horizontal-only Smagorinsky closure leaves the vertical to `vert_diff`.
    @test consistency(; smagorinsky_lilly = "UV") === nothing
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
        @test ogw.a_tofd == ogw_params.a_tofd
        # TOFD is on by default at 2.7 times the IFS amplitude
        # (`ogw_tofd_coefficient = 2.7`, ClimaParams 1.3.9).
        @test ogw.a_tofd == 2.7
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
            a_tofd = 0.0,
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

@testset "n_columns" begin
    function center_space(n_columns)
        config = CA.AtmosConfig(
            Dict("config" => "column", "n_columns" => n_columns, "z_stretch" => false);
            job_id = "test_n_columns",
        )
        grid = CA.get_grid(config, CA.ClimaAtmosParameters(config))
        return CA.get_spaces(grid).center_space
    end
    @test center_space(1) isa CA.Spaces.FiniteDifferenceSpace
    @test center_space(3) isa CA.Spaces.MultiColumnFiniteDifferenceSpace
    @test CA.Spaces.ncolumns(center_space(3)) == 3
end

@testset "Per-column site keys" begin
    parsed_args(config) =
        CA.AtmosConfig(config; job_id = "test_per_column_keys").parsed_args
    sites = parsed_args(
        Dict("site_latitude" => [17, 17], "site_longitude" => [-149, -150.5]),
    )
    @test sites["site_latitude"] == [17.0, 17.0]
    @test sites["site_latitude"] isa Vector{Float64}
    @test parsed_args(Dict("site_latitude" => 17))["site_latitude"] == 17.0
    # Other keys take one value
    @test_throws ErrorException parsed_args(Dict("z_elem" => [10, 20]))
end
