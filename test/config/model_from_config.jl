using ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA

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
