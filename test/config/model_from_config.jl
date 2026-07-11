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

@testset "Hyperdiffusion dt safety factor" begin
    getter(overrides) = CA.get_hyperdiffusion_model(
        CA.AtmosConfig(
            merge(Dict{String, Any}("hyperdiff" => "Hyperdiffusion"), overrides),
            job_id = "test_hyperdiff_dt_safety",
        ).parsed_args,
        Float64,
    )

    # `~` (default): the field is 0 (no limit).
    @test getter(Dict{String, Any}()).dt_safety_factor == 0

    # A set value is converted to `FT`, whether given as an Int or a Float.
    @test getter(Dict("hyperdiffusion_dt_safety_factor" => 2)).dt_safety_factor == 2
    @test getter(Dict("hyperdiffusion_dt_safety_factor" => 2.0)).dt_safety_factor == 2

    # Non-positive values are rejected.
    @test_throws ErrorException getter(Dict("hyperdiffusion_dt_safety_factor" => 0))
    @test_throws ErrorException getter(Dict("hyperdiffusion_dt_safety_factor" => -1))

    # The factor composes with the CAM_SE preset.
    cam_se = CA.get_hyperdiffusion_model(
        CA.AtmosConfig(
            Dict(
                "hyperdiff" => "CAM_SE",
                "vorticity_hyperdiffusion_coefficient" => 0.1857,  # CAM_SE preset (0.150 * 1.238)
                "hyperdiffusion_prandtl_number" => 0.2,
                "divergence_damping_factor" => 5.0,
                "hyperdiffusion_dt_safety_factor" => 2,
            ),
            job_id = "test_hyperdiff_dt_safety_cam_se",
        ).parsed_args,
        Float64,
    )
    @test cam_se.dt_safety_factor == 2
    @test cam_se.ν₄_vorticity_coeff == Float64(0.150 * 1.238)
end
