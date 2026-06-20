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

@testset "2M+P3 bulk-tendency averaging mode is config-selectable" begin
    pa(m) =
        CA.AtmosConfig(
            Dict(
                "microphysics_model" => "2M",
                "microphysics_averaging_mode" => m,
                "microphysics_n_substeps" => 5,
            ),
            job_id = "test_2m_avg_$(m)",
        ).parsed_args
    # global default `linearized` and the explicit `substepped` both map to the
    # explicit SubsteppedAverage for 2M+P3 (no donor-linearized 2M scheme)
    for m in ("linearized", "substepped")
        tm = CA.get_microphysics_model(pa(m)).tendency_mode
        @test nameof(typeof(tm)) == :SubsteppedAverage
        @test tm.n_substeps == 5
    end
    # rosenbrock_exact selects the exact-Jacobian linearized-implicit mode
    @test nameof(typeof(CA.get_microphysics_model(pa("rosenbrock_exact")).tendency_mode)) ==
          :RosenbrockAverage
    # 1-moment-only modes are rejected for 2M+P3
    @test_throws ErrorException CA.get_microphysics_model(pa("instantaneous"))
    @test_throws ErrorException CA.get_microphysics_model(pa("rosenbrock"))
end
