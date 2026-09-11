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
