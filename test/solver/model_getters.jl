import ClimaAtmos as CA

@testset "Model config" begin
    config = CA.AtmosConfig(Dict("config" => "box"))
    parsed_args = config.parsed_args
    @test CA.get_model_config(parsed_args) isa CA.BoxModel

    config = CA.AtmosConfig(Dict("config" => "plane"))
    parsed_args = config.parsed_args
    @test CA.get_model_config(parsed_args) isa CA.PlaneModel
end

@testset "Hyperdiffusion config" begin
    @info "CAM_SE (Special case of ClimaHyperdiffusion)"
    config = CA.AtmosConfig(
        Dict(
            "hyperdiff" => "CAM_SE",
            "vorticity_hyperdiffusion_coefficient" => 0.2,
            "scalar_hyperdiffusion_coefficient" => 99.0,
            "divergence_damping_factor" => 99.0,
        ),
    )
    parsed_args = config.parsed_args
    FT = eltype(config)
    hyperdiff_model = CA.get_hyperdiffusion_model(parsed_args, FT)
    # Coefficients are currently hardcoded in the CAM_SE type!!!
    # Regardless of user specified coeffs above, the CAM_SE selection
    # should override these coefficients to match values in
    # https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2017MS001257
    cam_se_ν₄_vort = FT(0.150 * 1.238)
    cam_se_ν₄_scalar = FT(0.751 * 1.238)
    cam_se_ν₄_dd = FT(5)
    @test hyperdiff_model isa CA.ClimaHyperdiffusion
    @test hyperdiff_model.ν₄_vorticity_coeff == cam_se_ν₄_vort
    @test hyperdiff_model.ν₄_scalar_coeff == cam_se_ν₄_scalar
    @test hyperdiff_model.divergence_damping_factor == cam_se_ν₄_dd

    @info "Test unrecognized Hyperdiffusion scheme"
    config = CA.AtmosConfig(Dict("hyperdiff" => "UnknownHyperdiffusion"))
    parsed_args = config.parsed_args
    FT = eltype(config)
    @test_throws ErrorException CA.get_hyperdiffusion_model(parsed_args, FT)
end
