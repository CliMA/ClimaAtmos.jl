using ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import JLD2

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
    @test_throws AssertionError CA.AtmosSimulation(config_wrong)

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

@testset "ProvidedColumnTimeVarying builds model from file when available" begin
    FT = Float64
    z = FT[100.0, 200.0]
    t = FT[0.0, 60.0]
    column_inputs = (; (k => fill(FT(1), length(z), length(t)) for k in CA.ColumnTVForcingKeys.column_inputs)...)
    surface_inputs = (; (k => fill(FT(1), length(t)) for k in CA.ColumnTVForcingKeys.surface_inputs)...)
    coords = (; z, t)
    path = tempname() * ".jld2"
    JLD2.save(path, Dict(
        "column_inputs" => column_inputs,
        "surface_inputs" => surface_inputs,
        "coords" => coords,
    ))
    pa = Dict(
        "external_forcing" => "ProvidedColumnTimeVarying",
        "external_forcing_file" => path,
        "config" => "column",
        "era5_diurnal_warming" => nothing,
    )
    m = CA.get_external_forcing_model(pa, FT)
    @test m isa CA.ProvidedColumnTVForcing

    pa_warn = Dict(
        "external_forcing" => "ProvidedColumnTimeVarying",
        "external_forcing_file" => nothing,
        "config" => "column",
        "era5_diurnal_warming" => nothing,
    )
    @test isnothing(CA.get_external_forcing_model(pa_warn, FT))
end
