# Utilities for `calibration/experiments/variance_adjustments` (no ClimaAtmos solve).
using Test
using LinearAlgebra
import YAML

const _VA_DIR = joinpath(@__DIR__, "..", "calibration", "experiments", "variance_adjustments")
include(joinpath(_VA_DIR, "experiment_common.jl"))

@testset "variance_adjustments experiment_common" begin
    expc = YAML.load_file(joinpath(_VA_DIR, "experiment_config.yml"))
    @test isfile(va_experiment_config_path(_VA_DIR))
    @test va_reference_output_dir(_VA_DIR, expc) ==
          joinpath(_VA_DIR, "simulation_output", "TRMM_LBA", "N_3", "varfix_off", "reference")
    @test va_observations_abs_path(_VA_DIR, expc) ==
          joinpath(_VA_DIR, "simulation_output", "TRMM_LBA", "N_3", "varfix_off", "reference", "observations.jld2")
    withenv("VA_EXPERIMENT_CONFIG" => "experiment_config_trmm_N3_varfix_on.yml") do
        p = va_experiment_config_path(_VA_DIR)
        @test basename(p) == "experiment_config_trmm_N3_varfix_on.yml"
    end
    z = va_z_elem(_VA_DIR, expc)
    @test z > 0
    specs = va_field_specs(expc)
    @test !isempty(specs)
    n_y = va_expected_obs_length(_VA_DIR, expc)
    @test n_y == z * length(specs)
    diag_pairs = va_model_diagnostic_shortname_period_pairs(_VA_DIR)
    @test length(diag_pairs) >= 10
    y = ones(Float64, n_y)
    Σ = va_build_noise_matrix(y, expc, _VA_DIR)
    @test size(Σ, 1) == n_y
    @test Matrix(Σ) ≈ (Float64(expc["observation_noise_std"])^2) * I(n_y)

    expc2 = Dict{String, Any}(
        "model_config_path" => expc["model_config_path"],
        "observation_noise_std" => [0.1, 0.2, 0.3],
        "observation_fields" => [
            Dict("short_name" => "a", "reduction" => "average", "period" => "10mins"),
            Dict("short_name" => "b", "reduction" => "average", "period" => "10mins"),
            Dict("short_name" => "c", "reduction" => "average", "period" => "10mins"),
        ],
    )
    n_y2 = z * 3
    y2 = ones(n_y2)
    Σ2 = va_build_noise_matrix(y2, expc2, _VA_DIR)
    d = diag(Matrix(Σ2))
    @test all(d[1:z] .≈ 0.1^2)
    @test all(d[(z + 1):(2z)] .≈ 0.2^2)
    @test all(d[(2z + 1):end] .≈ 0.3^2)
end
