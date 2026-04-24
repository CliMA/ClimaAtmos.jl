# Utilities for `calibration/experiments/variance_adjustments` (no ClimaAtmos solve).
using Test
using LinearAlgebra
import YAML

const _VA_DIR = joinpath(@__DIR__, "..", "calibration", "experiments", "variance_adjustments")
include(joinpath(_VA_DIR, "lib", "experiment_common.jl"))
include(joinpath(_VA_DIR, "scripts", "resolution_ladder.jl"))
include(joinpath(_VA_DIR, "lib", "forward_sweep_grid.jl"))

@testset "variance_adjustments experiment_common" begin
    @test va_model_config_path_layers(["a.yml", "b.yml"]) == ["a.yml", "b.yml"]
    @test va_model_config_path_layers(Any[Any["a.yml", "b.yml"]]) == ["a.yml", "b.yml"]
    @test va_parse_eki_calibration_backend("worker") === :worker
    @test va_parse_eki_calibration_backend("julia") === :julia
    @test isfile(va_experiment_config_path(_VA_DIR))
    # Path helpers: use a **named** experiment YAML so this test does not track whatever happens to be the repo
    # default under `config/experiment_config.yml` (TRMM vs GCM vs …). Changing defaults should not require edits here.
    expc_paths = va_load_experiment_config(
        _VA_DIR,
        "experiment_configs/experiment_config_bomex_N3_varfix_off.yml",
    )
    merged_paths = va_load_merged_case_yaml_dict(_VA_DIR, expc_paths["model_config_path"])
    @test va_reference_output_dir(_VA_DIR, expc_paths) == joinpath(
        _VA_DIR,
        "simulation_output",
        string(expc_paths["case_name"]),
        "N_$(Int(expc_paths["quadrature_order"]))",
        va_varfix_tag(expc_paths, merged_paths),
        "reference",
    )
    obs_rel = string(expc_paths["observations_path"])
    @test va_observations_abs_path(_VA_DIR, expc_paths) ==
          (isabspath(obs_rel) ? obs_rel : joinpath(_VA_DIR, obs_rel))
    out_rel = string(expc_paths["output_dir"])
    @test va_eki_output_root(_VA_DIR, expc_paths) ==
          (isabspath(out_rel) ? out_rel : joinpath(_VA_DIR, out_rel))

    expc = YAML.load_file(va_experiment_config_path(_VA_DIR))
    withenv("VA_EXPERIMENT_CONFIG" => "experiment_configs/experiment_config_trmm_N3_varfix_on.yml") do
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
    @test length(diag_pairs) >= 6
    y = ones(Float64, n_y)
    Σ = va_build_noise_matrix(y, expc, _VA_DIR)
    @test size(Σ, 1) == n_y
    σs = Float64.(expc["observation_noise_std"])
    @test Matrix(Σ) ≈ diagm(vcat([fill(σs[i]^2, z) for i in 1:length(σs)]...))

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

@testset "variance_adjustments forward_sweep_grid" begin
    @test va_forward_sweep_forward_subdir(ForwardSweepConfig(; forward_parameters = VA_FORWARD_PARAM_EKI_CALIBRATED)) ==
          "forward_eki"
    @test va_forward_sweep_forward_subdir(ForwardSweepConfig(; forward_parameters = VA_FORWARD_PARAM_BASELINE_SCM)) ==
          "forward_only"
    cfg0 = ForwardSweepConfig(; resolution_ladder = false)
    tasks = va_flatten_forward_sweep_tasks(_VA_DIR, cfg0)
    @test !isempty(tasks)
    _yml, _scm, slug, n, vf, tier, z_stretch, yaml_dz, _, _, vfon = first(tasks)
    @test vfon === nothing
    seg = va_tier_path_segment(tier, z_stretch, yaml_dz)
    ap_eki = va_forward_sweep_output_active_path(_VA_DIR, slug, seg, n, vf, ForwardSweepConfig())
    @test occursin("/forward_eki/output_active", ap_eki)
    ap_scm = va_forward_sweep_output_active_path(
        _VA_DIR,
        slug,
        seg,
        n,
        vf,
        ForwardSweepConfig(; forward_parameters = VA_FORWARD_PARAM_BASELINE_SCM),
    )
    @test occursin("/forward_only/output_active", ap_scm)
    # Default forward sweep registry is GoogleLES-only (HadGEM cfSite columns removed); first row is site 01.
    row = va_forward_sweep_registry_row_for(
        _VA_DIR,
        "GOOGLELES",
        "model_configs/googleles_column_varquad_hires.yml",
        ForwardSweepConfig(),
    )
    @test row !== nothing
    @test row.eki_varfix_off_config == "experiment_configs/experiment_config_googleles_01_N3_varfix_off.yml"
    @test va_forward_sweep_task_count(_VA_DIR, ForwardSweepConfig(; resolution_ladder = false)) == 100
    @test va_forward_sweep_task_count(_VA_DIR, ForwardSweepConfig(; resolution_ladder = true)) == 400
    expc_yaml = YAML.load_file(va_experiment_config_path(_VA_DIR))
    pairs_exp = va_model_diagnostic_shortname_period_pairs(_VA_DIR)
    pairs_case = va_case_yaml_diagnostic_shortname_period_pairs(_VA_DIR, expc_yaml["model_config_path"])
    @test pairs_case == pairs_exp
end

import Thermodynamics as TD
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP

@testset "GoogleLES thermo (thetaa observation definition)" begin
    params = CA.ClimaAtmosParameters(Float64)
    tp = CAP.thermodynamics_params(params)
    θ = TD.potential_temperature(tp, 280.0, 1.2, 0.015, 0.001, 0.0)
    @test θ > 200 && θ < 400
end

@testset "GoogleLES experiment YAML" begin
    g = YAML.load_file(joinpath(_VA_DIR, "experiment_configs", "experiment_config_googleles_01_N3_varfix_off.yml"))
    @test g["les_truth"]["source"] == "googleles_cloudbench"
    @test haskey(g, "googleles_forcing_path")
end

@testset "uncalibrated forward sweep registry" begin
    uc = ForwardSweepConfig(;
        registry_path = "registries/forward_sweep_cases_uncalibrated.yml",
        resolution_ladder = false,
        forward_parameters = VA_FORWARD_PARAM_BASELINE_SCM,
    )
    @test va_forward_sweep_task_count(_VA_DIR, uc) == 95
    uc2 = ForwardSweepConfig(;
        registry_path = "registries/forward_sweep_cases_uncalibrated.yml",
        resolution_ladder = false,
        forward_parameters = VA_FORWARD_PARAM_BASELINE_SCM,
        case_slugs = ["TRMM_LBA", "Bomex"],
    )
    @test va_forward_sweep_task_count(_VA_DIR, uc2) == 55
    row = va_forward_sweep_registry_row_for(
        _VA_DIR,
        "TRMM_LBA",
        "model_configs/trmm_column_varquad_hires.yml",
        uc,
    )
    @test row !== nothing
    @test row.eki_varfix_off_config === nothing
end
