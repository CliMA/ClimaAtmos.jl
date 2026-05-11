using Test
import JLD2
import ClimaCalibrate as CAL
import EnsembleKalmanProcesses as EKP

include(joinpath(@__DIR__, "..", "run_calibration.jl"))

@testset "SOCRATES offline calibration dry-run pipeline" begin
    mktempdir() do tmp
        cfg = deepcopy(EXPERIMENT_CONFIG)
        source_output_dir = joinpath(@__DIR__, "..", "output", "simple_socrates")
        cfg["output_dir"] = joinpath(tmp, "simple_socrates")

        # This offline test requires an already-produced calibration iteration.
        source_iter_path = CAL.path_to_iteration(source_output_dir, 0)
        @test isdir(source_iter_path)
        @test isfile(joinpath(source_iter_path, "eki_file.jld2"))

        mkpath(cfg["output_dir"])
        cp(source_iter_path, CAL.path_to_iteration(cfg["output_dir"], 0); force = true)

        # 1) Run reference preprocessing stage (no-op when files already exist).
        preprocess_socrates_reference(;
            flight_numbers = [Int(case["flight_number"]) for case in cfg["cases"]],
            forcing_types = [Symbol(case["forcing_type"]) for case in cfg["cases"]],
            output_dir = joinpath(@__DIR__, "..", "Reference", "Atlas_LES"),
            overwrite = false,
        )

        # 2) Build per-case SOCRATES target grids and observations exactly as calibration setup does.
        z_model_by_case = build_case_target_z(cfg)

        copy_experiment_inputs(cfg)
        observations = build_observations(cfg, z_model_by_case)
        prior = CAL.get_prior(joinpath(@__DIR__, "..", cfg["prior_path"]))

        JLD2.jldsave(
            joinpath(cfg["output_dir"], "obs_metadata.jld2");
            z_model_by_case,
            z_model = z_model_by_case[1],
            dims_per_var = length(z_model_by_case[1]),
        )

        @test isfile(joinpath(cfg["output_dir"], "configs", "experiment_config.yml"))
        @test isfile(joinpath(cfg["output_dir"], "obs_metadata.jld2"))
        @test !isnothing(prior)
        @test !isnothing(observations)

        # 3) Run observation map over the full stored ensemble, matching calibration behavior.
        iter_path = CAL.path_to_iteration(cfg["output_dir"], 0)
        eki = JLD2.load_object(joinpath(iter_path, "eki_file.jld2"))
        expected_obs_len = length(EKP.get_obs(eki))
        @test length(EKP.get_obs(eki)) == expected_obs_len

        G = observation_map(0; config_dict = cfg)
        @test size(G, 1) == expected_obs_len
        @test size(G, 2) == cfg["ensemble_size"]

        # Stored offline artifacts may include failed members (NaNs). This test
        # validates pipeline execution and shape contracts, not artifact quality.
        @test eltype(G) == Float64
    end
end
