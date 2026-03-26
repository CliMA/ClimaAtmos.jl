using Test
import JLD2
import YAML
import ClimaCalibrate as CAL
import EnsembleKalmanProcesses as EKP

include(joinpath(@__DIR__, "..", "observation_map.jl"))

@testset "SOCRATES observation_map helpers" begin
    @testset "resolve_dims_per_var! from config and metadata" begin
        cfg_direct = Dict{String, Any}("dims_per_var" => 17, "output_dir" => "unused")
        @test resolve_dims_per_var!(cfg_direct) == 17

        mktempdir() do tmp
            cfg_meta = Dict{String, Any}("output_dir" => tmp)
            JLD2.jldsave(joinpath(tmp, "obs_metadata.jld2"); dims_per_var = 23)
            @test resolve_dims_per_var!(cfg_meta) == 23
            @test cfg_meta["dims_per_var"] == 23
        end

        mktempdir() do tmp
            cfg_z = Dict{String, Any}("output_dir" => tmp)
            JLD2.jldsave(joinpath(tmp, "obs_metadata.jld2"); z_model = collect(1.0:5.0))
            @test resolve_dims_per_var!(cfg_z) == 5
            @test cfg_z["dims_per_var"] == 5
        end
    end

    @testset "resolve_z_model_by_case! from config and metadata" begin
        cfg_direct = Dict{String, Any}("z_model_by_case" => [[0.0, 10.0], [0.0, 20.0]], "output_dir" => "unused")
        z_by_case = resolve_z_model_by_case!(cfg_direct)
        @test length(z_by_case) == 2
        @test z_by_case[1] == [0.0, 10.0]
        @test z_by_case[2] == [0.0, 20.0]

        mktempdir() do tmp
            cfg_meta = Dict{String, Any}("output_dir" => tmp, "cases" => [Dict(), Dict()])
            JLD2.jldsave(joinpath(tmp, "obs_metadata.jld2"); z_model_by_case = [[0.0, 50.0], [0.0, 60.0]])
            z_out = resolve_z_model_by_case!(cfg_meta)
            @test z_out[1] == [0.0, 50.0]
            @test z_out[2] == [0.0, 60.0]
        end

        mktempdir() do tmp
            cfg_fallback = Dict{String, Any}("output_dir" => tmp, "cases" => [Dict(), Dict(), Dict()])
            JLD2.jldsave(joinpath(tmp, "obs_metadata.jld2"); z_model = [0.0, 100.0, 200.0])
            z_out = resolve_z_model_by_case!(cfg_fallback)
            @test length(z_out) == 3
            @test z_out[1] == [0.0, 100.0, 200.0]
            @test z_out[3] == [0.0, 100.0, 200.0]
        end
    end

    @testset "resolve_member_simdir prefers output_active" begin
        mktempdir() do tmp
            member_path = joinpath(tmp, "member_001")
            config_path = joinpath(member_path, "config_1")
            mkpath(config_path)

            active = joinpath(config_path, "output_active")
            mkpath(active)
            simdir = resolve_member_simdir(member_path, 1)
            @test occursin("output_active", sprint(show, simdir))
        end
     end

    @testset "resolve_z_model! reads metadata" begin
        mktempdir() do tmp
            cfg = Dict{String, Any}("output_dir" => tmp)
            z_model = [0.0, 100.0, 250.0]
            JLD2.jldsave(joinpath(tmp, "obs_metadata.jld2"); z_model)
            z_out = resolve_z_model!(cfg)
            @test z_out == z_model
            @test cfg["z_model"] == z_model
        end
    end

    @testset "interpolate_profile_to_target_grid handles sorting + extrapolation" begin
        z_src = [300.0, 100.0, 200.0]  # intentionally unsorted
        y_src = [30.0, 10.0, 20.0]
        z_target = [50.0, 150.0, 250.0, 350.0]
        y = interpolate_profile_to_target_grid(y_src, z_src, z_target)
        @test length(y) == length(z_target)
        @test isapprox(y[2], 15.0; atol = 1e-10)
        @test isapprox(y[3], 25.0; atol = 1e-10)
        @test isfinite(y[1]) && isfinite(y[4])
    end

    @testset "offline end-to-end observation_map pipeline" begin
        cfg = YAML.load_file(joinpath(@__DIR__, "..", "experiment_config.yml"))
        cfg["output_dir"] = joinpath(@__DIR__, "..", "output", "simple_socrates")
        cfg["ensemble_size"] = 1

        iter_path = CAL.path_to_iteration(cfg["output_dir"], 0)
        @test isdir(iter_path)
        @test isfile(joinpath(iter_path, "eki_file.jld2"))

        G = observation_map(0; config_dict = cfg)
        eki = JLD2.load_object(joinpath(iter_path, "eki_file.jld2"))

        @test size(G, 1) == length(EKP.get_obs(eki))
        @test size(G, 2) == 1
        @test all(isfinite, G)
    end
 end
