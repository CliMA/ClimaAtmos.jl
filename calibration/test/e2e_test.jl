#= End-to-end test
Runs a perfect model calibration, calibrating on the parameter `astronomical_unit`
with top-of-atmosphere radiative shortwave flux in the loss function.

The calibration is run twice, once on the backend obtained via `get_backend()`
and once on the `JuliaBackend`. The output of each calibration is tested individually
and compared to ensure reproducibility.
=#
import ClimaCalibrate as CAL
import ClimaAtmos as CA
import ClimaAnalysis: SimDir, get, slice, average_xy
import CairoMakie
import JLD2
import LinearAlgebra: I
import EnsembleKalmanProcesses as EKP
import Statistics: var, mean
using Test

using Dates

# Debug plots
function scatter_plot(eki::EKP.EnsembleKalmanProcess)
    f = CairoMakie.Figure(resolution = (800, 600))
    ax = CairoMakie.Axis(
        f[1, 1],
        ylabel = "Parameter Value",
        xlabel = "Top of atmosphere radiative SW flux",
    )

    g = vec.(EKP.get_g(eki; return_array = true))
    params = vec.((EKP.get_ϕ(prior, eki)))

    for (gg, uu) in zip(g, params)
        CairoMakie.scatter!(ax, gg, uu)
    end

    CairoMakie.hlines!(ax, [astronomical_unit], linestyle = :dash)
    CairoMakie.vlines!(ax, observations, linestyle = :dash)

    output = joinpath(output_dir, "scatter.png")
    CairoMakie.save(output, f)
    return output
end

function param_versus_iter_plot(eki::EKP.EnsembleKalmanProcess)
    f = CairoMakie.Figure(resolution = (800, 600))
    ax = CairoMakie.Axis(
        f[1, 1],
        ylabel = "Parameter Value",
        xlabel = "Iteration",
    )
    params = EKP.get_ϕ(prior, eki)
    for (i, param) in enumerate(params)
        CairoMakie.scatter!(ax, fill(i, length(param)), vec(param))
    end

    CairoMakie.hlines!(ax, [astronomical_unit]; color = :red, linestyle = :dash)

    output = joinpath(output_dir, "param_vs_iter.png")
    CairoMakie.save(output, f)
    return output
end

# Observation map
function CAL.observation_map(iteration)
    single_member_dims = (1,)
    G_ensemble = Array{Float64}(undef, single_member_dims..., ensemble_size)

    for m in 1:ensemble_size
        member_path = CAL.path_to_ensemble_member(output_dir, iteration, m)
        simdir_path = joinpath(member_path, "output_active")
        if isdir(simdir_path)
            simdir = SimDir(simdir_path)
            G_ensemble[:, m] .= process_member_data(simdir)
        else
            G_ensemble[:, m] .= NaN
        end
    end
    return G_ensemble
end

function process_member_data(simdir::SimDir)
    isempty(simdir.vars) && return NaN
    rsut =
        get(simdir; short_name = "rsut", reduction = "average", period = "30d")
    return slice(average_xy(rsut); time = 30).data
end

# EKI test
function minimal_eki_test(eki)
    params = EKP.get_ϕ(prior, eki)
    spread = map(var, params)

    # Spread should be heavily decreased as particles have converged
    @test last(spread) / first(spread) < 0.1
    # Parameter should be close to true value
    @test mean(last(params)) ≈ astronomical_unit rtol = 0.02
end

# Script:

if !(@isdefined backend)
    backend = CAL.get_backend()
end
# Check that the wait time for the last hour does not exceed 20 minutes.
# This test schedules many slurm jobs and will be prohibitively slow if the cluster is busy
if backend <: CAL.HPCBackend
    wait_times = readchomp(
        `sacct --allocations -u esmbuild --starttime now-1hour -o Submit,Start -n`,
    )
    wait_times = split(wait_times, '\n', keepempty = false)
    # Filter jobs that have not been submitted and started
    filter!(x -> !(contains(x, "Unknown") || contains(x, "None")), wait_times)

    mean_wait_time_in_mins =
        mapreduce(+, wait_times; init = 0) do line
            t1_str, t2_str = split(line)
            t1 = DateTime(t1_str, dateformat"yyyy-mm-ddTHH:MM:SS")
            t2 = DateTime(t2_str, dateformat"yyyy-mm-ddTHH:MM:SS")
            Dates.value(t2 - t1) / 1000 / 60
        end / length(wait_times)

    @show mean_wait_time_in_mins

    if mean_wait_time_in_mins > 10
        @warn """Average wait time for esmbuild is $(round(mean_wait_time_in_mins, digits=2)) minutes. \
                Cluster is too busy to run this test, exiting"""
        exit()
    end
end

# Paths and setup
const experiment_dir = joinpath(pkgdir(CA), "calibration", "test")
const model_interface =
    joinpath(pkgdir(CA), "calibration", "model_interface.jl")
const output_dir = "calibration_end_to_end_test"
include(model_interface)
ensemble_size = 15

# Generate observations
obs_path = joinpath(experiment_dir, "observations.jld2")
if !isfile(obs_path)
    @info "Generating observations"
    config = CA.AtmosConfig(joinpath(experiment_dir, "model_config.yml"))
    simulation = CA.get_simulation(config)
    CA.solve_atmos!(simulation)
    observations = Vector{Float64}(undef, 1)
    observations .= process_member_data(SimDir(simulation.output_dir))
    JLD2.save_object(obs_path, observations)
end

# Initialize experiment data
astronomical_unit = 149_597_870_000
observations = JLD2.load_object(obs_path)
noise = 0.1 * I
n_iterations = 4
prior = CAL.get_prior(joinpath(experiment_dir, "prior.toml"))
experiment_config = CAL.ExperimentConfig(;
    n_iterations,
    ensemble_size,
    observations,
    noise,
    output_dir,
    prior,
)

@info "Running calibration E2E test" backend
if backend <: CAL.HPCBackend
    test_eki = CAL.calibrate(
        backend,
        experiment_config;
        hpc_kwargs = CAL.kwargs(time = 15),
        model_interface,
        verbose = true,
    )
else
    test_eki = CAL.calibrate(backend, experiment_config)
end

scatter_plot(test_eki)
param_versus_iter_plot(test_eki)

@testset "Test Calibration on $backend" begin
    minimal_eki_test(test_eki)
end

# Run calibration
julia_eki = CAL.calibrate(CAL.JuliaBackend, experiment_config)

@testset "Julia-only comparison calibration" begin
    minimal_eki_test(julia_eki)
end

@testset "Compare $backend output to JuliaBackend" begin
    for (uu, slurm_uu) in zip(EKP.get_u(julia_eki), EKP.get_u(test_eki))
        @test uu ≈ slurm_uu rtol = 0.02
    end
end
