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

# Paths and setup
const experiment_dir = joinpath(pkgdir(CA), "calibration", "test")
const model_interface =
    joinpath(pkgdir(CA), "calibration", "model_interface.jl")
const output_dir = joinpath("output", "calibration_end_to_end_test")
include(model_interface)

# Observation map
function CAL.observation_map(iteration)
    ensemble_size = 10
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
n_iterations = 3
ensemble_size = 10
prior = CAL.get_prior(joinpath(experiment_dir, "prior.toml"))
experiment_config = CAL.ExperimentConfig(;
    n_iterations,
    ensemble_size,
    observations,
    noise,
    output_dir,
    prior,
)

# Minimal EKI test
function minimal_eki_test(eki)
    params = EKP.get_u(eki)
    spread = map(x -> var(abs.(x)), params)

    # Spread should be heavily decreased as particles have converged
    @test last(spread) / first(spread) < 0.1
    # Parameter should be close to true value
    @test mean(last(params)) ≈ astronomical_unit rtol = 0.02
end

# Caltech HPC backend
if !(@isdefined backend)
    backend = CAL.get_backend()
end
@info "Running calibration E2E test" backend
if backend <: CAL.SlurmBackend
    slurm_kwargs = CAL.kwargs(time = 15)
    test_eki = CAL.calibrate(
        backend,
        experiment_config;
        slurm_kwargs,
        model_interface,
        verbose = true,
    )
else
    test_eki = CAL.calibrate(backend, experiment_config)
end
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

# Debug plots
function scatter_plot(eki::EKP.EnsembleKalmanProcess)
    f = CairoMakie.Figure(resolution = (800, 600))
    ax = CairoMakie.Axis(
        f[1, 1],
        ylabel = "Absolute Parameter Value",
        xlabel = "Top of atmosphere radiative SW flux",
    )

    g = vec.(EKP.get_g(eki; return_array = true))
    params = map(x -> abs.(x), vec.((EKP.get_u(eki))))

    for (gg, uu) in zip(g, params)
        CairoMakie.scatter!(ax, gg, uu)
    end

    CairoMakie.hlines!(ax, [astronomical_unit], linestyle = :dash)
    CairoMakie.vlines!(ax, observations, linestyle = :dash)

    output = joinpath(output_dir, "scatter.png")
    CairoMakie.save(output, f)
    return output
end

scatter_plot(test_eki)

function param_versus_iter_plot(eki::EKP.EnsembleKalmanProcess)
    f = CairoMakie.Figure(resolution = (800, 600))
    ax = CairoMakie.Axis(
        f[1, 1],
        ylabel = "Parameter Value",
        xlabel = "Iteration",
    )
    params = EKP.get_u(eki)
    for (i, param) in enumerate(params)
        CairoMakie.scatter!(ax, fill(i, length(param)), vec(param))
    end

    CairoMakie.hlines!(ax, [astronomical_unit]; color = :red, linestyle = :dash)

    output = joinpath(output_dir, "param_vs_iter.png")
    CairoMakie.save(output, f)
    return output
end

param_versus_iter_plot(test_eki)
