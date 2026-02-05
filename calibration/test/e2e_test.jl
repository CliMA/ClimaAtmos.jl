#= End-to-end test
Runs a perfect model calibration, calibrating on the parameter `total_solar_irradiance`
with top-of-atmosphere radiative shortwave flux in the loss function.

Currently uses ClimaCalibrate.SlurmManager, which integrates with Distributed.jl's workers.

`addprocs(CAL.SlurmManager(10))` starts up an `srun` session consisting of 
10 Julia workers with TCP connections to the host process.
`calibrate(CAL.WorkerBackend, ...)` uses `remotecall` to execute the `forward_model`
function (found in `calibration/model_interface.jl`) on the remote workers.

Further documentation can be found at https://clima.github.io/ClimaCalibrate.jl/dev/backends/
=#
using Distributed
import ClimaCalibrate as CAL
import ClimaAnalysis: SimDir, get, slice

const days = 86_400

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
    return slice(rsut, time = 0days).data
end

addprocs(CAL.SlurmManager())

@everywhere begin
    ENV["CLIMACOMMS_CONTEXT"] = "SINGLETON"
    import ClimaCalibrate as CAL
    import ClimaAtmos as CA
    import JLD2
    import EnsembleKalmanProcesses:
        I, ParameterDistributions.constrained_gaussian

    experiment_dir = joinpath(pkgdir(CA), "calibration", "test")
    model_interface = joinpath(pkgdir(CA), "calibration", "model_interface.jl")
    output_dir = "calibration_end_to_end_test"
    include(model_interface)

    # Experiment Configuration
    ensemble_size = 50
    n_iterations = 10
    total_solar_irradiance = 1362
    noise = 0.1 * I
    prior = constrained_gaussian("total_solar_irradiance", 1000, 200, 0, Inf)
    obs_path = joinpath(experiment_dir, "observations.jld2")
end

# Generate observations if needed
if !isfile(obs_path)
    @info "Generating observations"
    comms_ctx = ClimaComms.SingletonCommsContext()
    model_config = joinpath(experiment_dir, "model_config.yml")
    atmos_config = CA.AtmosConfig(model_config; comms_ctx)
    simulation = CA.get_simulation(atmos_config)
    CA.solve_atmos!(simulation)
    observations = Vector{Float64}(undef, 1)
    observations .= process_member_data(SimDir(simulation.output_dir))
    JLD2.save_object(obs_path, observations)
end

# Initialize experiment data
@everywhere observations = JLD2.load_object(obs_path)

eki = CAL.calibrate(
    CAL.WorkerBackend,
    ensemble_size,
    n_iterations,
    observations,
    noise,
    prior,
    output_dir,
)

# TODO: Enable `calibrate` to checkpoint, rerunning from midway through calibration
# Postprocessing
import EnsembleKalmanProcesses as EKP
import Statistics: var, mean
using Test
import CairoMakie

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

    CairoMakie.hlines!(ax, [total_solar_irradiance], linestyle = :dash)
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

    CairoMakie.hlines!(ax, [total_solar_irradiance]; color = :red, linestyle = :dash)

    output = joinpath(output_dir, "param_vs_iter.png")
    CairoMakie.save(output, f)
    return output
end

scatter_plot(eki)
param_versus_iter_plot(eki)

params = EKP.get_ϕ(prior, eki)
spread = map(var, params)

# Spread should be heavily decreased as particles have converged
@test last(spread) / first(spread) < 0.1
# Parameter should be close to true value
@test mean(last(params)) ≈ total_solar_irradiance rtol = 0.02
