# EKI driver for the variance-adjustments experiment.
#
# Local (adds Julia workers on this machine):
#   julia --project=. run_calibration.jl
#
# Requires `observations_reference.jld2` (run `generate_observations_reference.jl` first).
#
using Distributed
import LinearAlgebra: I
import YAML
import JLD2
import ClimaCalibrate as CAL
import ClimaAtmos as CA

const experiment_dir = dirname(@__FILE__) |> abspath

function warn_unless_branch(repo::AbstractString, want::AbstractString)
    try
        br = readchomp(`git -C $repo branch --show-current`)
        if br != want
            @warn "Expected git branch `$want`, on `$br`. README: keep model edits on jb/variance_adjustments."
        end
    catch
        @warn "Could not read git branch in $repo"
    end
    return nothing
end

include(joinpath(experiment_dir, "model_interface.jl"))
include(joinpath(experiment_dir, "observation_map.jl"))

load_experiment_config!(experiment_dir)
expc = experiment_config()

warn_unless_branch(pkgdir(CA), get(expc, "expected_git_branch", "jb/variance_adjustments"))

obs_path = joinpath(experiment_dir, expc["observations_path"])
if !isfile(obs_path)
    error(
        "Missing $obs_path. Run: julia --project=. generate_observations_reference.jl",
    )
end
observations = JLD2.load_object(obs_path)
σ = Float64(expc["observation_noise_std"])
noise = (σ^2) * I(length(observations))

prior = CAL.get_prior(joinpath(experiment_dir, expc["prior_path"]))
output_dir = expc["output_dir"]
ensemble_size = Int(expc["ensemble_size"])
n_iterations = Int(expc["n_iterations"])

mkpath(output_dir)
mkpath(joinpath(output_dir, "configs"))
cp(
    joinpath(experiment_dir, "experiment_config.yml"),
    joinpath(output_dir, "configs", "experiment_config.yml"),
    force = true,
)

nw = parse(Int, get(ENV, "VARIANCE_CALIB_WORKERS", string(min(4, max(1, Sys.CPU_THREADS - 1)))))
if nworkers() == 0
    addprocs(nw; exeflags = `--project=$experiment_dir`)
end

@everywhere const _va_exp_dir = $experiment_dir
@everywhere begin
    ENV["CLIMACOMMS_CONTEXT"] = "SINGLETON"
    import Pkg
    Pkg.activate(_va_exp_dir)
    import ClimaCalibrate as CAL
    import ClimaAtmos as CA
    include(joinpath(_va_exp_dir, "model_interface.jl"))
    include(joinpath(_va_exp_dir, "observation_map.jl"))
    load_experiment_config!(_va_exp_dir)
end

@info "Starting calibration" output_dir ensemble_size n_iterations nworkers()

eki = CAL.calibrate(
    CAL.WorkerBackend,
    ensemble_size,
    n_iterations,
    observations,
    noise,
    prior,
    output_dir,
)

@info "Calibration finished" eki
