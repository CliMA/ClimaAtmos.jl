# EKI driver (definitions only; no `Pkg.activate`, no auto-run).
#
# Call after activating this directory's project:
#   include("eki_calibration.jl"); run_variance_calibration!()

using Distributed
import YAML
import JLD2
import ClimaCalibrate as CAL
import ClimaAtmos as CA

const _EKI_EXPERIMENT_DIR = dirname(@__FILE__) |> abspath

# Top-level load so `CAL.calibrate` / `observation_map` are not defined newer than the caller (Julia 1.12+).
include(joinpath(_EKI_EXPERIMENT_DIR, "model_interface.jl"))
include(joinpath(_EKI_EXPERIMENT_DIR, "observation_map.jl"))

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

function run_variance_calibration!()
    experiment_dir = _EKI_EXPERIMENT_DIR

    load_experiment_config!(experiment_dir)
    expc = experiment_config()

    warn_unless_branch(pkgdir(CA), get(expc, "expected_git_branch", "jb/variance_adjustments"))

    obs_path = va_observations_abs_path(experiment_dir, expc)
    if !isfile(obs_path)
        error(
            "Missing $obs_path. Run `generate_observations_reference!()` or the `run_e2e.jl` / `generate_observations_reference.jl` script.",
        )
    end
    observations = JLD2.load_object(obs_path)
    noise = Matrix(va_build_noise_matrix(observations, expc, experiment_dir))

    prior = CAL.get_prior(joinpath(experiment_dir, expc["prior_path"]))
    out_rel = expc["output_dir"]
    output_dir = isabspath(out_rel) ? String(out_rel) : joinpath(experiment_dir, out_rel)
    ensemble_size = Int(expc["ensemble_size"])
    n_iterations = Int(expc["n_iterations"])

    mkpath(output_dir)
    mkpath(joinpath(output_dir, "configs"))
    cfg_src = va_experiment_config_path(experiment_dir)
    cp(cfg_src, joinpath(output_dir, "configs", basename(cfg_src)), force = true)

    nw = parse(
        Int,
        get(ENV, "VARIANCE_CALIB_WORKERS", string(min(4, max(1, Sys.CPU_THREADS - 1)))),
    )
    if nworkers() == 0
        addprocs(nw; exeflags = `--project=$experiment_dir`)
    end

    @everywhere const _va_exp_dir = $experiment_dir
    # Worker imports cannot sit in `run_variance_calibration!()` body (Julia 1.12+ syntax).
    @everywhere include(joinpath(_va_exp_dir, "worker_init.jl"))

    @info "Starting calibration" output_dir ensemble_size n_iterations nworkers() calibration_mode = get(
        expc,
        "calibration_mode",
        nothing,
    ) observation_length = length(observations)

    eki = CAL.calibrate(
        CAL.WorkerBackend(),
        ensemble_size,
        n_iterations,
        observations,
        noise,
        prior,
        output_dir,
    )

    @info "Calibration finished" eki
    return eki
end
