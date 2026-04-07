# EKI driver (definitions only; no `Pkg.activate`, no auto-run).
#
# Call after activating this directory's project:
#   include("eki_calibration.jl"); run_variance_calibration!()
#
# **Configuration:** use `EkiCalibrationOptions` and `va_merge_eki_calibration_env!` so `ENV` is only read at the
# boundary; `run_variance_calibration!` consumes **`opts` only** (defaults merge env via `va_eki_calibration_options_from_env`).
#
# **Worker threads:** ensemble members use `Distributed.addprocs` with [`va_worker_julia_exeflags`](experiment_common.jl).
# Set `worker_threads` (default `1`) so workers run **`-t 1`** while the main process can use **`julia -t N`**.

using Distributed: Distributed
using YAML: YAML
using JLD2: JLD2
using ClimaCalibrate: ClimaCalibrate as CAL
using ClimaAtmos: ClimaAtmos as CA

const _EKI_EXPERIMENT_DIR = dirname(@__FILE__) |> abspath

# Top-level load so `CAL.calibrate` / `observation_map` are not defined newer than the caller (Julia 1.12+).
include(joinpath(_EKI_EXPERIMENT_DIR, "model_interface.jl"))
include(joinpath(_EKI_EXPERIMENT_DIR, "observation_map.jl"))

Base.@kwdef mutable struct EkiCalibrationOptions
    worker_count::Int = min(4, max(1, Sys.CPU_THREADS - 1))
    backend::Symbol = :worker
    worker_threads::Int = 1
end

"""
    va_merge_eki_calibration_env!(opts::EkiCalibrationOptions)

Populate `opts` from `VARIANCE_CALIB_*` environment variables (single boundary for subprocess / CLI).
"""
function va_merge_eki_calibration_env!(opts::EkiCalibrationOptions)
    if haskey(ENV, "VARIANCE_CALIB_WORKERS")
        opts.worker_count = parse(Int, ENV["VARIANCE_CALIB_WORKERS"])
    end
    if haskey(ENV, "VARIANCE_CALIB_WORKER_THREADS")
        opts.worker_threads = parse(Int, ENV["VARIANCE_CALIB_WORKER_THREADS"])
    end
    if haskey(ENV, "VARIANCE_CALIB_BACKEND")
        opts.backend = va_parse_eki_calibration_backend(ENV["VARIANCE_CALIB_BACKEND"])
    end
    return opts
end

"""Default options with `va_merge_eki_calibration_env!` applied (e.g. `run_calibration.jl` entry)."""
function va_eki_calibration_options_from_env()
    opts = EkiCalibrationOptions()
    va_merge_eki_calibration_env!(opts)
    return opts
end

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

function run_variance_calibration!(opts::EkiCalibrationOptions = va_eki_calibration_options_from_env())
    experiment_dir = _EKI_EXPERIMENT_DIR

    load_experiment_config!(experiment_dir)
    expc = experiment_config()

    warn_unless_branch(pkgdir(CA), get(expc, "expected_git_branch", "jb/variance_adjustments"))

    obs_path = va_observations_abs_path(experiment_dir, expc)
    if !isfile(obs_path)
        error(
            "Missing $obs_path. EKI needs **y** before it runs: place **`observations.jld2`** at **`observations_path`** " *
                "(from your workflow). `run_calibration_sweep.jl` only runs calibration; it does not build **`y`**. " *
                "After a completed EKI, `generate_observations_reference.jl` can regenerate **`y`** using **`reference_truth_from_eki`**.",
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

    nw = opts.worker_count
    use_workers = opts.backend === :worker
    use_julia = opts.backend === :julia

    if !use_workers && !use_julia
        error("EkiCalibrationOptions.backend must be :worker or :julia; got $(repr(opts.backend)).")
    end

    if use_workers
        # `nworkers()==0` is never true when `nprocs()==1` (stdlib: `nworkers()` is `nprocs()==1 ? 1 : nprocs()-1`).
        if Distributed.nprocs() == 1
            Distributed.addprocs(nw; exeflags = va_worker_julia_exeflags(experiment_dir, opts.worker_threads))
        end
        Distributed.@everywhere const _va_exp_dir = $experiment_dir
        # Worker imports cannot sit in `run_variance_calibration!()` body (Julia 1.12+ syntax).
        Distributed.@everywhere include(joinpath(_va_exp_dir, "worker_init.jl"))
    end

    calib_backend = use_workers ? CAL.WorkerBackend() : CAL.JuliaBackend()

    @info "Starting calibration" output_dir ensemble_size n_iterations Distributed.nworkers() calibration_mode = get(
        expc,
        "calibration_mode",
        nothing,
    ) observation_length = length(observations) backend = calib_backend options = opts

    eki = CAL.calibrate(
        calib_backend,
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
