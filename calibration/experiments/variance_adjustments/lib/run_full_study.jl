# Full experiment driver: ensure LES `observations.jld2` per sweep YAML → EKI sweep → naive forwards → forward sweep (merged θ) → figures.
#
# Default forward registry `registries/forward_sweep_cases.yml` uses **GoogleLES** cases only (GCM cfSite rows removed
# from that sweep; cfSite could still *run* but was dropped for study quality). Varfix-on SGS variant:
# `sgs_distribution_varfix_on` in `model_configs/master_column_varquad_diagnostic_edmfx.yml`.
# Default **`config/experiment_config.yml`** stays **TRMM** (fewer prerequisites than GoogleLES forcing).
#
# **Uncalibrated study (SCM-only forwards, no EKI):** use `--uncalibrated-study` or pass `--forward-registry=registries/forward_sweep_cases_uncalibrated.yml`
# with `--forward-baseline-scm` and `--skip-calib` / `--skip-naive` (the preset sets these). Example:
#   julia --project=. scripts/run_full_study.jl --uncalibrated-study 2>&1 | tee logs/uncalibrated_full_study.log
#
# **CLI** — from `calibration/experiments/variance_adjustments/`:
#   julia --project=. scripts/run_full_study.jl [flags]
# From **any** cwd (no `cd`): invoke by absolute path, e.g.
#   julia /.../variance_adjustments/scripts/run_full_study.jl [flags]
#
# **REPL** (kwargs; optional `VA_*` / `VARIANCE_*` merged when fields are left unset — see `va_full_study_merge_env!`):
#   using Pkg; Pkg.activate("."); include("scripts/run_full_study.jl")
#   run_full_study!()                              # default pipeline
#   run_full_study!(; skip_forward = true)         # only EKI + naive + figures (forward-sweep figures need existing forward_eki/ data)
#   run_full_study_figures_only!()                   # alias for figures_only = true
#   run_full_study_uncalibrated_figures_only!()     # figures only + uncalibrated preset
#   run_full_study!(; figures_only = true)         # same as run_full_study_figures_only!
#   run_full_study!(; figures_only = true, uncalibrated_study = true)
#   run_full_study!(; forward_baseline_only = true) # 20 forwards (N_quad 1–5), no resolution ladder
#
# CLI flags (same options as kwargs on `FullStudyOptions`):
#   --skip-instantiate
#   --skip-forward
#   --forward-baseline-only     # forward sweep: single YAML tier per case (no resolution ladder)
#   --forward-baseline-scm      # forward sweep: registry scm_toml only (forward_only/); default uses merged EKI TOML (forward_eki/)
#   --skip-calib
#   --skip-naive
#   --skip-figures
#   --figures-only              # same as skipping forward, calib, naive, instantiate (re-run post-analysis only)
#   --forward-skip-done         # passed to forward sweep (--skip-done)
#   --forward-fail-fast         # forward sweep: abort on first failed run (--fail-fast); needs merged TOMLs unless --forward-baseline-scm
#   --calib-fail-fast           # EKI sweep: abort on first failed calibration YAML (--fail-fast)
#   --naive-fail-fast           # naive forwards: abort on first failure (--fail-fast)
#   --naive-skip-done           # passed to naive script (--skip-done)
#   --help
#
# **Configuration:** options are carried on `FullStudyOptions`; env is merged in `va_full_study_merge_env!` (before argv).
# Job arrays may still set `SWEEP_TASK_ID` / `SLURM_ARRAY_TASK_ID` / `NAIVE_SWEEP_TASK_ID` when not using `--task-id=`.
#
# **Parallelism:** The full study runs **in one Julia process**: EKI uses `EkiCalibrationOptions` (`addprocs` /
# `rmprocs` after each EKI calibration YAML), forward sweep uses `ForwardSweepConfig` (`addprocs` / `rmprocs` for `parallel=:distributed`).
# No extra `julia` subprocesses for calibration / naive / forward (avoids duplicate compilation and orphan PIDs).
#
isdefined(Main, :_VA_ROOT) ||
    error("Internal: include `lib/run_full_study.jl` only after setting `Main._VA_ROOT` to the experiment directory.")
import Pkg

const _VA_LIB = joinpath(_VA_ROOT, "lib")
include(joinpath(_VA_LIB, "stdio_flush.jl"))
va_setup_stdio_flushing!()
include(joinpath(_VA_LIB, "calibration_sweep_configs.jl"))
include(joinpath(_VA_LIB, "les_truth_build.jl"))

# Load plotting + grid helpers **before** `run_full_study!` is defined so Julia ≥1.12 does not compile
# `run_full_study!` in a world age older than `va_run_post_analysis!` / `ForwardSweepConfig` (avoids
# MethodError when driving from one session: `include("scripts/run_full_study.jl"); run_full_study!()`).
include(joinpath(_VA_ROOT, "analysis/plotting/run_post_analysis.jl"))
include(joinpath(_VA_LIB, "eki_calibration.jl"))
include(joinpath(_VA_ROOT, "scripts", "sweep_forward_core.jl"))
include(joinpath(_VA_ROOT, "analysis", "plotting", "plot_forward_sweep_body.jl"))
include(joinpath(_VA_ROOT, "analysis", "plotting", "plot_naive_vs_calibrated_varfix_on.jl"))
include(joinpath(_VA_ROOT, "scripts", "run_calibration_sweep.jl"))
include(joinpath(_VA_ROOT, "scripts", "run_naive_varfix_on_forwards.jl"))

Base.@kwdef mutable struct FullStudyOptions
    skip_instantiate::Bool = false
    skip_forward::Bool = false
    """If false, forward sweep uses single YAML tier per case (`baseline-only` mode)."""
    forward_resolution_ladder::Bool = true
    """
    If true, forward sweep uses baseline SCM TOML (`forward_only/`). Default false uses merged EKI TOML (`forward_eki/`),
    which requires completed EKI before the forward sweep.
    """
    forward_baseline_scm::Bool = false
    skip_calib::Bool = false
    skip_naive::Bool = false
    skip_figures::Bool = false
    forward_skip_done::Bool = false
    """Abort forward sweep on first failed run (forward sweep `fail_fast`)."""
    forward_fail_fast::Bool = false
    """Pass `--fail-fast` to `run_calibration_sweep.jl`."""
    calib_fail_fast::Bool = false
    """Pass `--fail-fast` to naive forwards script."""
    naive_fail_fast::Bool = false
    naive_skip_done::Bool = false
    """Forward sweep parallel mode (`:sequential`, `:threads`, `:distributed`)."""
    forward_parallel::Union{Nothing, Symbol} = nothing
    """Worker count when forward sweep uses `parallel=:distributed`."""
    forward_distributed_workers::Union{Nothing, Int} = nothing
    """Per-worker `-t` for forward sweep `Distributed` workers."""
    forward_distributed_worker_threads::Union{Nothing, Int} = nothing
    """EKI calibration worker count (`EkiCalibrationOptions.worker_count`)."""
    calib_worker_count::Union{Nothing, Int} = nothing
    """Per-worker `-t` for EKI calibration `Distributed` workers."""
    calib_worker_threads::Union{Nothing, Int} = nothing
    """EKI backend (`:worker` or `:julia`)."""
    calib_backend::Union{Nothing, Symbol} = nothing
    """If true, do not build missing `observations.jld2` from LES before the calibration sweep (expert / CI)."""
    skip_les_observations_build::Bool = false
    """
    If set, passed to `sweep_forward_runs.jl` as `--registry=` (path relative to this directory or absolute).
    Default `nothing` → `registries/forward_sweep_cases.yml`.
    """
    forward_registry::Union{Nothing, String} = nothing
    """
    If set, forward sweep and forward-sweep figures only include registry rows whose merged case slug is in this list
    (same strings as directory names under `simulation_output/`, e.g. `TRMM_LBA`, `GOOGLELES_01`). Env: **`VA_FORWARD_SWEEP_CASE_SLUGS`** (comma-separated). CLI: **`--forward-case-slugs=...`**.
    """
    forward_case_slugs::Union{Nothing, Vector{String}} = nothing
    """
    If true (uncalibrated study only), only **`forward_sweep_clw_plus_cli_summary.png`** is built from the **full**
    uncalibrated registry (`registries/forward_sweep_cases_uncalibrated.yml`, no [`forward_case_slugs`](@ref)), so every
    case row with data on disk appears in the summary. Per-case profile and scalar plots
    ([`va_plot_forward_sweep_comparisons!`](@ref), [`va_plot_forward_sweep_scalars_vs_nquad!`](@ref)) still use the **same**
    filter as [`run_forward_sweep!`](@ref), so e.g. only new GCM rows get refreshed individual PNGs. Env:
    **`VA_FORWARD_FIGURES_FULL_UNCALIBRATED_REGISTRY=1`**. CLI: **`--forward-figures-full-uncalibrated-registry`**.
    """
    forward_figures_full_uncalibrated_registry::Bool = false
    """
    If set, forward sweep runs only these varfix legs (e.g. `[true]` = varfix-on / vertical-profile SGS only).
    Overrides **`VA_FORWARD_SWEEP_VARFIX`** after env merge. Default **`nothing`** → **`[false, true]`** on
    [`ForwardSweepConfig`](@ref) unless env sets otherwise. CLI: **`--forward-varfix=...`** (same tokens as
    [`va_forward_sweep_varfix_values_from_spec`](@ref)).
    """
    forward_varfix_values::Union{Nothing, Vector{Bool}} = nothing
    """
    Preset: skip LES-obs build, calibration, naive forwards; use `--baseline-scm-forward` and
    `registries/forward_sweep_cases_uncalibrated.yml` unless `forward_registry` is set. One-command uncalibrated pipeline.
    """
    uncalibrated_study::Bool = false
    """
    Preset: skip instantiate, forward sweep, calibration, naive forwards, and LES obs build; **only** run figure
    stages (forward-sweep comparison plots + optional EKI post-analysis when calibration was not skipped).
    Compose with `uncalibrated_study = true` for `forward_only/` + uncalibrated registry paths. CLI: `--figures-only`.
    Env: `VA_FIGURES_ONLY=1`.
    """
    figures_only::Bool = false
end

function va_full_study_apply_figures_only_preset!(opts::FullStudyOptions)
    opts.figures_only || return opts
    opts.skip_instantiate = true
    opts.skip_forward = true
    opts.skip_calib = true
    opts.skip_naive = true
    opts.skip_les_observations_build = true
    return opts
end

function va_full_study_apply_uncalibrated_preset!(opts::FullStudyOptions)
    opts.uncalibrated_study || return opts
    opts.skip_calib = true
    opts.skip_naive = true
    opts.skip_les_observations_build = true
    opts.forward_baseline_scm = true
    if opts.forward_registry === nothing || isempty(strip(opts.forward_registry))
        opts.forward_registry = "registries/forward_sweep_cases_uncalibrated.yml"
    end
    return opts
end

"""Merge `VA_*` / `VARIANCE_*` env into `opts` where the corresponding field is still `nothing` (boundary only)."""
function va_full_study_merge_env!(opts::FullStudyOptions)
    if opts.forward_parallel === nothing && haskey(ENV, "VA_FORWARD_SWEEP_PARALLEL")
        opts.forward_parallel = va_parse_forward_sweep_parallel_mode(ENV["VA_FORWARD_SWEEP_PARALLEL"])
    end
    if opts.forward_distributed_workers === nothing && haskey(ENV, "VA_FORWARD_SWEEP_DISTRIBUTED_WORKERS")
        opts.forward_distributed_workers = parse(Int, ENV["VA_FORWARD_SWEEP_DISTRIBUTED_WORKERS"])
    end
    if opts.forward_distributed_worker_threads === nothing && haskey(ENV, "VA_FORWARD_SWEEP_WORKER_THREADS")
        opts.forward_distributed_worker_threads = parse(Int, ENV["VA_FORWARD_SWEEP_WORKER_THREADS"])
    end
    if opts.calib_worker_count === nothing && haskey(ENV, "VARIANCE_CALIB_WORKERS")
        opts.calib_worker_count = parse(Int, ENV["VARIANCE_CALIB_WORKERS"])
    end
    if opts.calib_worker_threads === nothing && haskey(ENV, "VARIANCE_CALIB_WORKER_THREADS")
        opts.calib_worker_threads = parse(Int, ENV["VARIANCE_CALIB_WORKER_THREADS"])
    end
    if opts.calib_backend === nothing && haskey(ENV, "VARIANCE_CALIB_BACKEND")
        opts.calib_backend = va_parse_eki_calibration_backend(ENV["VARIANCE_CALIB_BACKEND"])
    end
    if !opts.skip_les_observations_build &&
            strip(get(ENV, "VA_SKIP_LES_OBSERVATIONS_BUILD", "")) in ("1", "true", "yes")
        opts.skip_les_observations_build = true
    end
    if opts.forward_registry === nothing && haskey(ENV, "VA_FORWARD_SWEEP_REGISTRY")
        opts.forward_registry = String(strip(ENV["VA_FORWARD_SWEEP_REGISTRY"]))
    end
    if opts.forward_case_slugs === nothing && haskey(ENV, "VA_FORWARD_SWEEP_CASE_SLUGS")
        opts.forward_case_slugs = va_forward_sweep_parse_case_slugs(ENV["VA_FORWARD_SWEEP_CASE_SLUGS"])
    end
    if !opts.uncalibrated_study && strip(get(ENV, "VA_UNCALIBRATED_STUDY", "")) in ("1", "true", "yes")
        opts.uncalibrated_study = true
    end
    if !opts.figures_only && strip(get(ENV, "VA_FIGURES_ONLY", "")) in ("1", "true", "yes")
        opts.figures_only = true
    end
    if !opts.forward_figures_full_uncalibrated_registry &&
            strip(get(ENV, "VA_FORWARD_FIGURES_FULL_UNCALIBRATED_REGISTRY", "")) in ("1", "true", "yes")
        opts.forward_figures_full_uncalibrated_registry = true
    end
    return opts
end

function forward_baseline_only(opts::FullStudyOptions)
    return !opts.forward_resolution_ladder
end

function _full_study_print_help()
    println("""
Usage: julia --project=. scripts/run_full_study.jl [flags]

  --skip-instantiate
  --skip-forward
  --forward-baseline-only
  --forward-baseline-scm
  --skip-calib
  --skip-naive
  --skip-figures
  --figures-only              Preset: only post-analysis + forward-sweep figures (same as figures_only=true / VA_FIGURES_ONLY=1)
  --forward-skip-done
  --forward-fail-fast
  --calib-fail-fast
  --naive-fail-fast
  --naive-skip-done
  --forward-parallel=MODE   sequential | threads | distributed (forward sweep only)
  --forward-distributed-workers=N
  --forward-distributed-worker-threads=N
  --calib-workers=N            VARIANCE_CALIB_WORKERS for each calibration subprocess
  --calib-worker-threads=N     VARIANCE_CALIB_WORKER_THREADS
  --calib-backend=worker|julia VARIANCE_CALIB_BACKEND
  --skip-les-observations     Skip building missing LES `observations.jld2` before EKI (see also `VA_SKIP_LES_OBSERVATIONS_BUILD`)
  --forward-registry=REL.yml  Forward sweep + figures: pass `--registry=` to `sweep_forward_runs.jl` (default: `registries/forward_sweep_cases.yml`)
  --forward-case-slugs=a,b,c   Only these merged case slugs (subset of the registry). Env: VA_FORWARD_SWEEP_CASE_SLUGS
  --forward-figures-full-uncalibrated-registry   Figures: full uncalibrated registry + no slug filter (forwards unchanged). Env: VA_FORWARD_FIGURES_FULL_UNCALIBRATED_REGISTRY
  --forward-varfix=both|on|off|off,on   Varfix axis for forward sweep (default both). Env: VA_FORWARD_SWEEP_VARFIX
  --uncalibrated-study        Preset: SCM-only forwards on `registries/forward_sweep_cases_uncalibrated.yml` (skips calib, naive, LES obs build)

REPL: include(\"scripts/run_full_study.jl\"); run_full_study!(; skip_forward = true)
REPL (figures only): run_full_study!(; figures_only = true)
REPL (uncalibrated): run_full_study!(; uncalibrated_study = true)
REPL (uncalibrated figures only): run_full_study!(; figures_only = true, uncalibrated_study = true)
""")
    va_flush_stdio()
    return nothing
end

function parse_full_study_cli(argv::Vector{String})::FullStudyOptions
    opts = FullStudyOptions()
    va_full_study_merge_env!(opts)
    for a in argv
        if a == "--help" || a == "-h"
            _full_study_print_help()
            exit(0)
        elseif a == "--skip-instantiate"
            opts.skip_instantiate = true
        elseif a == "--skip-forward"
            opts.skip_forward = true
        elseif a == "--forward-baseline-only"
            opts.forward_resolution_ladder = false
        elseif a == "--forward-baseline-scm"
            opts.forward_baseline_scm = true
        elseif a == "--skip-calib"
            opts.skip_calib = true
        elseif a == "--skip-naive"
            opts.skip_naive = true
        elseif a == "--skip-figures"
            opts.skip_figures = true
        elseif a == "--figures-only"
            opts.figures_only = true
        elseif a == "--forward-skip-done"
            opts.forward_skip_done = true
        elseif a == "--forward-fail-fast"
            opts.forward_fail_fast = true
        elseif a == "--calib-fail-fast"
            opts.calib_fail_fast = true
        elseif a == "--naive-fail-fast"
            opts.naive_fail_fast = true
        elseif a == "--naive-skip-done"
            opts.naive_skip_done = true
        elseif startswith(a, "--forward-parallel=")
            opts.forward_parallel = va_parse_forward_sweep_parallel_mode(split(a, '=', limit = 2)[2])
        elseif startswith(a, "--forward-distributed-workers=")
            opts.forward_distributed_workers = parse(Int, split(a, '=', limit = 2)[2])
        elseif startswith(a, "--forward-distributed-worker-threads=")
            opts.forward_distributed_worker_threads = parse(Int, split(a, '=', limit = 2)[2])
        elseif startswith(a, "--calib-workers=")
            opts.calib_worker_count = parse(Int, split(a, '=', limit = 2)[2])
        elseif startswith(a, "--calib-worker-threads=")
            opts.calib_worker_threads = parse(Int, split(a, '=', limit = 2)[2])
        elseif startswith(a, "--calib-backend=")
            opts.calib_backend = va_parse_eki_calibration_backend(split(a, '=', limit = 2)[2])
        elseif a == "--skip-les-observations"
            opts.skip_les_observations_build = true
        elseif startswith(a, "--forward-registry=")
            opts.forward_registry = String(strip(split(a, '=', limit = 2)[2]))
        elseif startswith(a, "--forward-case-slugs=")
            opts.forward_case_slugs = va_forward_sweep_parse_case_slugs(String(split(a, '=', limit = 2)[2]))
        elseif a == "--forward-figures-full-uncalibrated-registry"
            opts.forward_figures_full_uncalibrated_registry = true
        elseif startswith(a, "--forward-varfix=")
            opts.forward_varfix_values = va_forward_sweep_varfix_values_from_spec(split(a, '=', limit = 2)[2])
        elseif a == "--uncalibrated-study"
            opts.uncalibrated_study = true
        else
            error("Unknown argument: $(repr(a)). Try --help.")
        end
    end
    return opts
end

function va_eki_calibration_options_from_full_study(opts::FullStudyOptions)
    o = EkiCalibrationOptions()
    va_merge_eki_calibration_env!(o)
    opts.calib_worker_count !== nothing && (o.worker_count = opts.calib_worker_count)
    opts.calib_worker_threads !== nothing && (o.worker_threads = opts.calib_worker_threads)
    opts.calib_backend !== nothing && (o.backend = opts.calib_backend)
    return o
end

function va_full_study_forward_sweep_config(opts::FullStudyOptions)::ForwardSweepConfig
    cfg = ForwardSweepConfig()
    va_forward_sweep_merge_env!(cfg)
    cfg.resolution_ladder = opts.forward_resolution_ladder
    cfg.forward_parameters = opts.forward_baseline_scm ? VA_FORWARD_PARAM_BASELINE_SCM : VA_FORWARD_PARAM_EKI_CALIBRATED
    if opts.forward_skip_done
        cfg.skip_done = true
    end
    if opts.forward_fail_fast
        cfg.fail_fast = true
    end
    if opts.forward_registry !== nothing && !isempty(strip(opts.forward_registry))
        cfg.registry_path = String(strip(opts.forward_registry))
    end
    if opts.forward_parallel !== nothing
        cfg.parallel = opts.forward_parallel
    end
    if opts.forward_distributed_workers !== nothing
        cfg.distributed_workers = opts.forward_distributed_workers
    end
    if opts.forward_distributed_worker_threads !== nothing
        cfg.distributed_worker_threads = opts.forward_distributed_worker_threads
    end
    if opts.forward_varfix_values !== nothing
        cfg.varfix_values = copy(opts.forward_varfix_values)
    end
    if opts.forward_case_slugs !== nothing
        cfg.case_slugs = copy(opts.forward_case_slugs)
    end
    return cfg
end

"""Forward-sweep plotting config: optionally full uncalibrated grid while forwards used a subset."""
function va_full_study_forward_sweep_figure_config(opts::FullStudyOptions)::ForwardSweepConfig
    cfg = va_full_study_forward_sweep_config(opts)
    if !opts.forward_figures_full_uncalibrated_registry
        return cfg
    end
    if !opts.uncalibrated_study
        @warn "forward_figures_full_uncalibrated_registry: ignored (requires uncalibrated_study = true)"
        return cfg
    end
    cfg = deepcopy(cfg)
    cfg.case_slugs = nothing
    cfg.registry_path = "registries/forward_sweep_cases_uncalibrated.yml"
    return cfg
end

function run_full_study!(opts::FullStudyOptions)
    va_full_study_merge_env!(opts)
    va_full_study_apply_figures_only_preset!(opts)
    va_full_study_apply_uncalibrated_preset!(opts)
    if opts.figures_only
        @info "Figures-only preset: skipping instantiate / forwards / calibration / naive / LES obs build"
    end
    if opts.uncalibrated_study
        @info "Uncalibrated study preset: forward registry" registry = opts.forward_registry skip_calib = opts.skip_calib forward_baseline_scm =
            opts.forward_baseline_scm
    end
    if !opts.skip_instantiate
        Pkg.instantiate()
    end

    if !opts.skip_calib && !opts.skip_les_observations_build
        @info "Ensuring LES observations (observations.jld2) for each calibration sweep YAML" configs =
            va_calibration_sweep_configs()
        va_ensure_les_observations_for_calibration_sweep!(_VA_ROOT, String.(va_calibration_sweep_configs()))
    elseif !opts.skip_calib && opts.skip_les_observations_build
        @info "Skipping LES observation build (--skip-les-observations or VA_SKIP_LES_OBSERVATIONS_BUILD)"
    end

    if !opts.skip_calib
        @info "EKI calibration sweep" configs = va_calibration_sweep_configs()
        calib_opts = va_eki_calibration_options_from_full_study(opts)
        run_calibration_sweep!(calib_opts; fail_fast = opts.calib_fail_fast)
    else
        @info "Skipping calibration sweep (--skip-calib)"
    end

    if !opts.skip_naive
        if opts.skip_calib
            @warn "Calibration was skipped; naive forwards need existing varfix-off EKI output (or --skip-naive)"
        end
        @info "Naive varfix-on forwards"
        naive_cfg = NaiveForwardConfig()
        naive_cfg.fail_fast = opts.naive_fail_fast
        naive_cfg.skip_done = opts.naive_skip_done
        run_naive_forwards!(naive_cfg)
    else
        @info "Skipping naive forwards (--skip-naive)"
    end

    if !opts.skip_forward
        if !opts.forward_baseline_scm && opts.skip_calib
            @warn "Calibration was skipped; EKI-parameter forward sweep expects merged member TOMLs on disk. Use --forward-baseline-scm for SCM-only forwards, or run calibration first."
        end
        @info "Forward grid (registry × N_quad × varfix × resolution ladder)" forward_skip_done =
            opts.forward_skip_done forward_resolution_ladder = opts.forward_resolution_ladder forward_baseline_scm =
            opts.forward_baseline_scm forward_registry = opts.forward_registry forward_case_slugs =
            opts.forward_case_slugs
        run_forward_sweep!(va_full_study_forward_sweep_config(opts); merge_env = false)
    else
        @info "Skipping forward grid (--skip-forward)"
    end

    if !opts.skip_figures
        cfg_fig_summary = va_full_study_forward_sweep_figure_config(opts)
        cfg_fig_cases = va_full_study_forward_sweep_config(opts)
        if opts.forward_figures_full_uncalibrated_registry && opts.uncalibrated_study
            @info "Forward-sweep figures: summary = full uncalibrated registry; per-case profile/scalar PNGs = forward filter only"
            va_plot_forward_sweep_comparisons!(_VA_ROOT, cfg_fig_cases)
            va_plot_forward_sweep_clw_plus_cli_summary!(_VA_ROOT, cfg_fig_summary)
            va_plot_forward_sweep_scalars_vs_nquad!(_VA_ROOT, cfg_fig_cases)
        else
            va_plot_forward_sweep_comparisons!(_VA_ROOT, cfg_fig_summary)
            va_plot_forward_sweep_clw_plus_cli_summary!(_VA_ROOT, cfg_fig_summary)
            va_plot_forward_sweep_scalars_vs_nquad!(_VA_ROOT, cfg_fig_summary)
        end
        if !opts.skip_calib && !opts.skip_naive
            va_plot_all_naive_vs_calibrated_varfix_on_profiles!(_VA_ROOT)
        end
        if !opts.skip_calib
            for c in va_calibration_sweep_configs()
                @info "Post-analysis figures" experiment_config = c
                va_run_post_analysis!(; experiment_dir = _VA_ROOT, experiment_config = c)
            end
            if !opts.skip_naive
                for c in va_naive_varfix_off_source_configs()
                    expc = va_load_experiment_config(_VA_ROOT, c)
                    naive_active = va_naive_forward_output_active(_VA_ROOT, c)
                    if !isdir(naive_active)
                        @warn "Skipping naive post-analysis (missing output_active)" experiment_config =
                            c naive_active
                        continue
                    end
                    ref = va_reference_output_active(_VA_ROOT, c)
                    fig = va_naive_post_analysis_figure_dir(_VA_ROOT, expc)
                    @info "Post-analysis figures (naive overlay)" experiment_config = c figure_root = fig
                    va_run_post_analysis!(;
                        experiment_dir = _VA_ROOT,
                        experiment_config = c,
                        profile_paths = String[ref, naive_active],
                        figure_root = fig,
                    )
                end
            end
        else
            @info "Skipping EKI post-analysis per calibration case (--skip-calib or uncalibrated study)"
        end
    else
        @info "Skipping figures (--skip-figures)"
    end

    @info "Full study pipeline finished" root = _VA_ROOT
    va_flush_stdio()
    return nothing
end

function run_full_study!(;
    forward_baseline_only::Bool = false,
    forward_baseline_scm::Bool = false,
    uncalibrated_study::Bool = false,
    figures_only::Bool = false,
    forward_registry::Union{Nothing, String} = nothing,
    kwargs...,
)
    o = FullStudyOptions(;
        kwargs...,
        forward_baseline_scm = forward_baseline_scm,
        uncalibrated_study = uncalibrated_study,
        figures_only = figures_only,
        forward_registry = forward_registry,
    )
    if forward_baseline_only
        o.forward_resolution_ladder = false
    end
    return run_full_study!(o)
end

"""Re-run post-analysis + forward-sweep figures only (no instantiate / forwards / EKI / naive)."""
run_full_study_figures_only!(; kwargs...) = run_full_study!(; figures_only = true, kwargs...)

"""Same as [`run_full_study_figures_only!`](@ref) with [`uncalibrated_study`](@ref) preset (baseline SCM registry)."""
run_full_study_uncalibrated_figures_only!(; kwargs...) =
    run_full_study!(; figures_only = true, uncalibrated_study = true, kwargs...)
