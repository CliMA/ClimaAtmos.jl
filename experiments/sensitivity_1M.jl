"""
Sensitivity sweep for 1M microphysics parameters.

Sweeps 4 parameters one-at-a-time across three SCM cases:
  - dycoms  : prognostic_edmfx_dycoms_rf02_column
  - rico    : prognostic_edmfx_rico_implicit_column
  - trmm    : prognostic_edmfx_trmm_implicit_column

Parameters swept (others held at their base-config defaults):
  cet   condensation_evaporation_timescale           [1, 10, 100, 1000]  s
  rat   rain_autoconversion_timescale                [1, 10, 100, 1000]  s
  qth   cloud_liquid_water_specific_humidity_...     [1e-6,1e-5,1e-4,1e-3] kg/kg
  chia  rain_cross_section_size_relation_..._chia    [0.1, 1, 10, 100]   -

Run history (slurm-227964.out):
  ✅ dycoms × {cet,rat,qth,chia} × all values  (16 jobs, completed)
  ✅ rico   × {cet,rat,qth,chia} × all values  (16 jobs, completed)
  ❌ trmm   × all params × all values           (16 jobs, not run)
       trmm_cet_1 crashed with DomainError in log() inside
       saturation_vapor_pressure — cet=1s is too aggressive for TRMM.
       The remaining 15 trmm jobs never started (sequential runner aborted).

Output:
  experiments/sensitivity_output/<case>_<param>_<value>/
  (output_dir_style=RemovePreexisting so re-runs overwrite cleanly)

Usage:
  julia --project=.buildkite experiments/sensitivity_1M.jl [--dry-run] [--parallel N]

  --dry-run   Print commands without running them
  --parallel N Run N simulations concurrently (default: 1 = sequential)
"""

using TOML

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

const ATMOS_ROOT = dirname(@__DIR__)  # ClimaAtmos.jl root
const SCRATCH_DIR = joinpath(ATMOS_ROOT, "experiments", "sensitivity_scratch")
const OUTPUT_ROOT = joinpath(ATMOS_ROOT, "experiments", "sensitivity_output")

# ---------------------------------------------------------------------------
# Full sweep configuration  (COMMENTED OUT — completed in slurm-227964)
# To re-run everything, uncomment CASES + SWEEP_PARAMS and call run_sensitivity()
# ---------------------------------------------------------------------------

# const CASES = [
#     ("dycoms", "config/model_configs/prognostic_edmfx_dycoms_rf02_column.yml"),
#     ("rico",   "config/model_configs/prognostic_edmfx_rico_implicit_column.yml"),
#     ("trmm",   "config/model_configs/prognostic_edmfx_trmm_implicit_column.yml"),
# ]
#
# # Each entry: (abbrev, full_param_name, sweep_values)
# const SWEEP_PARAMS = [
#     (
#         "cet",
#         "condensation_evaporation_timescale",
#         [1.0, 10.0, 100.0, 1000.0],
#     ),
#     (
#         "rat",
#         "rain_autoconversion_timescale",
#         [1.0, 10.0, 100.0, 1000.0],
#     ),
#     (
#         "qth",
#         "cloud_liquid_water_specific_humidity_autoconversion_threshold",
#         [1e-6, 1e-5, 1e-4, 1e-3],
#     ),
#     (
#         "chia",
#         "rain_cross_section_size_relation_coefficient_chia",
#         [0.1, 1.0, 10.0, 100.0],
#     ),
# ]

# ---------------------------------------------------------------------------
# Missing jobs — the 16 TRMM simulations that still need to run
#
# Each entry: (case_abbrev, base_config_path, param_abbrev, param_name, value)
#
# ⚠️  trmm_cet_1 previously crashed: DomainError(log of negative number) inside
#     saturation_vapor_pressure at ~2h46m sim time.
#     condensation_evaporation_timescale = 1 s is physically too aggressive for
#     TRMM deep convection.  It is included here so the failure is on record;
#     expect it to crash again.
# ---------------------------------------------------------------------------

const TRMM_CONFIG = "config/model_configs/prognostic_edmfx_trmm_implicit_column.yml"

const MISSING_JOBS = [
    # param: condensation_evaporation_timescale
    # ⚠️ trmm_cet_1 previously crashed (see note above)
    ("trmm", TRMM_CONFIG, "cet", "condensation_evaporation_timescale",                             1.0),
    ("trmm", TRMM_CONFIG, "cet", "condensation_evaporation_timescale",                            10.0),
    ("trmm", TRMM_CONFIG, "cet", "condensation_evaporation_timescale",                           100.0),
    ("trmm", TRMM_CONFIG, "cet", "condensation_evaporation_timescale",                          1000.0),
    # param: rain_autoconversion_timescale
    ("trmm", TRMM_CONFIG, "rat", "rain_autoconversion_timescale",                                  1.0),
    ("trmm", TRMM_CONFIG, "rat", "rain_autoconversion_timescale",                                 10.0),
    ("trmm", TRMM_CONFIG, "rat", "rain_autoconversion_timescale",                                100.0),
    ("trmm", TRMM_CONFIG, "rat", "rain_autoconversion_timescale",                               1000.0),
    # param: cloud_liquid_water_specific_humidity_autoconversion_threshold
    ("trmm", TRMM_CONFIG, "qth", "cloud_liquid_water_specific_humidity_autoconversion_threshold",  1e-6),
    ("trmm", TRMM_CONFIG, "qth", "cloud_liquid_water_specific_humidity_autoconversion_threshold",  1e-5),
    ("trmm", TRMM_CONFIG, "qth", "cloud_liquid_water_specific_humidity_autoconversion_threshold",  1e-4),
    ("trmm", TRMM_CONFIG, "qth", "cloud_liquid_water_specific_humidity_autoconversion_threshold",  1e-3),
    # param: rain_cross_section_size_relation_coefficient_chia
    ("trmm", TRMM_CONFIG, "chia", "rain_cross_section_size_relation_coefficient_chia",             0.1),
    ("trmm", TRMM_CONFIG, "chia", "rain_cross_section_size_relation_coefficient_chia",             1.0),
    ("trmm", TRMM_CONFIG, "chia", "rain_cross_section_size_relation_coefficient_chia",            10.0),
    ("trmm", TRMM_CONFIG, "chia", "rain_cross_section_size_relation_coefficient_chia",           100.0),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

"""Format a Float64 into a compact string safe for file/dir names.
Examples: 1.0 → "1", 10.0 → "10", 1e-6 → "1em6", 16.605 → "16p605"
"""
function value_str(v::Float64)::String
    if v == round(v) && abs(v) < 1e6
        return string(Int(round(v)))
    end
    # Try scientific notation shorthand
    e = floor(Int, log10(abs(v)))
    m = v / 10.0^e
    if m ≈ round(m)
        return "$(Int(round(m)))em$(abs(e))"
    end
    # Fall back to replacing . with p and - with m
    s = string(v)
    s = replace(s, "." => "p", "e-" => "em", "e+" => "e")
    return s
end

"""
Write a combined TOML: starts from all entries in `base_toml_path`, then
overrides (or adds) `param_name = value`. This avoids the ClimaParams
`merge_toml_files` "Duplicate TOML entry" error that arises when two files
in the `toml:` list both define the same key.
"""
function write_override_toml(path::String, base_toml_path::String,
                             param_name::String, value::Float64)
    # Parse the base TOML — Dict{String, Any} where each entry looks like
    # {"value" => ..., "type" => ..., "description" => ...}
    base = TOML.parsefile(base_toml_path)
    # Override the swept parameter (preserve any extra keys like "type")
    if haskey(base, param_name)
        base[param_name]["value"] = value
    else
        base[param_name] = Dict{String, Any}("value" => value)
    end
    open(path, "w") do io
        TOML.print(io, base)
    end
end

"""Write a minimal override YAML that sets job_id, output_dir, and toml list."""
function write_override_yaml(path::String, job_id::String, output_dir::String,
                             override_toml::String)
    open(path, "w") do io
        println(io, "job_id: \"$job_id\"")
        println(io, "output_dir: \"$output_dir\"")
        println(io, "output_dir_style: RemovePreexisting")
        println(io, "toml:")
        println(io, "  - $override_toml")
    end
end

"""Build the julia command for one simulation run."""
function build_command(base_config::String, override_yaml::String)::Cmd
    return Cmd([
        "julia", "+1.11",
        "--project=$(joinpath(ATMOS_ROOT, ".buildkite"))",
        joinpath(ATMOS_ROOT, ".buildkite", "ci_driver.jl"),
        "--config_file", joinpath(ATMOS_ROOT, base_config),
        "--config_file", override_yaml,
    ])
end

# ---------------------------------------------------------------------------
# Runner for missing jobs
# ---------------------------------------------------------------------------

function run_missing(; dry_run::Bool = false, n_parallel::Int = 1)

    mkpath(SCRATCH_DIR)
    mkpath(OUTPUT_ROOT)

    base_toml = joinpath(ATMOS_ROOT, "toml", "prognostic_edmfx_1M.toml")

    # Build (cmd, label) pairs from MISSING_JOBS
    jobs = Tuple{Cmd, String}[]

    for (case_abbrev, base_config, param_abbrev, param_name, v) in MISSING_JOBS
        vstr         = value_str(v)
        job_id       = "$(case_abbrev)_$(param_abbrev)_$(vstr)"
        override_toml = joinpath(SCRATCH_DIR, "$(job_id).toml")
        override_yaml = joinpath(SCRATCH_DIR, "$(job_id).yml")
        output_dir    = joinpath(OUTPUT_ROOT, job_id)

        write_override_toml(override_toml, base_toml, param_name, v)
        write_override_yaml(override_yaml, job_id, output_dir, override_toml)

        cmd = build_command(base_config, override_yaml)
        push!(jobs, (cmd, job_id))
    end

    n_jobs = length(jobs)
    println("=== Missing TRMM jobs: $n_jobs simulations ===")
    println("Output: $OUTPUT_ROOT")
    println("Parallel: $n_parallel")
    println()
    println("⚠️  Note: trmm_cet_1 (condensation_evaporation_timescale=1s) previously")
    println("   crashed with DomainError in log(). Expect it to crash again.")
    println()

    if dry_run
        println("--- DRY RUN: commands that would be executed ---")
        for (cmd, label) in jobs
            println("[$label]")
            println("  ", join(cmd.exec, " "))
            println()
        end
        return
    end

    # Execute sequentially or with simple task-based parallelism
    if n_parallel == 1
        for (i, (cmd, label)) in enumerate(jobs)
            println("[$i/$n_jobs] Running: $label")
            t0 = time()
            try
                run(cmd)
                elapsed = round(time() - t0, digits = 1)
                println("[$i/$n_jobs] Done: $label ($elapsed s)")
            catch e
                elapsed = round(time() - t0, digits = 1)
                println("[$i/$n_jobs] FAILED: $label ($elapsed s) — $e")
                println("  Continuing to next job...")
            end
            println()
        end
    else
        # Run in batches of n_parallel
        idx = 1
        while idx <= n_jobs
            batch = jobs[idx:min(idx + n_parallel - 1, n_jobs)]
            println("Launching batch $(idx)–$(idx + length(batch) - 1) of $n_jobs")
            tasks = [
                (@async begin
                    println("  START: $label")
                    try
                        run(cmd)
                        println("  DONE:  $label")
                    catch e
                        println("  FAILED: $label — $e")
                    end
                end)
                for (cmd, label) in batch
            ]
            wait.(tasks)
            idx += n_parallel
        end
    end

    println()
    println("=== Run complete ===")
    println("Results in: $OUTPUT_ROOT")
end

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

let
    dry_run    = "--dry-run" in ARGS
    n_parallel = let idx = findfirst(==("--parallel"), ARGS)
        idx !== nothing && idx < length(ARGS) ? parse(Int, ARGS[idx + 1]) : 1
    end
    run_missing(; dry_run, n_parallel)
end
