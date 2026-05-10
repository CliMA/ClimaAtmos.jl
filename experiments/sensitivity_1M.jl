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
# Configuration
# ---------------------------------------------------------------------------

const ATMOS_ROOT = dirname(@__DIR__)  # ClimaAtmos.jl root
const SCRATCH_DIR = joinpath(ATMOS_ROOT, "experiments", "sensitivity_scratch")
const OUTPUT_ROOT = joinpath(ATMOS_ROOT, "experiments", "sensitivity_output")

const CASES = [
    ("dycoms", "config/model_configs/prognostic_edmfx_dycoms_rf02_column.yml"),
    ("rico",   "config/model_configs/prognostic_edmfx_rico_implicit_column.yml"),
    ("trmm",   "config/model_configs/prognostic_edmfx_trmm_implicit_column.yml"),
]

# Each entry: (abbrev, full_param_name, sweep_values)
const SWEEP_PARAMS = [
    (
        "cet",
        "condensation_evaporation_timescale",
        [1.0, 10.0, 100.0, 1000.0],
    ),
    (
        "rat",
        "rain_autoconversion_timescale",
        [1.0, 10.0, 100.0, 1000.0],
    ),
    (
        "qth",
        "cloud_liquid_water_specific_humidity_autoconversion_threshold",
        [1e-6, 1e-5, 1e-4, 1e-3],
    ),
    (
        "chia",
        "rain_cross_section_size_relation_coefficient_chia",
        [0.1, 1.0, 10.0, 100.0],
    ),
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
# Main loop
# ---------------------------------------------------------------------------

function run_sensitivity(; dry_run::Bool = false, n_parallel::Int = 1)

    mkpath(SCRATCH_DIR)
    mkpath(OUTPUT_ROOT)

    # Collect all (cmd, label) pairs
    jobs = Tuple{Cmd, String}[]

    for (case_abbrev, base_config) in CASES
        # Base TOML path — must be absolute so the override YAML resolves it correctly
        base_toml = joinpath(ATMOS_ROOT, "toml", "prognostic_edmfx_1M.toml")

        for (param_abbrev, param_name, values) in SWEEP_PARAMS
            for v in values
                vstr    = value_str(v)
                job_id  = "$(case_abbrev)_$(param_abbrev)_$(vstr)"
                label   = job_id

                override_toml = joinpath(SCRATCH_DIR, "$(job_id).toml")
                override_yaml = joinpath(SCRATCH_DIR, "$(job_id).yml")
                output_dir    = joinpath(OUTPUT_ROOT, job_id)

                write_override_toml(override_toml, base_toml, param_name, v)
                write_override_yaml(override_yaml, job_id, output_dir, override_toml)

                cmd = build_command(base_config, override_yaml)
                push!(jobs, (cmd, label))
            end
        end
    end

    n_jobs = length(jobs)
    println("=== Sensitivity sweep: $n_jobs simulations ===")
    println("Cases:  ", join(first.(CASES), ", "))
    println("Params: ", join(first.(SWEEP_PARAMS), ", "))
    println("Output: $OUTPUT_ROOT")
    println("Parallel: $n_parallel")
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
            run(cmd)   # throws on non-zero exit
            elapsed = round(time() - t0, digits = 1)
            println("[$i/$n_jobs] Done: $label ($elapsed s)")
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
                    run(cmd)
                    println("  DONE:  $label")
                end)
                for (cmd, label) in batch
            ]
            wait.(tasks)
            idx += n_parallel
        end
    end

    println()
    println("=== All simulations complete ===")
    println("Results in: $OUTPUT_ROOT")
    println()
    println("Output directory layout:")
    for (case_abbrev, _) in CASES
        for (param_abbrev, _, values) in SWEEP_PARAMS
            for v in values
                println("  $OUTPUT_ROOT/$(case_abbrev)_$(param_abbrev)_$(value_str(v))/")
            end
        end
    end
end

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

let
    dry_run   = "--dry-run" in ARGS
    n_parallel = let idx = findfirst(==("--parallel"), ARGS)
        idx !== nothing && idx < length(ARGS) ? parse(Int, ARGS[idx + 1]) : 1
    end
    run_sensitivity(; dry_run, n_parallel)
end
