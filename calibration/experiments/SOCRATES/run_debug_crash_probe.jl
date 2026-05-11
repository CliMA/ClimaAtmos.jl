import Dates
import NCDatasets as NC
import YAML

include(joinpath(@__DIR__, "model_interface.jl"))

function parse_int_arg(args, idx, default)
    return length(args) >= idx ? parse(Int, args[idx]) : default
end

function parse_str_arg(args, idx, default)
    return length(args) >= idx ? String(args[idx]) : default
end

function optional_str_arg(args, idx)
    if length(args) < idx
        return nothing
    end
    value = String(args[idx])
    return isempty(value) || value == "none" ? nothing : value
end

function find_first_nonfinite_time_index(nc_path::String, varname::String)
    NC.NCDataset(nc_path, "r") do ds
        haskey(ds, varname) || return 0
        v = ds[varname]
        nd = ndims(v)
        nt = size(v, nd)
        for t in 1:nt
            inds = ntuple(i -> i == nd ? t : Colon(), nd)
            slab = v[inds...]
            if any(x -> !isfinite(x), slab)
                return t
            end
        end
        return 0
    end
end

function summarize_nonfinite_outputs(config::Dict)
    out_active = joinpath(config["output_dir"], "output_active")
    if !isdir(out_active)
        println("No output_active directory found at: ", out_active)
        return
    end

    println("---- Non-finite scan for output_active ----")
    for (fname, varname) in (
        ("thetaa_10m_inst.nc", "thetaa"),
        ("hus_10m_inst.nc", "hus"),
        ("clw_10m_inst.nc", "clw"),
        ("ta_10m_inst.nc", "ta"),
        ("cli_10m_inst.nc", "cli"),
        ("husra_10m_inst.nc", "husra"),
        ("hussn_10m_inst.nc", "hussn"),
        ("arup_10m_inst.nc", "arup"),
        ("lwp_10m_inst.nc", "lwp"),
        ("clivi_10m_inst.nc", "clivi"),
        ("rwp_10m_inst.nc", "rwp"),
    )
        fpath = joinpath(out_active, fname)
        if !isfile(fpath)
            continue
        end
        first_bad = find_first_nonfinite_time_index(fpath, varname)
        println("$(fname): first_nonfinite_time_index=", first_bad)
    end

    checkpoints = filter(
        x -> endswith(x, ".hdf5") && startswith(x, "day"),
        readdir(out_active),
    )
    sort!(checkpoints)
    println("Checkpoints in output_active: ", length(checkpoints))
    if !isempty(checkpoints)
        println("Last checkpoint: ", checkpoints[end])
    end
end

function build_debug_case_config(
    ;
    iteration = 0,
    member = 3,
    case_idx = 1,
    t_end = "2hours",
    restart_file = nothing,
    dt = nothing,
    dt_save_state_to_disk = "5mins",
    diagnostics_period = "1mins",
)
    experiment_config = load_experiment_config()
    base_model_config = YAML.load_file(joinpath(@__DIR__, experiment_config["model_config"]))

    member_path = path_to_ensemble_member(experiment_config["output_dir"], iteration, member)
    isdir(member_path) || error("Member path not found: $member_path")

    cases = experiment_config["cases"]
    1 <= case_idx <= length(cases) || error("case_idx=$case_idx out of bounds")
    case = cases[case_idx]

    config = configure_member_case(base_model_config, member_path, case, case_idx)

    run_tag = "iter_$(lpad(iteration, 3, '0'))_member_$(lpad(member, 3, '0'))_case_$(case_idx)"
    if !isnothing(restart_file)
        restart_tag = replace(basename(restart_file), ".hdf5" => "")
        dt_tag = isnothing(dt) ? "defaultdt" : replace(String(dt), r"[^A-Za-z0-9]+" => "_")
        run_tag *= "_restart_$(restart_tag)_$(dt_tag)"
    end
    config["output_dir"] = joinpath(@__DIR__, "output", "debug_crash_probe", run_tag)
    mkpath(config["output_dir"])

    # Crash-probing settings
    config["t_start"] = "0secs"
    config["t_end"] = t_end
    if !isnothing(restart_file)
        config["restart_file"] = restart_file
    end
    if !isnothing(dt)
        config["dt"] = dt
    end
    config["log_progress"] = true
    config["log_to_file"] = true
    config["check_nan_every"] = 1
    config["dt_save_state_to_disk"] = dt_save_state_to_disk
    config["check_conservation"] = true
    config["output_default_diagnostics"] = false

    # Higher-frequency diagnostics to capture pre-crash evolution.
    config["diagnostics"] = [
        Dict("short_name" => ["thetaa", "hus", "clw", "ta", "cli", "husra", "hussn", "arup"], "period" => diagnostics_period),
        Dict("short_name" => ["lwp", "clivi", "rwp"], "period" => diagnostics_period),
    ]

    return config
end

function main(args = ARGS)
    iteration = parse_int_arg(args, 1, 0)
    member = parse_int_arg(args, 2, 3)
    case_idx = parse_int_arg(args, 3, 1)
    t_end = parse_str_arg(args, 4, "2hours")
    restart_file = optional_str_arg(args, 5)
    dt = optional_str_arg(args, 6)
    dt_save_state_to_disk = parse_str_arg(args, 7, "5mins")
    diagnostics_period = parse_str_arg(args, 8, "1mins")

    config = build_debug_case_config(
        ;
        iteration,
        member,
        case_idx,
        t_end,
        restart_file,
        dt,
        dt_save_state_to_disk,
        diagnostics_period,
    )

    println("Running debug crash probe with:")
    println("  output_dir            = ", config["output_dir"])
    println("  external_forcing_file = ", config["external_forcing_file"])
    println("  restart_file          = ", get(config, "restart_file", "<none>"))
    println("  dt                    = ", get(config, "dt", "<unset>"))
    println("  t_end                 = ", config["t_end"])
    println("  check_nan_every       = ", config["check_nan_every"])
    println("  dt_save_state_to_disk = ", config["dt_save_state_to_disk"])
    println("  diagnostics_period    = ", diagnostics_period)

    try
        run_single_case(config)
        println("Debug run completed without simulation_crashed.")
    catch err
        println("Debug run threw error: ", err)
        summarize_nonfinite_outputs(config)
        rethrow(err)
    end

    summarize_nonfinite_outputs(config)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
