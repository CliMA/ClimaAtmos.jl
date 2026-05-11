import YAML

include(joinpath(@__DIR__, "model_interface.jl"))

function parse_int_arg(args, idx, default)
    return length(args) >= idx ? parse(Int, args[idx]) : default
end

function main(args = ARGS)
    # Defaults match the member/case where NaNs were previously observed.
    iteration = parse_int_arg(args, 1, 0)
    member = parse_int_arg(args, 2, 3)
    case_idx = parse_int_arg(args, 3, 1)

    experiment_config = load_experiment_config()
    base_model_config = YAML.load_file(joinpath(@__DIR__, experiment_config["model_config"]))

    member_path = path_to_ensemble_member(experiment_config["output_dir"], iteration, member)
    isdir(member_path) || error("Member path not found: $member_path")

    cases = experiment_config["cases"]
    1 <= case_idx <= length(cases) || error("case_idx=$case_idx out of bounds")
    case = cases[case_idx]

    config = configure_member_case(base_model_config, member_path, case, case_idx)

    # Only switch the turbulence-convection closure mode; keep normal dt/t_end/checkpoint/diagnostics.
    config["turbconv"] = "diagnostic_edmfx"

    run_tag = "iter_$(lpad(iteration, 3, '0'))_member_$(lpad(member, 3, '0'))_case_$(case_idx)"
    config["output_dir"] = joinpath(@__DIR__, "output", "diagnostic_edmfx_from_start", run_tag)
    mkpath(config["output_dir"])

    println("Running SOCRATES with diagnostic EDMFX from start:")
    println("  output_dir            = ", config["output_dir"])
    println("  external_forcing_file = ", config["external_forcing_file"])
    println("  turbconv              = ", config["turbconv"])
    println("  dt                    = ", get(config, "dt", "<unset>"))
    println("  t_end                 = ", get(config, "t_end", "<unset>"))

    run_single_case(config)

    println("Run completed successfully.")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
