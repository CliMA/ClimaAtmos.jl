import ArgParse

function argparse_settings()
    s = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table! s begin
        "--config_file"
        help = "A yaml file used to set model configurations. If this is called multiple times, all config files will be merged."
        action = :append_arg
        arg_type = String
        default = [default_config_file]
        "--job_id"
        help = "A unique job identifier, among all possible (parallel) running jobs."
        arg_type = String
        default = config_id_from_config_file(default_config_file)
    end
    return s
end

to_named_tuple(dict::Dict) = (; (Symbol(k) => v for (k, v) in dict)...)
commandline_kwargs() = to_named_tuple(parse_commandline())

parse_commandline() = parse_commandline(argparse_settings())
parse_commandline(s) = ArgParse.parse_args(ARGS, s)
parse_commandline(args, s) = ArgParse.parse_args(args, s)

function cli_defaults(s::ArgParse.ArgParseSettings)
    defaults = Dict()
    # TODO: Don't use ArgParse internals
    for arg in s.args_table.fields
        defaults[arg.dest_name] = arg.default
    end
    return defaults
end
