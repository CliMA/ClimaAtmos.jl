import ArgParse

function argparse_settings()
    s = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table s begin
        "--config_file"
        help = "A yaml file used to set model configurations"
        arg_type = String
    end
    return s
end

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
