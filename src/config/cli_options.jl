import ArgParse

"""
    argparse_settings()

Build the `ArgParse.ArgParseSettings` for the ClimaAtmos command line.

Two options are supported: `--config_file` (repeatable; all given files are merged in
order) and `--job_id` (falls back to the `job_id` config key or a name derived from
the config file names).
"""
function argparse_settings()
    s = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table! s begin
        #! format: off
        "--config_file"
        help = "A yaml file used to set model configurations. If this is called multiple times, all config files will be merged."
        action = :append_arg
        arg_type = String
        default = [default_config_file]
        "--job_id"
        help = "A unique job identifier, among all possible (parallel) running jobs. If omitted, it is taken from the `job_id` key in the configuration, or derived from the config file names."
        arg_type = String
        default = job_id_from_config_file(default_config_file)
        #! format: on
    end
    return s
end

to_named_tuple(dict::Dict) = (; (Symbol(k) => v for (k, v) in dict)...)

"""
    commandline_kwargs()

Parse `ARGS` with `argparse_settings` and return the result as a `NamedTuple`,
suitable for splatting into `AtmosConfig`.
"""
commandline_kwargs() = to_named_tuple(parse_commandline())

"""
    parse_commandline()
    parse_commandline(s)
    parse_commandline(args, s)

Parse command-line arguments (`ARGS` by default, or an explicit `args` vector) with the
`ArgParse.ArgParseSettings` `s` (default: `argparse_settings()`).
"""
parse_commandline() = parse_commandline(argparse_settings())
parse_commandline(s) = ArgParse.parse_args(ARGS, s)
parse_commandline(args, s) = ArgParse.parse_args(args, s)

"""
    cli_defaults(s::ArgParse.ArgParseSettings)

Return a `Dict` mapping each command-line option name to its default value.
"""
function cli_defaults(s::ArgParse.ArgParseSettings)
    defaults = Dict()
    # TODO: Don't use ArgParse internals
    for arg in s.args_table.fields
        defaults[arg.dest_name] = arg.default
    end
    return defaults
end
