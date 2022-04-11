import ArgParse
s = ArgParse.ArgParseSettings()
ArgParse.@add_arg_table s begin
    "--FLOAT_TYPE"
    help = "Float type"
    arg_type = String
    default = "Float64" # TODO: default to Float32
    "--t_end"
    help = "Simulation end time"
    arg_type = Float64
    "--regression_test"
    help = "(Bool) perform regression test"
    arg_type = Bool
    default = true
    "--TEST_NAME" # TODO: change to "JobName"?
    help = "Job name"
    arg_type = String
end

parsed_args = ArgParse.parse_args(ARGS, s)

function cli_defaults(s::ArgParse.ArgParseSettings)
    defaults = Dict()
    # TODO: Don't use ArgParse internals
    for arg in s.args_table.fields
        defaults[arg.dest_name] = arg.default
    end
    return defaults
end

#= Use the job ID for the output folder =#
function output_directory(
    s::ArgParse.ArgParseSettings,
    parsed_args = ArgParse.parse_args(ARGS, s),
)
    return job_id_from_parsed_args(s, parsed_args)
end

"""
    job_id_from_parsed_args(
        s::ArgParseSettings,
        parsed_args = ArgParse.parse_args(ARGS, s)
    )

Returns a unique name (`String`) given
 - `s::ArgParse.ArgParseSettings` The arg parse settings
 - `parsed_args` The parse arguments

The `ArgParseSettings` are used for truncating
this string based on the default values.
"""
job_id_from_parsed_args(s, parsed_args = ArgParse.parse_args(ARGS, s)) =
    job_id_from_parsed_args(cli_defaults(s), parsed_args)

function job_id_from_parsed_args(defaults::Dict, parsed_args)
    _parsed_args = deepcopy(parsed_args)
    s = ""
    for k in keys(_parsed_args)
        # Skip defaults to alleviate verbose names
        defaults[k] == _parsed_args[k] && continue

        if _parsed_args[k] isa String
            # We don't need keys if the value is a string
            # (alleviate verbose names)
            s *= _parsed_args[k]
        else
            s *= k * "=" * string(_parsed_args[k])
        end
        s *= "_"
    end
    s = replace(s, "/" => "_")
    s = strip(s, '_')
    return s
end
