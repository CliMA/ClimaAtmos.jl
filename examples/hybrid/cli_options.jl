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
