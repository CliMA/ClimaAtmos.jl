import OrderedCollections
import JSON
import NCRegressionTests
import ArgParse

function parse_commandline()
    s = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table s begin
        "--job_id"
        help = "Uniquely identifying string for a particular job"
        arg_type = String
        "--out_dir"
        help = "Output data directory"
        arg_type = String
    end
    parsed_args = ArgParse.parse_args(ARGS, s)
    return (s, parsed_args)
end

function get_params()
    (s, parsed_args) = parse_commandline()
    job_id = parsed_args["job_id"]
    out_dir = parsed_args["out_dir"]
    return (; job_id, out_dir)
end

(; job_id, out_dir) = get_params()
include(joinpath(@__DIR__, "mse_tables.jl"))
best_mse = all_best_mse[job_id]
best_mse_string =
    Dict(map(x -> string(x) => best_mse[x], collect(keys(best_mse))))
computed_mse_filename = joinpath(out_dir, "computed_mse.json")
computed_mse = JSON.parsefile(
    computed_mse_filename;
    dicttype = OrderedCollections.OrderedDict,
)

NCRegressionTests.test_mse(computed_mse, best_mse_string)
