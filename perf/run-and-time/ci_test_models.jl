#=
This file is intended to be used as a smoke test in CI to verify that all
reduced component models run correctly.
=#
using Logging

# List of files to skipe
case_files = Set([
    "models/hyperdiff_tendency.jl",
    "models/implicit_tendency.jl",
    "models/set_cloud_frac.jl",
    "models/set_CM_cache.jl",
    "models/sgs.jl",
    ])

has_faliures = false


for model in case_files
    # It overrides defined `case_setup` and `case_run` functions
    include(model)
    print("$model : ")

    # Do not print the log messages
    with_logger(NullLogger()) do
        try
            s = case_setup()
            case_run(s)
            printstyled("OK - it runs\n"; color = :green, bold = true)
        catch e
            printstyled("$model : ERROR\n"; color = :red, bold = true)
            print(" - $e\n")
            has_faliures = true
        end
    end
end

exit_code = has_faliures ? 1 : 0
exit(exit_code)
