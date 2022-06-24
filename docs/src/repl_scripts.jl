const ca_dir = joinpath(@__DIR__, "..", "..")
include(joinpath(ca_dir, "examples", "hybrid", "cli_options.jl"))
using PrettyTables
(s, parsed_args) = parse_commandline();

buildkite_commands = readlines(joinpath(ca_dir, ".buildkite", "pipeline.yml"));
filter!(x -> occursin("driver.jl", x), buildkite_commands)

@assert length(buildkite_commands) > 0

buildkite_flags = Dict()
for bkcs in buildkite_commands
    job_id = first(split(last(split(bkcs, "--job_id ")), " "))
    println("### Buildkite job `$job_id`")
    print_repl_script(bkcs)
    println()
end
