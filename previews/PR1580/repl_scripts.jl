const ca_dir = joinpath(@__DIR__, "..", "..")
include(joinpath(ca_dir, "examples", "hybrid", "cli_options.jl"))
using PrettyTables
(s, parsed_args) = parse_commandline();
include(joinpath(ca_dir, "src", "utils", "yaml_helper.jl"))

buildkite_yaml = joinpath(ca_dir, ".buildkite", "pipeline.yml");
buildkite_commands =
    commands_from_yaml(buildkite_yaml; filter_name = "driver.jl")
@assert length(buildkite_commands) > 0

buildkite_flags = Dict()
for bkcs in buildkite_commands
    job_id = first(split(last(split(bkcs, "--job_id ")), " "))
    println("### Buildkite job `$job_id`")
    print_repl_script(bkcs)
    println()
end
