flattened_cli_options = (
    "--TEST_NAME sphere/baroclinic_wave_rhoe",
    "--TEST_NAME sphere/baroclinic_wave_rhoe --FLOAT_TYPE Float32",
    "--TEST_NAME sphere/baroclinic_wave_rhotheta",
    "--TEST_NAME sphere/baroclinic_wave_rhoe_equilmoist --FLOAT_TYPE Float32",
    "--TEST_NAME sphere/baroclinic_wave_rhotheta_equilmoist --FLOAT_TYPE Float32",
    "--TEST_NAME sphere/held_suarez_rhoe --FLOAT_TYPE Float32",
    "--TEST_NAME sphere/held_suarez_rhotheta",
    "--TEST_NAME sphere/held_suarez_rhotheta --FLOAT_TYPE Float32",
    "--TEST_NAME sphere/held_suarez_rhoe_int --FLOAT_TYPE Float32",
    "--TEST_NAME sphere/held_suarez_rhoe_equilmoist --FLOAT_TYPE Float32",
    "--TEST_NAME sphere/held_suarez_rhoe_int_equilmoist --FLOAT_TYPE Float32",
)

# Convert to tuples:
flattened_cli_options = map(flattened_cli_options) do clio
    tup = split(clio, " --")
    length(tup) > 1 && (tup[2:end] = map(x -> " --" * x, tup[2:end]))
    tup
end

include(joinpath("..", "examples", "hybrid", "cli_options.jl"))

#= A string for the buildkite "label" field =#
function label_name(cli_options)
    s = join(cli_options)
    s = replace(s, "--FLOAT_TYPE Float32" => " Float32")
    s = replace(s, "--TEST_NAME" => "")
    s = replace(s, "sphere/" => "")
    s = replace(s, "_" => " ")
    s = replace(s, "rhoe" => "(ρe)")
    s = replace(s, "rhotheta" => "(ρθ)")
end

#= A string for the buildkite "artifact_path" field =#
function artifact_path(s, cli_options)
    parsed_args = parsed_args_from_cli_options(s, cli_options)
    return job_id_from_parsed_args(s, parsed_args)
end

"""
    parsed_args_from_cli_options(cli_options)

`parsed_args` given a string of all command line
interface options.
"""
function parsed_args_from_cli_options(s, cli_options)
    tup = split.(cli_options, " ")
    args = vcat(tup...)
    filter!(x -> !isempty(x), args)
    ArgParse.parse_args(args, s)
end

spaces = "      "
println("----------- Pipeline for examples:")
for cli_options in flattened_cli_options
    println("$(spaces)- label: \":computer:$(label_name(cli_options))\"")
    print("$(spaces)  command: \"julia --color=yes --project=examples ")
    print("examples/hybrid/driver.jl ")
    print(join(cli_options, " ") * "\"\n")
    println("$(spaces)  artifact_paths: \"$(artifact_path(s, cli_options))/*\"\n")
end

println("----------- Print JobIDs")
for cli_options in flattened_cli_options
    _parsed_args = parsed_args_from_cli_options(s, cli_options)
    @show job_id_from_parsed_args(s, _parsed_args)
end
