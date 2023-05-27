import YAML

"""
    commands_from_yaml(yaml_file; filter_name=nothing)

A vector of strings containing the `command`s in a (buildkite)
yaml file, `yaml_file`. `filter_name` can be used to filter out
required substrings from each elements of the vector.
"""
function commands_from_yaml(yaml_file; filter_name = nothing)
    data = YAML.load_file(yaml_file)
    steps = filter(x -> x isa Dict && haskey(x, "steps"), data["steps"])
    cmds = map(x -> x["steps"], steps)
    cmds = vcat(cmds...)
    cmds = map(x -> x["command"], cmds)
    cmds = vcat(cmds...)
    cmds = map(x -> split(x, "\n"), cmds) # handle commands that run multiple scripts
    cmds = vcat(cmds...)
    cmds = map(x -> x isa AbstractString ? x : join(x), cmds)
    if filter_name ≠ nothing
        filter!(x -> occursin(filter_name, x), cmds)
    end
    return cmds
end
