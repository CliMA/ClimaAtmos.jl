"""
    add_parameter_filepath!(config_dict, new_parameter_file)

Update `config_dict` to include `new_parameter_file` which override the existing
parameters.
"""
function add_parameter_filepath!(config_dict, new_parameter_file)
    tomls = get!(config_dict, "toml", String[])
    push!(tomls, new_parameter_file)
    return nothing
end

"""
    append_diagnostic_dicts!(config_dict, diagnostic_dicts::Vector)

Update `config_dict` by adding the diagnostics in `diagnostic_dicts`.
"""
function append_diagnostic_dicts!(config_dict, diagnostic_dicts::Vector)
    config_diagnostic_dicts = get!(config_dict, "diagnostics", Vector{Dict{Any, Any}}())
    append!(config_diagnostic_dicts, diagnostic_dicts)
    return nothing
end

"""
    replace_diagnostic_dicts!(config_dict, diagnostic_dicts::Vector)

Replace the diagnostincs in `config_dict` with `diagnostic_dicts`.
"""
function replace_diagnostic_dicts!(config_dict, diagnostic_dicts::Vector)
    config_diagnostic_dicts = get!(config_dict, "diagnostics", Vector{Dict{Any, Any}}())
    empty!(config_diagnostic_dicts)
    append!(config_diagnostic_dicts, diagnostic_dicts)
    return nothing
end
