import ClimaAtmos as CA
include(joinpath(pkgdir(CA), "parameters", "create_parameters.jl"))

function create_parameter_set(::Type{FT}, parsed_args, default_args) where {FT}
    toml_dict = CP.create_toml_dict(
            FT;
            override_file = parsed_args["toml"],
            dict_type = "alias",
        )
    toml_dict, parsed_args =
        merge_parsed_args_with_toml(toml_dict, parsed_args, default_args)
    dt = FT(CA.time_to_seconds(parsed_args["dt"]))
    return if CA.is_column_edmf(parsed_args)
        overrides = (; MSLP = 100000.0, τ_precip = dt)
        create_climaatmos_parameter_set(toml_dict, parsed_args, overrides)
    elseif CA.is_column_without_edmf(parsed_args)
        overrides = (; τ_precip = dt)
        create_climaatmos_parameter_set(toml_dict, parsed_args, overrides)
    else
        overrides = (;
            R_d = 287.0,
            MSLP = 1.0e5,
            grav = 9.80616,
            Omega = 7.29212e-5,
            planet_radius = 6.371229e6,
            ρ_cloud_liq = 1e3,
            τ_precip = dt,
            qc_0 = 5e-6, # criterion for removal after supersaturation
        )
        create_climaatmos_parameter_set(toml_dict, parsed_args, overrides)
    end
end

"""
Merges parsed_args with the toml_dict generated from CLIMAParameters. 
Priority for clashes: parsed_args > toml_dict > default_args
Converts `nothing` to empty string, since CLIMAParameters does not support type Nothing.
The dictionary overrides existing toml_dict values if there are clashes.
"""
function merge_parsed_args_with_toml(toml_dict, parsed_args, default_args)
    toml_type(val::AbstractFloat) = "float"
    toml_type(val::Integer) = "integer"
    toml_type(val::Bool) = "bool"
    toml_type(val::String) = "string"
    toml_type(val::Symbol) = "string"
    toml_type(val::Nothing) = "string"
    toml_value(val::Nothing) = ""
    toml_value(val::Symbol) = String(val)
    toml_value(val) = val

    for (key, value) in parsed_args
        if haskey(default_args, key)
            if parsed_args[key] != default_args[key] ||
               !haskey(toml_dict.data, key)
                toml_dict.data[key] = Dict(
                    "type" => toml_type(value),
                    "value" => toml_value(value),
                    "alias" => key,
                )
            end
            parsed_args[key] = if toml_dict.data[key]["value"] == ""
                nothing
            elseif parsed_args[key] isa Symbol
                Symbol(toml_dict.data[key]["value"])
            else
                toml_dict.data[key]["value"]
            end
        end
    end
    return toml_dict, parsed_args
end
