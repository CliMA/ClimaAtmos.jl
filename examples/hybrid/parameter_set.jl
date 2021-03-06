import ClimaAtmos

include(joinpath(pkgdir(ClimaAtmos), "parameters", "create_parameters.jl"))

function create_parameter_set(::Type{FT}, parsed_args, namelist) where {FT}
    toml_dict = CP.create_toml_dict(FT; dict_type = "alias")
    dt = FT(time_to_seconds(parsed_args["dt"]))
    return if is_column_edmf(parsed_args)
        tc_parameter_set(toml_dict, dt)
    elseif is_column_radiative_equilibrium(parsed_args)
        overrides = (; τ_precip = dt)
        create_climaatmos_parameter_set(toml_dict, overrides)
    else
        baro_wave_parameter_set(toml_dict, dt)
    end
end

function baro_wave_parameter_set(toml_dict, dt)
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
    create_climaatmos_parameter_set(toml_dict, overrides)
end

function tc_parameter_set(toml_dict, dt)
    overrides = (; MSLP = 100000.0, τ_precip = dt)
    create_climaatmos_parameter_set(toml_dict, overrides)
end
