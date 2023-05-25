import .TurbulenceConvection as TC
import .TurbulenceConvection.Parameters as TCP
import CLIMAParameters as CP
import RRTMGP.Parameters as RP
import SurfaceFluxes as SF
import SurfaceFluxes.UniversalFunctions as UF
import ClimaCore
import ClimaCore as CC
import Insolation.Parameters as IP
import Thermodynamics as TD
import CloudMicrophysics as CM

function override_climaatmos_defaults(
    defaults::NamedTuple,
    overrides::NamedTuple,
)
    intersect_keys = intersect(keys(defaults), keys(overrides))
    intersect_vals = getproperty.(Ref(overrides), intersect_keys)
    intersect_overrides = (; zip(intersect_keys, intersect_vals)...)
    return merge(defaults, intersect_overrides)
end

function create_climaatmos_parameter_set(
    toml_dict::CP.AbstractTOMLDict,
    parsed_args,
    overrides::NamedTuple = NamedTuple(),
)
    FT = CP.float_type(toml_dict)
    FTD = FT # can change to Dual for testing duals

    aliases = string.(fieldnames(TD.Parameters.ThermodynamicsParameters))
    pairs = CP.get_parameter_values!(toml_dict, aliases, "Thermodynamics")
    pairs = override_climaatmos_defaults((; pairs...), overrides)
    thermo_params = TD.Parameters.ThermodynamicsParameters{FTD}(; pairs...)
    TP = typeof(thermo_params)

    aliases = string.(fieldnames(CM.Parameters.CloudMicrophysicsParameters))
    aliases = setdiff(aliases, ["thermo_params"])
    pairs = CP.get_parameter_values!(toml_dict, aliases, "CloudMicrophysics")
    pairs = override_climaatmos_defaults((; pairs...), overrides)
    microphys_params = CM.Parameters.CloudMicrophysicsParameters{FTD, TP}(;
        pairs...,
        thermo_params,
    )
    MP = typeof(microphys_params)

    aliases = [
        "Pr_0_Businger",
        "a_m_Businger",
        "a_h_Businger",
        "ζ_a_Businger",
        "γ_Businger",
    ]
    pairs = CP.get_parameter_values!(toml_dict, aliases, "UniversalFunctions")
    pairs = (; pairs...) # convert to NamedTuple
    pairs = (;
        Pr_0 = pairs.Pr_0_Businger,
        a_m = pairs.a_m_Businger,
        a_h = pairs.a_h_Businger,
        ζ_a = pairs.ζ_a_Businger,
        γ = pairs.γ_Businger,
    )
    pairs = override_climaatmos_defaults((; pairs...), overrides)
    ufp = UF.BusingerParams{FTD}(; pairs...)
    UFP = typeof(ufp)

    pairs = CP.get_parameter_values!(
        toml_dict,
        ["von_karman_const"],
        "SurfaceFluxesParameters",
    )
    pairs = override_climaatmos_defaults((; pairs...), overrides)
    surf_flux_params = SF.Parameters.SurfaceFluxesParameters{FTD, UFP, TP}(;
        pairs...,
        ufp,
        thermo_params,
    )
    SFP = typeof(surf_flux_params)

    aliases = string.(fieldnames(TCP.TurbulenceConvectionParameters))
    pairs = CP.get_parameter_values!(toml_dict, aliases, "EDMF")
    pairs = override_climaatmos_defaults((; pairs...), overrides)
    tc_params = TCP.TurbulenceConvectionParameters{FTD, MP, SFP}(;
        pairs...,
        microphys_params,
        surf_flux_params,
    )

    aliases = string.(fieldnames(RP.RRTMGPParameters))
    pairs = CP.get_parameter_values!(toml_dict, aliases, "RRTMGP")
    params = override_climaatmos_defaults((; pairs...), overrides) # overrides
    rrtmgp_params = RP.RRTMGPParameters{FTD}(; params...)

    aliases = string.(fieldnames(IP.InsolationParameters))
    pairs = CP.get_parameter_values!(toml_dict, aliases, "Insolation")
    params = override_climaatmos_defaults((; pairs...), overrides) # overrides
    insolation_params = IP.InsolationParameters{FTD}(; params...)

    pairs = CP.get_parameter_values!(
        toml_dict,
        ["Omega", "planet_radius", "astro_unit"],
        "ClimaAtmos",
    )
    pairs = (; pairs...) # convert to NamedTuple
    pairs = override_climaatmos_defaults((; pairs...), overrides)

    param_set = CAP.ClimaAtmosParameters(;
        ug = FTD(1.0), # for Ekman problem
        vg = FTD(0.0), # for Ekman problem
        f = FTD(5e-5), # for Ekman problem
        Cd = FTD(0.01 / (2e2 / 30)), # for Ekman problem
        Omega = FTD(pairs.Omega),
        planet_radius = FTD(pairs.planet_radius),
        astro_unit = FTD(pairs.astro_unit),
        f_plane_coriolis_frequency = FTD(0),
        thermodynamics_params = thermo_params,
        microphysics_params = microphys_params,
        insolation_params = insolation_params,
        rrtmgp_params = rrtmgp_params,
        surfacefluxes_params = surf_flux_params,
        turbconv_params = tc_params,
    )
    # logfilepath = joinpath(@__DIR__, "logfilepath_$FT.toml")
    # CP.log_parameter_information(toml_dict, logfilepath)
    return param_set
end

# TODO: unify these parameters and refactor this method.
function create_parameter_set(config::AtmosConfig)
    (; toml_dict, parsed_args) = config
    FT = eltype(config)
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
