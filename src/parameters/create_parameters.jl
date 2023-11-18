import CLIMAParameters as CP
import RRTMGP.Parameters as RP
import SurfaceFluxes as SF
import SurfaceFluxes.UniversalFunctions as UF
import Insolation.Parameters as IP
import Thermodynamics as TD
import CloudMicrophysics as CM

function create_parameter_set(config::AtmosConfig)
    # Helper function that creates a parameter struct. If a struct has nested
    # parameter structs, they must be passed to subparam_structs as a NamedTuple.
    function create_parameter_struct(param_struct; subparam_structs = (;))
        aliases = string.(fieldnames(param_struct))
        aliases = setdiff(aliases, string.(propertynames(subparam_structs)))
        pairs = CP.get_parameter_values!(toml_dict, aliases)
        return param_struct{FT, typeof.(values(subparam_structs))...}(;
            pairs...,
            subparam_structs...,
        )
    end

    (; toml_dict, parsed_args) = config
    FT = CP.float_type(toml_dict)

    # EDMF parameters
    turbconv_params =
        create_parameter_struct(CAP.TurbulenceConvectionParameters)
    # Thermodynamics.jl parameters
    thermo_params =
        create_parameter_struct(TD.Parameters.ThermodynamicsParameters)
    # Radiation parameters
    rrtmgp_params = create_parameter_struct(RP.RRTMGPParameters)
    # Insolation.jl parameters
    insolation_params = create_parameter_struct(IP.InsolationParameters)
    # Water properties parameters (from CloudMicrophysics.jl)
    water_params = CM.Parameters.WaterProperties(FT, toml_dict)

    # Microphysics scheme parameters (from CloudMicrophysics.jl)
    # TODO - repeating the logic from solver/model_getters.jl...
    if parsed_args["override_τ_precip"]
        toml_dict["τ_precip"]["value"] =
            FT(CA.time_to_seconds(parsed_args["dt"]))
    end
    precip_model = parsed_args["precip_model"]
    microphys_params = if precip_model == nothing || precip_model == "nothing"
        nothing
    elseif precip_model == "0M"
        CM.Parameters.Parameters0M(FT, toml_dict)
    elseif precip_model == "1M"
        (;
            cl = CM.Parameters.CloudLiquid(FT, toml_dict),
            ci = CM.Parameters.CloudIce(FT, toml_dict),
            pr = CM.Parameters.Rain(FT, toml_dict),
            ps = CM.Parameters.Snow(FT, toml_dict),
            ce = CM.Parameters.CollisionEff(FT, toml_dict),
            tv = CM.Parameters.Blk1MVelType(FT, toml_dict),
            aps = CM.Parameters.AirProperties(FT, toml_dict),
        )
    else
        error("Invalid precip_model $(precip_model)")
    end

    # SurfaceFluxes.jl parameters
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
    ufp = UF.BusingerParams{FT}(; pairs...)
    surf_flux_params = create_parameter_struct(
        SF.Parameters.SurfaceFluxesParameters;
        subparam_structs = (; ufp, thermo_params),
    )

    # Create the big ClimaAtmos parameters struct
    return create_parameter_struct(
        CAP.ClimaAtmosParameters;
        subparam_structs = (;
            thermodynamics_params = thermo_params,
            rrtmgp_params,
            insolation_params,
            microphysics_params = microphys_params,
            water_params,
            surface_fluxes_params = surf_flux_params,
            turbconv_params,
        ),
    )
end
