import TurbulenceConvection as TC
import CLIMAParameters as CP
import CloudMicrophysics as CM
import SurfaceFluxes as SF
import SurfaceFluxes.UniversalFunctions as UF
import Thermodynamics as TD
import TurbulenceConvection.Parameters as TCP

#! format: off
function create_parameter_set(namelist, toml_dict_default::CP.AbstractTOMLDict, FTD = CP.float_type(toml_dict_default))
    FT = CP.float_type(toml_dict_default)
    _, out_dir = nc_fileinfo(namelist)
    override_file = joinpath(out_dir, "override_dict.toml")
    open(override_file, "w") do io
        println(io, "[mean_sea_level_pressure]")
        println(io, "alias = \"MSLP\"")
        println(io, "value = 100000.0")
        println(io, "type = \"float\"")
    end
    toml_dict = CP.create_toml_dict(FT; override_file, dict_type="alias")
    isfile(override_file) && rm(override_file; force=true)

    aliases = string.(fieldnames(TD.Parameters.ThermodynamicsParameters))
    param_pairs = CP.get_parameter_values!(toml_dict, aliases, "Thermodynamics")
    thermo_params = TD.Parameters.ThermodynamicsParameters{FTD}(; param_pairs...)
    # logfilepath = joinpath(@__DIR__, "logfilepath_$FT.toml")
    # CP.log_parameter_information(toml_dict, logfilepath)
    TP = typeof(thermo_params)

    aliases = string.(fieldnames(CM.Parameters.CloudMicrophysicsParameters))
    aliases = setdiff(aliases, ["thermo_params"])
    pairs = CP.get_parameter_values!(toml_dict, aliases, "CloudMicrophysics")
    microphys_params = CM.Parameters.CloudMicrophysicsParameters{FTD, TP}(;
        pairs...,
        thermo_params,
    )
    MP = typeof(microphys_params)

    aliases = ["Pr_0_Businger", "a_m_Businger", "a_h_Businger"]
    pairs = CP.get_parameter_values!(toml_dict, aliases, "UniversalFunctions")
    pairs = (; pairs...) # convert to NamedTuple
    pairs = (; Pr_0 = pairs.Pr_0_Businger, a_m = pairs.a_m_Businger, a_h = pairs.a_h_Businger)
    ufp = UF.BusingerParams{FTD}(; pairs...)
    UFP = typeof(ufp)

    pairs = CP.get_parameter_values!(toml_dict, ["von_karman_const"], "SurfaceFluxesParameters")
    surf_flux_params = SF.Parameters.SurfaceFluxesParameters{FTD, UFP, TP}(; pairs..., ufp, thermo_params)

    aliases = [
    "microph_scaling_dep_sub",
    "microph_scaling_melt",
    "microph_scaling",
    "Omega",
    "planet_radius"]
    pairs = CP.get_parameter_values!(toml_dict, aliases, "TurbulenceConvection")

    SFP = typeof(surf_flux_params)
    param_set = TCP.TurbulenceConvectionParameters{FTD, MP, SFP}(; pairs..., microphys_params, surf_flux_params)
    if !isbits(param_set)
        @warn "The parameter set SHOULD be isbits in order to be stack-allocated."
    end
    return param_set
end
#! format: on
