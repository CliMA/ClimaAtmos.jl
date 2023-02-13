import ClimaAtmos.TurbulenceConvection as TC
import CLIMAParameters as CP
import RRTMGP.Parameters as RP
import CloudMicrophysics as CM
import Insolation.Parameters as IP
import ClimaAtmos.Parameters as CAP
import SurfaceFluxes as SF
import SurfaceFluxes.UniversalFunctions as UF
import Thermodynamics as TD
import ClimaAtmos.TurbulenceConvection.Parameters as TCP
import ClimaCore

#=
ClimaCore.Operators.getidx(
    scalar::ConstRef,
    loc::ClimaCore.Operators.Location,
    idx,
    hidx,
) = scalar[]
struct ConstRef{T} <: AbstractArray{T, 0}
    val::T
end
Base.getindex(c::ConstRef) = c.val
Base.size(c::ConstRef) = ()
=#
# TODO: move to corresponding packages
# disable for now as causing problems
Base.broadcastable(ps::SF.Parameters.SurfaceFluxesParameters) = Ref(ps)
Base.broadcastable(ps::CM.Parameters.CloudMicrophysicsParameters) = Ref(ps)
Base.broadcastable(ps::TD.Parameters.ThermodynamicsParameters) = Ref(ps)


function override_climaatmos_defaults(
    defaults::NamedTuple,
    overrides::NamedTuple,
)
    intersect_keys = intersect(keys(defaults), keys(overrides))
    intersect_vals = getproperty.(Ref(overrides), intersect_keys)
    intersect_overrides = (; zip(intersect_keys, intersect_vals)...)
    return merge(defaults, intersect_overrides)
end

#=
import ClimaAtmos
include(joinpath(pkgdir(ClimaAtmos), "parameters", "create_parameters.jl"))
params = create_climaatmos_parameter_set(FT)
=#
function create_climaatmos_parameter_set(
    ::Type{FT},
    overrides::NamedTuple = NamedTuple(),
) where {FT}
    toml_dict = CP.create_toml_dict(FT; dict_type = "alias")
    create_climaatmos_parameter_set(toml_dict, overrides)
end

function create_climaatmos_parameter_set(
    toml_dict::CP.AbstractTOMLDict,
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

    tc_params = 
    if haskey(overrides, :case_name)
        aliases = string.(fieldnames(TCP.TurbulenceConvectionParameters))
        pairs = CP.get_parameter_values!(toml_dict, aliases, "EDMF")
        pairs = override_climaatmos_defaults((; pairs...), overrides)
        pairs = (;pairs..., 
            microphys_params,
            surf_flux_params,
        )
        tc_params = TCP.TurbulenceConvectionParameters(pairs)
    else
        nothing
    end

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
