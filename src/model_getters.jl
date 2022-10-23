function moisture_model(parsed_args)
    moisture_name = parsed_args["moist"]
    @assert moisture_name in ("dry", "equil", "nonequil")
    return if moisture_name == "dry"
        DryModel()
    elseif moisture_name == "equil"
        EquilMoistModel()
    elseif moisture_name == "nonequil"
        NonEquilMoistModel()
    end
end

function energy_form(parsed_args)
    energy_name = parsed_args["energy_name"]
    @assert energy_name in ("rhoe", "rhoe_int", "rhotheta")
    vert_diff = parsed_args["vert_diff"]
    if vert_diff
        @assert energy_name == "rhoe"
    end
    return if energy_name == "rhoe"
        TotalEnergy()
    elseif energy_name == "rhoe_int"
        InternalEnergy()
    elseif energy_name == "rhotheta"
        PotentialTemperature()
    end
end

function compressibility_model(parsed_args)
    anelastic_dycore = parsed_args["anelastic_dycore"]
    @assert anelastic_dycore in (true, false)
    return if anelastic_dycore
        AnelasticFluid()
    else
        CompressibleFluid()
    end
end

function radiation_mode(parsed_args, ::Type{FT}) where {FT}
    radiation_name = parsed_args["rad"]
    @assert radiation_name in (
        nothing,
        "clearsky",
        "gray",
        "allsky",
        "allskywithclear",
        "DYCOMS_RF01",
        "TRMM_LBA",
    )
    return if radiation_name == "clearsky"
        RRTMGPI.ClearSkyRadiation()
    elseif radiation_name == "gray"
        RRTMGPI.GrayRadiation()
    elseif radiation_name == "allsky"
        RRTMGPI.AllSkyRadiation()
    elseif radiation_name == "allskywithclear"
        RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics()
    elseif radiation_name == "DYCOMS_RF01"
        RadiationDYCOMS_RF01{FT}()
    elseif radiation_name == "TRMM_LBA"
        RadiationTRMM_LBA(FT)
    else
        nothing
    end
end

function microphysics_model(parsed_args)
    microphysics_name = parsed_args["microphy"]
    @assert microphysics_name in (nothing, "0M")
    return if microphysics_name == nothing
        nothing
    elseif microphysics_name == "0M"
        Microphysics0Moment()
    end
end

function forcing_type(parsed_args)
    forcing = parsed_args["forcing"]
    @assert forcing in (nothing, "held_suarez")
    return if forcing == nothing
        nothing
    elseif forcing == "held_suarez"
        HeldSuarezForcing()
    end
end

function precipitation_model(parsed_args, namelist)
    namelist isa Nothing && return TC.NoPrecipitation()

    precip_name = TC.parse_namelist(
        namelist,
        "microphysics",
        "precipitation_model";
        default = "None",
        valid_options = ["None", "cutoff", "clima_1m"],
    )
    # TODO: use parsed_args
    # precip_name = parsed_args["precipitation_model"]
    # TODO: move to grid mean model
    return if precip_name == "None"
        TC.NoPrecipitation()
    elseif precip_name == "cutoff"
        TC.CutoffPrecipitation()
    elseif precip_name == "clima_1m"
        TC.Clima1M()
    else
        error("Invalid precip_name $(precip_name)")
    end
end


function turbconv_model(FT, moisture_model, precip_model, parsed_args, namelist)
    turbconv = parsed_args["turbconv"]
    @assert turbconv in (nothing, "edmf")
    return if turbconv == "edmf"
        TC.EDMFModel(FT, namelist, moisture_model, precip_model, parsed_args)
    else
        nothing
    end
end

function surface_scheme(FT, parsed_args)
    surface_scheme = parsed_args["surface_scheme"]
    @assert surface_scheme in (nothing, "bulk", "monin_obukhov")
    return if surface_scheme == "bulk"
        BulkSurfaceScheme()
    elseif surface_scheme == "monin_obukhov"
        MoninObukhovSurface()
    elseif surface_scheme == nothing
        surface_scheme
    end
end

"""
    ThermoDispatcher(model_spec)

A helper method for creating a thermodynamics dispatcher
from the model specification struct.
"""
function ThermoDispatcher(model_spec)
    (; energy_form, moisture_model, compressibility_model) = model_spec
    return ThermoDispatcher(;
        energy_form,
        moisture_model,
        compressibility_model,
    )
end
