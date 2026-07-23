module COSPCloudSatReflectivity

import ClimaCore: Fields, Spaces

export cloudsat_gas_path_attenuation!, cloudsat_reflectivity_subcolumn!

"""
    cloudsat_gas_path_attenuation!(gas_attenuation, g_vol, height_km, nlevels)

Integrate the gas path attenuation from the model top once per CloudSat
callback. This is the gas half of the original combined recurrence.
"""
function cloudsat_gas_path_attenuation!(
    gas_attenuation,
    g_vol,
    height_km,
    nlevels = Spaces.nlevels(axes(g_vol)),
)
    FT = eltype(gas_attenuation)
    if nlevels == 1
        @. gas_attenuation = zero(FT)
        return nothing
    end

    for ilev in nlevels:-1:1
        gas_level = Fields.level(gas_attenuation, ilev)
        g_level = Fields.level(g_vol, ilev)
        z_level = Fields.level(height_km, ilev)

        if ilev == nlevels
            z_below = Fields.level(height_km, ilev - 1)
            @. gas_level = FT(0.5) * g_level * (z_level - z_below)
        else
            gas_above = Fields.level(gas_attenuation, ilev + 1)
            g_above = Fields.level(g_vol, ilev + 1)
            z_above = Fields.level(height_km, ilev + 1)
            @. gas_level =
                gas_above + FT(0.5) * (g_above + g_level) * (z_above - z_level)
        end
    end

    return nothing
end

"""
    cloudsat_reflectivity_subcolumn!(
        Ze_non,
        DBZe,
        z_vol,
        kr_vol,
        hydro_attenuation,
        gas_attenuation,
        height_km,
        nlevels;
        R_UNDEF,
    )

Compute hydrometeor path attenuation and reflectivity for one streamed
subcolumn. All intermediate fields are overwritten on every call.
"""
function cloudsat_reflectivity_subcolumn!(
    Ze_non,
    DBZe,
    z_vol,
    kr_vol,
    hydro_attenuation,
    gas_attenuation,
    height_km,
    nlevels = Spaces.nlevels(axes(z_vol));
    R_UNDEF = nothing,
)
    FT = eltype(z_vol)
    missing_value = isnothing(R_UNDEF) ? FT(-1e30) : FT(R_UNDEF)
    _hydrometeor_path_attenuation_from_top!(
        hydro_attenuation,
        kr_vol,
        height_km,
        nlevels,
    )
    _reflectivity_from_path_attenuation!(
        Ze_non,
        DBZe,
        z_vol,
        hydro_attenuation,
        gas_attenuation,
        missing_value,
    )
    return nothing
end

function _hydrometeor_path_attenuation_from_top!(
    hydro_attenuation,
    kr_vol,
    height_km,
    nlevels,
)
    FT = eltype(hydro_attenuation)
    if nlevels == 1
        @. hydro_attenuation = zero(FT)
        return nothing
    end

    for ilev in nlevels:-1:1
        hydro_level = Fields.level(hydro_attenuation, ilev)
        kr_level = Fields.level(kr_vol, ilev)
        z_level = Fields.level(height_km, ilev)

        if ilev == nlevels
            z_below = Fields.level(height_km, ilev - 1)
            @. hydro_level = kr_level * (z_level - z_below)
        else
            hydro_above = Fields.level(hydro_attenuation, ilev + 1)
            kr_above = Fields.level(kr_vol, ilev + 1)
            z_above = Fields.level(height_km, ilev + 1)
            @. hydro_level =
                hydro_above + (kr_above + kr_level) * (z_above - z_level)
        end
    end

    return nothing
end

function _reflectivity_from_path_attenuation!(
    Ze_non,
    DBZe,
    z_vol,
    hydro_attenuation,
    gas_attenuation,
    missing_value,
)
    FT = eltype(z_vol)
    @. Ze_non = _nonattenuated_reflectivity(z_vol, missing_value)
    @. DBZe =
        ifelse(
            z_vol > zero(FT),
            Ze_non - hydro_attenuation - gas_attenuation,
            missing_value,
        )
    return nothing
end

@inline function _nonattenuated_reflectivity(z_vol, missing_value)
    FT = typeof(z_vol)
    return z_vol > zero(FT) ? FT(10) * log10(z_vol) : missing_value
end

end
