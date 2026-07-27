module COSPCloudSatReflectivity

import ClimaCore: Fields, Spaces

export cloudsat_gas_path_attenuation!, cloudsat_reflectivity_subcolumn!

"""
    cloudsat_gas_path_attenuation!(
        gas_attenuation,
        g_vol,
        height_km,
        top_height_km,
        nlevels,
    )

Integrate the two-way gas path attenuation from the model top once per
CloudSat callback. `g_vol` is the one-way attenuation coefficient in dB/km.
"""
function cloudsat_gas_path_attenuation!(
    gas_attenuation,
    g_vol,
    height_km,
    top_height_km,
    nlevels = Spaces.nlevels(axes(g_vol)),
)
    return _two_way_path_attenuation_from_top!(
        gas_attenuation,
        g_vol,
        height_km,
        top_height_km,
        nlevels,
    )
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
        top_height_km,
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
    top_height_km,
    nlevels = Spaces.nlevels(axes(z_vol));
    R_UNDEF = nothing,
)
    FT = eltype(z_vol)
    missing_value = isnothing(R_UNDEF) ? FT(-1e30) : FT(R_UNDEF)
    _two_way_path_attenuation_from_top!(
        hydro_attenuation,
        kr_vol,
        height_km,
        top_height_km,
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

function _two_way_path_attenuation_from_top!(
    path_attenuation,
    attenuation_coefficient,
    height_km,
    top_height_km,
    nlevels,
)
    FT = eltype(path_attenuation)

    for ilev in nlevels:-1:1
        path_level = Fields.level(path_attenuation, ilev)
        coefficient_level = Fields.level(attenuation_coefficient, ilev)
        z_level = Fields.level(height_km, ilev)

        if ilev == nlevels
            @. path_level =
                FT(2) * coefficient_level * (top_height_km - z_level)
        else
            path_above = Fields.level(path_attenuation, ilev + 1)
            coefficient_above =
                Fields.level(attenuation_coefficient, ilev + 1)
            z_above = Fields.level(height_km, ilev + 1)
            @. path_level =
                path_above +
                (coefficient_above + coefficient_level) *
                (z_above - z_level)
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
