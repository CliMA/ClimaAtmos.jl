module COSPCloudSatReflectivity

import ClimaCore: Fields, Spaces

export cloudsat_reflectivity!,
    cloudsat_gas_path_attenuation!, cloudsat_reflectivity_subcolumn!

"""
    cloudsat_reflectivity!(
        Ze_non_cloudsat,
        DBZe_cloudsat,
        z_vol_cloudsat,
        kr_vol_cloudsat,
        g_vol_cloudsat,
    )

Compute CloudSat nonattenuated and attenuated radar reflectivity from
optical quantities.

This tuple API is retained as a compatibility and test-reference wrapper. The
production callback reuses cached work fields through the single-subcolumn API.

This follows the COSPv2 `quickbeam_subcolumn` convention for a spaceborne radar,
where attenuation is accumulated from the top model level toward the surface, using
center-level heights in kilometers.
"""
function cloudsat_reflectivity!(
    Ze_non_cloudsat::NTuple{N},
    DBZe_cloudsat::NTuple{N},
    z_vol_cloudsat::NTuple{N},
    kr_vol_cloudsat::NTuple{N},
    g_vol_cloudsat;
    R_UNDEF = nothing,
) where {N}
    N > 0 ||
        throw(ArgumentError("CloudSat reflectivity needs at least one subcolumn"))

    reference = z_vol_cloudsat[1]
    _check_field_tuple_axes(Ze_non_cloudsat, reference, "Ze_non_cloudsat")
    _check_field_tuple_axes(DBZe_cloudsat, reference, "DBZe_cloudsat")
    _check_field_tuple_axes(z_vol_cloudsat, reference, "z_vol_cloudsat")
    _check_field_tuple_axes(kr_vol_cloudsat, reference, "kr_vol_cloudsat")
    axes(g_vol_cloudsat) == axes(reference) ||
        throw(DimensionMismatch("g_vol_cloudsat must have matching axes"))

    FT = eltype(reference)
    missing_value = isnothing(R_UNDEF) ? FT(-1e30) : FT(R_UNDEF)
    height_km = Fields.coordinate_field(axes(reference)).z ./ FT(1000)
    hydro_attenuation = similar(reference)
    gas_attenuation = similar(reference)
    nlevels = Spaces.nlevels(axes(reference))

    cloudsat_gas_path_attenuation!(
        gas_attenuation,
        g_vol_cloudsat,
        height_km,
        nlevels,
    )

    for isubcolumn in 1:N
        cloudsat_reflectivity_subcolumn!(
            Ze_non_cloudsat[isubcolumn],
            DBZe_cloudsat[isubcolumn],
            z_vol_cloudsat[isubcolumn],
            kr_vol_cloudsat[isubcolumn],
            hydro_attenuation,
            gas_attenuation,
            height_km,
            nlevels;
            R_UNDEF = missing_value,
        )
    end

    return nothing
end

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

function _check_field_tuple_axes(fields, reference, name)
    for field in fields
        axes(field) == axes(reference) ||
            throw(DimensionMismatch("$name fields must have matching axes"))
    end
end

end
