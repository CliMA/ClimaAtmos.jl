module COSPCloudSatReflectivity

import ClimaCore: Operators, Spaces

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
    nlevels == Spaces.nlevels(axes(attenuation_coefficient)) ||
        throw(ArgumentError("column accumulation requires all vertical levels"))
    input = Base.broadcasted(
        _initial_attenuation_state,
        attenuation_coefficient,
        height_km,
        top_height_km,
    )
    Operators.column_accumulate!(
        _accumulate_attenuation,
        path_attenuation,
        input;
        transform = _attenuation_path,
        reverse = true,
    )

    return nothing
end

@inline function _initial_attenuation_state(coefficient, z, z_top)
    FT = typeof(coefficient)
    return (;
        path = FT(2) * coefficient * (z_top - z),
        coefficient,
        z,
    )
end

@inline function _accumulate_attenuation(state, level)
    return (;
        path =
            state.path +
            (state.coefficient + level.coefficient) * (state.z - level.z),
        coefficient = level.coefficient,
        z = level.z,
    )
end

@inline _attenuation_path(state) = state.path

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
