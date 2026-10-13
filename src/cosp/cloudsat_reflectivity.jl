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
        DBZe,
        z_vol,
        kr_vol,
        gas_attenuation,
        height_km,
        top_height_km,
        nlevels;
        R_UNDEF,
    )

Compute reflectivity for one streamed subcolumn. Hydrometeor path attenuation
is accumulated from the model top and transformed directly into `DBZe`.
"""
function cloudsat_reflectivity_subcolumn!(
    DBZe,
    z_vol,
    kr_vol,
    gas_attenuation,
    height_km,
    top_height_km,
    nlevels = Spaces.nlevels(axes(z_vol));
    R_UNDEF = nothing,
)
    FT = eltype(z_vol)
    missing_value = isnothing(R_UNDEF) ? FT(-1e30) : FT(R_UNDEF)
    nlevels == Spaces.nlevels(axes(kr_vol)) ||
        throw(ArgumentError("column accumulation requires all vertical levels"))
    input = Base.broadcasted(
        _initial_reflectivity_state,
        z_vol,
        kr_vol,
        gas_attenuation,
        height_km,
        top_height_km,
    )
    Operators.column_accumulate!(
        _accumulate_reflectivity,
        DBZe,
        input;
        transform = _DBZeTransform(missing_value),
        reverse = true,
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
        path = _next_attenuation_path(state, level),
        coefficient = level.coefficient,
        z = level.z,
    )
end

@inline _attenuation_path(state) = state.path

@inline _next_attenuation_path(state, level) =
    state.path +
    (state.coefficient + level.coefficient) * (state.z - level.z)

@inline function _initial_reflectivity_state(
    z_vol,
    coefficient,
    gas_path,
    z,
    z_top,
)
    FT = typeof(coefficient)
    return (;
        path = FT(2) * coefficient * (z_top - z),
        coefficient,
        z,
        z_vol,
        gas_path,
    )
end

@inline function _accumulate_reflectivity(state, level)
    return (;
        path = _next_attenuation_path(state, level),
        coefficient = level.coefficient,
        z = level.z,
        z_vol = level.z_vol,
        gas_path = level.gas_path,
    )
end

struct _DBZeTransform{FT}
    missing_value::FT
end

@inline function (transform::_DBZeTransform)(state)
    ze_non =
        _nonattenuated_reflectivity(state.z_vol, transform.missing_value)
    return ifelse(
        state.z_vol > zero(state.z_vol),
        ze_non - state.path - state.gas_path,
        transform.missing_value,
    )
end

@inline function _nonattenuated_reflectivity(z_vol, missing_value)
    FT = typeof(z_vol)
    return z_vol > zero(FT) ? FT(10) * log10(z_vol) : missing_value
end

end
