function Models.default_ode_cache(
    model::Nonhydrostatic2DModel,
    cache::CacheEmpty,
    space_center,
    space_face,
)
    return nothing
end

function Models.default_ode_cache(
    model::Nonhydrostatic2DModel,
    cache::CacheBase,
    space_center,
    space_face,
)
    local_geometry_center = Fields.local_geometry_field(space_center)

    FT = eltype(model.domain)
    zero_instance = zero(FT) # somehow need this, otherwise eltype inference error

    p = map(coord -> zero_instance, local_geometry_center)
    Φ = map(coord -> zero_instance, local_geometry_center)

    return Fields.FieldVector(; p, Φ)
end

function Models.default_ode_cache(
    model::Nonhydrostatic2DModel,
    cache::CacheZeroMomentMicro,
    space_center,
    space_face,
)
    local_geometry_center = Fields.local_geometry_field(space_center)

    FT = eltype(model.domain)
    zero_instance = zero(FT) # somehow need this, otherwise eltype inference error

    p = map(coord -> zero_instance, local_geometry_center)
    Φ = map(coord -> zero_instance, local_geometry_center)
    K = map(coord -> zero_instance, local_geometry_center)
    e_int = map(coord -> zero_instance, local_geometry_center)
    microphysics_cache = map(
        coord -> (;
            S_q_tot = zero_instance,
            S_e_tot = zero_instance,
            q_liq = zero_instance,
            q_ice = zero_instance,
        ),
        local_geometry_center,
    )

    return Fields.FieldVector(; p, Φ, K, e_int, microphysics_cache)
end

function Models.default_ode_cache(
    model::Nonhydrostatic2DModel,
    cache::CacheOneMomentMicro,
    space_center,
    space_face,
)
    local_geometry_center = Fields.local_geometry_field(space_center)

    FT = eltype(model.domain)
    zero_instance = zero(FT) # somehow need this, otherwise eltype inference error

    p = map(coord -> zero_instance, local_geometry_center)
    Φ = map(coord -> zero_instance, local_geometry_center)
    K = map(coord -> zero_instance, local_geometry_center)
    e_int = map(coord -> zero_instance, local_geometry_center)
    microphysics_cache = map(
        coord -> (;
            S_q_tot = zero_instance,
            S_e_tot = zero_instance,
            S_q_rai = zero_instance,
            S_q_sno = zero_instance,
            q_liq = zero_instance,
            q_ice = zero_instance,
        ),
        local_geometry_center,
    )

    return Fields.FieldVector(; p, Φ, K, e_int, microphysics_cache)
end
