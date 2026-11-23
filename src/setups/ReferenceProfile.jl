"""
    ReferenceProfile()

A resting atmosphere on the hydrostatic reference state, `p = p_ref(z)`.
"""
struct ReferenceProfile end

function center_initial_condition(::ReferenceProfile, local_geometry, params)
    FT = eltype(params)
    thermo_params = CAP.thermodynamics_params(params)
    (; z) = local_geometry.coordinates
    Φ = CAP.grav(params) * z
    p = pref_from_phi(thermo_params, Φ)
    T = air_temperature_reference(thermo_params, p)
    return physical_state(; T, p)
end
