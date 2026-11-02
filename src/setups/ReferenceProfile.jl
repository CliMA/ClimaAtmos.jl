"""
    ReferenceProfile()

A resting atmosphere on the hydrostatic reference state (`T_r(p)` of
`air_temperature_reference`, `p_r(z)` from inverting `phi_r`):
`pm = p − p_ref ≡ 0` pointwise, so the reference-deviation amplitudes of the
flux-form DG interface dissipation vanish and the terrain C-property of the
discretization can be verified directly.
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
