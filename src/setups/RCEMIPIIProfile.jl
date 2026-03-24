"""
    RCEMIPIIProfile(temperature, humidity)

An initial condition following the sounding for RCEMIPII as described by
Wing et al. (2018) (https://doi.org/10.5194/gmd-11-793-2018).

Three convenience constructors are provided:
- `RCEMIPIIProfile_295()` — SST = 295 K
- `RCEMIPIIProfile_300()` — SST = 300 K
- `RCEMIPIIProfile_305()` — SST = 305 K

Note: this should be used for RCE_small and NOT RCE_large — RCE_large must
be initialized with the final state of RCE_small.
"""
struct RCEMIPIIProfile{FT}
    temperature::FT
    humidity::FT
end

RCEMIPIIProfile_295() = RCEMIPIIProfile(295.0, 12e-3)
RCEMIPIIProfile_300() = RCEMIPIIProfile(300.0, 18.65e-3)
RCEMIPIIProfile_305() = RCEMIPIIProfile(305.0, 24e-3)

function center_initial_condition(setup::RCEMIPIIProfile, local_geometry, params)
    FT = eltype(params)
    R_d = CAP.R_d(params)
    grav = CAP.grav(params)

    T_0 = FT(setup.temperature)
    q_0 = FT(setup.humidity)

    q_t = FT(10^(-14))   # kg/kg
    z_q1 = FT(4000)      # m
    z_q2 = FT(7500)      # m
    z_t = FT(15000)      # m
    Γ = FT(0.0067)       # K/m
    p_0 = FT(101480)     # Pa

    T_v0 = T_0 * (1 + FT(0.608) * q_0)
    T_vt = T_v0 - Γ * z_t

    p_t = p_0 * (T_vt / T_v0)^(grav / (R_d * Γ))

    (; z) = local_geometry.coordinates
    if z ≤ z_t
        q = q_0 * exp(-z / z_q1) * exp(-(z / z_q2)^2)
        T_v = T_v0 - Γ * z
        p = p_0 * ((T_v0 - Γ * z) / T_v0)^(grav / (R_d * Γ))
    else
        q = q_t
        T_v = T_vt
        p = p_t * exp(-grav * (z - z_t) / (R_d * T_vt))
    end
    T = T_v / (1 + FT(0.608) * q)

    return physical_state(; T, p, q_tot = q)
end

insolation_model(::RCEMIPIIProfile) = RCEMIPIIInsolation()
surface_temperature_model(::RCEMIPIIProfile) = RCEMIPIISST()
