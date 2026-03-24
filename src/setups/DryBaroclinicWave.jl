"""
    DryBaroclinicWave(; perturb = true, deep_atmosphere = false)

A setup with a dry baroclinic wave initial condition, following the test case
described in Ullrich et al. (2014).

When `perturb` is true, a localized perturbation is applied to the horizontal
velocity field to trigger baroclinic instability.
"""
struct DryBaroclinicWave
    perturb::Bool
    deep_atmosphere::Bool
end

function DryBaroclinicWave(; perturb::Bool = true, deep_atmosphere::Bool = false)
    return DryBaroclinicWave(perturb, deep_atmosphere)
end

function shallow_atmos_barowave_values(z, ϕ, λ, params, perturb)
    FT = eltype(params)
    R_d = CAP.R_d(params)
    MSLP = CAP.MSLP(params)
    grav = CAP.grav(params)
    Ω = CAP.Omega(params)
    R = CAP.planet_radius(params)

    # Constants from paper
    k = 3
    T_e = FT(310) # temperature at the equator
    T_p = FT(240) # temperature at the pole
    T_0 = FT(0.5) * (T_e + T_p)
    Γ = FT(0.005)
    A = 1 / Γ
    B = (T_0 - T_p) / T_0 / T_p
    C = FT(0.5) * (k + 2) * (T_e - T_p) / T_e / T_p
    b = 2
    H = R_d * T_0 / grav
    z_t = FT(15e3)
    λ_c = FT(20)
    ϕ_c = FT(40)
    d_0 = R / 6
    V_p = FT(1)

    # Virtual temperature and pressure
    τ_z_1 = exp(Γ * z / T_0)
    τ_z_2 = 1 - 2 * (z / b / H)^2
    τ_z_3 = exp(-(z / b / H)^2)
    τ_1 = 1 / T_0 * τ_z_1 + B * τ_z_2 * τ_z_3
    τ_2 = C * τ_z_2 * τ_z_3
    τ_int_1 = A * (τ_z_1 - 1) + B * z * τ_z_3
    τ_int_2 = C * z * τ_z_3
    I_T = cosd(ϕ)^k - k * (cosd(ϕ))^(k + 2) / (k + 2)
    T = (τ_1 - τ_2 * I_T)^(-1)
    p = MSLP * exp(-grav / R_d * (τ_int_1 - τ_int_2 * I_T))

    # Horizontal velocity
    U = grav * k / R * τ_int_2 * T * (cosd(ϕ)^(k - 1) - cosd(ϕ)^(k + 1))
    u = -Ω * R * cosd(ϕ) + sqrt((Ω * R * cosd(ϕ))^2 + R * cosd(ϕ) * U)
    v = FT(0)
    if perturb
        F_z = (1 - 3 * (z / z_t)^2 + 2 * (z / z_t)^3) * (z ≤ z_t)
        r = R * acos(sind(ϕ_c) * sind(ϕ) + cosd(ϕ_c) * cosd(ϕ) * cosd(λ - λ_c))
        c3 = cos(π * r / 2 / d_0)^3
        s1 = sin(π * r / 2 / d_0)
        cond = (0 < r < d_0) * (r != R * pi)
        u +=
            -16 * V_p / 3 / sqrt(FT(3)) *
            F_z *
            c3 *
            s1 *
            (-sind(ϕ_c) * cosd(ϕ) + cosd(ϕ_c) * sind(ϕ) * cosd(λ - λ_c)) /
            sin(r / R) * cond
        v +=
            16 * V_p / 3 / sqrt(FT(3)) *
            F_z *
            c3 *
            s1 *
            cosd(ϕ_c) *
            sind(λ - λ_c) / sin(r / R) * cond
    end

    return (; T, p, u, v)
end

function deep_atmos_barowave_values(z, ϕ, λ, params, perturb)
    FT = eltype(params)
    R_d = CAP.R_d(params)
    MSLP = CAP.MSLP(params)
    grav = CAP.grav(params)
    Ω = CAP.Omega(params)
    R = CAP.planet_radius(params)

    # Constants from paper (See Table 1. in Ullrich et al (2014))
    k = 3         # Power for temperature field
    T_e = FT(310) # Surface temperature at the equator
    T_p = FT(240) # Surface temperature at the pole
    T_0 = FT(0.5) * (T_e + T_p)
    Γ = FT(0.005) # Lapse rate
    A = 1 / Γ  # (Eq 16)
    B = (T_0 - T_p) / T_0 / T_p # (Eq 17)
    C = FT(0.5) * (k + 2) * (T_e - T_p) / T_e / T_p # (Eq 17)
    b = 2 # half-width parameter
    H = R_d * T_0 / grav
    z_t = FT(15e3) # Top of perturbation domain
    λ_c = FT(20) # Geographical location (λ dim) of perturbation center
    ϕ_c = FT(40) # Geographical location (ϕ dim) of perturbation center
    d_0 = R / 6
    V_p = FT(1)

    # Virtual temperature and pressure
    τ̃₁ =
        A * Γ / T_0 * exp(Γ * z / T_0) +
        B * (1 - 2 * (z / b / H)^2) * exp(-(z / b / H)^2)# (Eq 14)
    τ̃₂ = C * (1 - 2 * (z / b / H)^2) * exp(-(z / b / H)^2) # (Eq 15)
    ∫τ̃₁ = (A * (exp(Γ * z / T_0) - 1)) + B * z * exp(-(z / b / H)^2) # (Eq A1)
    ∫τ̃₂ = C * z * exp(-(z / b / H)^2) # (Eq A2)
    I_T =
        ((z + R) / R * cosd(ϕ))^k -
        (k / (k + 2)) * ((z + R) / R * cosd(ϕ))^(k + 2)
    T = FT((R / (z + R))^2 * (τ̃₁ - τ̃₂ * I_T)^(-1)) # (Eq A3)
    p = FT(MSLP * exp(-grav / R_d * (∫τ̃₁ - ∫τ̃₂ * I_T))) # (Eq A6)
    # Horizontal velocity
    U =
        grav / R *
        k *
        T *
        ∫τ̃₂ *
        (((z + R) * cosd(ϕ) / R)^(k - 1) - ((R + z) * cosd(ϕ) / R)^(k + 1)) # wind-proxy (Eq A4)
    u = FT(
        -Ω * (R + z) * cosd(ϕ) +
        sqrt((Ω * (R + z) * cosd(ϕ))^2 + (R + z) * cosd(ϕ) * U),
    )
    v = FT(0)
    if perturb
        F_z = (1 - 3 * (z / z_t)^2 + 2 * (z / z_t)^3) * (z ≤ z_t)
        r = R * acos(sind(ϕ_c) * sind(ϕ) + cosd(ϕ_c) * cosd(ϕ) * cosd(λ - λ_c))
        c3 = cos(π * r / 2 / d_0)^3
        s1 = sin(π * r / 2 / d_0)
        cond = (0 < r < d_0) * (r != R * pi)
        u +=
            -16 * V_p / 3 / sqrt(FT(3)) *
            F_z *
            c3 *
            s1 *
            (-sind(ϕ_c) * cosd(ϕ) + cosd(ϕ_c) * sind(ϕ) * cosd(λ - λ_c)) /
            sin(r / R) * cond
        v +=
            16 * V_p / 3 / sqrt(FT(3)) *
            F_z *
            c3 *
            s1 *
            cosd(ϕ_c) *
            sind(λ - λ_c) / sin(r / R) * cond
    end
    return (; T, p, u, v)
end

function center_initial_condition(
    setup::DryBaroclinicWave,
    local_geometry,
    params,
)
    (; z, lat, long) = local_geometry.coordinates
    values =
        setup.deep_atmosphere ? deep_atmos_barowave_values :
        shallow_atmos_barowave_values
    return physical_state(; values(z,
        lat,
        long,
        params,
        setup.perturb)...)
end
