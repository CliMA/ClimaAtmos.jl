"""
    init_dry_shallow_baroclinic_wave(FT, params, isbalanced = true)

    - isbalanced: false is the standard baroclinic set up as in the reference;
                  true is the same background flow without perturbation.
    Dry shallow atmosphere baroclinic wave initial condition for 3D sphere benchmarking.
    Reference: https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/qj.2241 
"""
function init_dry_shallow_baroclinic_wave(
    ::Type{FT},
    params;
    isbalanced = false,
) where {FT}
    # physics parameters
    R::FT = CLIMAParameters.Planet.planet_radius(params)
    Ω::FT = CLIMAParameters.Planet.Omega(params)
    p_0::FT = CLIMAParameters.Planet.MSLP(params)
    cv_d::FT = CLIMAParameters.Planet.cv_d(params)
    R_d::FT = CLIMAParameters.Planet.R_d(params)
    T_tri::FT = CLIMAParameters.Planet.T_triple(params)
    g::FT = CLIMAParameters.Planet.grav(params)

    # initial condition specific parameters
    k::FT = 3.0
    T_e::FT = 310.0 # temperature at the equator
    T_p::FT = 240.0 # temperature at the pole
    Γ::FT = 0.005
    b::FT = 2.0
    z_t::FT = 15.0e3
    λ_c::FT = 20.0
    ϕ_c::FT = 40.0
    V_p::FT = 1.0
    T_0::FT = 0.5 * (T_e + T_p)
    A::FT = 1 / Γ
    B::FT = (T_0 - T_p) / T_0 / T_p
    C::FT = 0.5 * (k + 2) * (T_e - T_p) / T_e / T_p
    H::FT = R_d * T_0 / g
    d_0::FT = R / 6

    # auxiliary functions for this initial condition
    τ_z_1(z) = exp(Γ * z / T_0)
    τ_z_2(z) = 1 - 2 * (z / b / H)^2
    τ_z_3(z) = exp(-(z / b / H)^2)
    τ_1(z) = 1 / T_0 * τ_z_1(z) + B * τ_z_2(z) * τ_z_3(z)
    τ_2(z) = C * τ_z_2(z) * τ_z_3(z)
    τ_int_1(z) = A * (τ_z_1(z) - 1) + B * z * τ_z_3(z)
    τ_int_2(z) = C * z * τ_z_3(z)
    F_z(z) = (1 - 3 * (z / z_t)^2 + 2 * (z / z_t)^3) * (z ≤ z_t)
    I_T(ϕ) = cosd(ϕ)^k - k / (k + 2) * (cosd(ϕ))^(k + 2)
    T(ϕ, z) = (τ_1(z) - τ_2(z) * I_T(ϕ))^(-1)
    p(ϕ, z) = p_0 * exp(-g / R_d * (τ_int_1(z) - τ_int_2(z) * I_T(ϕ)))
    r(λ, ϕ) =
        R * acos(sind(ϕ_c) * sind(ϕ) + cosd(ϕ_c) * cosd(ϕ) * cosd(λ - λ_c))
    U(ϕ, z) =
        g * k / R * τ_int_2(z) * T(ϕ, z) * (cosd(ϕ)^(k - 1) - cosd(ϕ)^(k + 1))
    u(ϕ, z) =
        -Ω * R * cosd(ϕ) + sqrt((Ω * R * cosd(ϕ))^2 + R * cosd(ϕ) * U(ϕ, z))
    v(ϕ, z) = FT(0)
    c3(λ, ϕ) = cos(π * r(λ, ϕ) / 2 / d_0)^3
    s1(λ, ϕ) = sin(π * r(λ, ϕ) / 2 / d_0)
    cond(λ, ϕ) = (0 < r(λ, ϕ) < d_0) * (r(λ, ϕ) != R * pi)
    δu(λ, ϕ, z) = begin
        if isbalanced
            return FT(0)
        else
            return -16 * V_p / 3 / sqrt(3) *
                   F_z(z) *
                   c3(λ, ϕ) *
                   s1(λ, ϕ) *
                   (-sind(ϕ_c) * cosd(ϕ) + cosd(ϕ_c) * sind(ϕ) * cosd(λ - λ_c)) / sin(r(λ, ϕ) / R) * cond(λ, ϕ)
        end
    end
    δv(λ, ϕ, z) = begin
        if isbalanced
            return FT(0)
        else
            return 16 * V_p / 3 / sqrt(3) *
                   F_z(z) *
                   c3(λ, ϕ) *
                   s1(λ, ϕ) *
                   cosd(ϕ_c) *
                   sind(λ - λ_c) / sin(r(λ, ϕ) / R) * cond(λ, ϕ)
        end
    end
    uu(λ, ϕ, z) = u(ϕ, z) + δu(λ, ϕ, z)
    uv(λ, ϕ, z) = v(ϕ, z) + δv(λ, ϕ, z)
    uw(λ, ϕ, z) = FT(0)

    # density
    function ρ(local_geometry)
        @unpack lat, long, z = local_geometry.coordinates
        return FT(p(lat, z) / R_d / T(lat, z))
    end

    # total energy density
    function ρe_tot(local_geometry)
        @unpack lat, long, z = local_geometry.coordinates
        Φ(z) = g * z
        e_tot(λ, ϕ, z) =
            cv_d * (T(ϕ, z) - T_tri) +
            Φ(z) +
            (uu(λ, ϕ, z)^2 + uv(λ, ϕ, z)^2 + uw(λ, ϕ, z)^2) / 2
        return FT(ρ(local_geometry) * e_tot(long, lat, z))
    end

    # horizontal velocity
    # TODO!: Doesn't seem to work with Float32
    function uh(local_geometry)
        coord = local_geometry.coordinates
        λ = coord.long
        ϕ = coord.lat
        z = coord.z
        return Geometry.transform(
            Geometry.Covariant12Axis(),
            Geometry.UVVector(FT(uu(λ, ϕ, z)), FT(uv(λ, ϕ, z))),
            local_geometry,
        )
    end

    # vertical velocity
    w(local_geometry) = Geometry.Covariant3Vector(FT(0))

    return (ρ = ρ, ρe_tot = ρe_tot, uh = uh, w = w)
end
