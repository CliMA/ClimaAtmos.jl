"""
    init_solid_body_rotation(params, case = :baroclinic_wave)

    Hydrostatically balanced initial condition for 3D sphere benchmarking.
"""
function init_solid_body_rotation(::Type{FT}, params) where {FT}
    # physics parameters
    p_0::FT = CLIMAParameters.Planet.MSLP(params)
    cv_d::FT = CLIMAParameters.Planet.cv_d(params)
    R_d::FT = CLIMAParameters.Planet.R_d(params)
    T_tri::FT = CLIMAParameters.Planet.T_triple(params)
    g::FT = CLIMAParameters.Planet.grav(params)

    # initial condition specific parameters
    T_0::FT = 300.0
    H::FT = R_d * T_0 / g # scale height

    # auxiliary functions
    p(z) = p_0 * exp(-z / H)

    # density
    function ρ(local_geometry)
        @unpack z = local_geometry.coordinates
        return FT(p(z) / R_d / T_0)
    end

    # total energy density
    function ρe_tot(local_geometry)
        @unpack z = local_geometry.coordinates
        Φ(z) = g * z
        e_tot(z) = cv_d * (T_0 - T_tri) + Φ(z)
        return FT(ρ(local_geometry) * e_tot(z))
    end

    # horizontal velocity
    uh(local_geometry) = Geometry.Covariant12Vector(FT(0), FT(0))

    # vertical velocity
    w(local_geometry) = Geometry.Covariant3Vector(FT(0))

    return (ρ = ρ, ρe_tot = ρe_tot, uh = uh, w = w)
end
