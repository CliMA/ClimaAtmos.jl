"""
    init_3d_rising_bubble(params)

    Rising bubble initial condition for 3D box benchmarking.
    Reference: https://journals.ametsoc.org/view/journals/mwre/140/4/mwr-d-10-05073.1.xml, Section 5a
    Reference parameter values:
        - x_c = 0 m
        - y_c = 0 m
        - z_c = 350 m
        - r_c = 250 m
        - θ_b = 300 K
        - θ_c = 0.5 K
        - p_0 = 1e5 Pa
        - cp_d = 1004 J kg⁻¹ K⁻¹
        - cv_d = 717.5 J kg⁻¹ K⁻¹
        - R_d = 287.0 J kg⁻¹ K⁻¹
        - g = 9.80616 m s⁻²

    # TODO!: Replace expression with Thermodynamics.jl expressions.
"""
function init_3d_rising_bubble(::Type{FT}, params, thermovar = :ρθ) where {FT}
    # physics parameters
    p_0::FT = CLIMAParameters.Planet.MSLP(params)
    cp_d::FT = CLIMAParameters.Planet.cp_d(params)
    cv_d::FT = CLIMAParameters.Planet.cv_d(params)
    R_d::FT = CLIMAParameters.Planet.R_d(params)
    g::FT = CLIMAParameters.Planet.grav(params)

    # initial condition specific parameters
    x_c::FT = 0
    y_c::FT = 0
    z_c::FT = 350
    r_c::FT = 250
    θ_b::FT = 300
    θ_c::FT = 0.5

    # auxiliary quantities
    # potential temperature perturbation
    θ_p(local_geometry) = begin
        @unpack x, y, z = local_geometry.coordinates
        r = sqrt((x - x_c)^2 + (y - y_c)^2 + (z - z_c)^2)
        return r < r_c ? FT(0.5) * θ_c * (FT(1) + cospi(r / r_c)) : FT(0)
    end

    # potential temperature    
    θ(local_geometry) = θ_b + θ_p(local_geometry)

    # exner function
    π_exn(local_geometry) = begin
        @unpack z = local_geometry.coordinates
        return FT(1) - g * z / cp_d / θ(local_geometry)
    end

    # temperature
    T(local_geometry) = π_exn(local_geometry) * θ(local_geometry)

    # pressure
    p(local_geometry) = p_0 * π_exn(local_geometry)^(cp_d / R_d)

    # internal energy
    e_int(local_geometry) = cv_d * T(local_geometry)

    # kintetic energy
    e_kin(local_geometry) = FT(0)

    # potential energy
    e_pot(local_geometry) = begin
        @unpack z = local_geometry.coordinates
        return g * z
    end

    # total energy
    e_tot(local_geometry) =
        e_int(local_geometry) + e_kin(local_geometry) + e_pot(local_geometry)

    # prognostic quantities
    # density
    ρ(local_geometry) = p(local_geometry) / R_d / T(local_geometry)

    # potential temperature density
    ρθ(local_geometry) = ρ(local_geometry) * θ(local_geometry)

    # total energy density
    ρe_tot(local_geometry) = ρ(local_geometry) * e_tot(local_geometry)

    # horizontal momentum vector
    uh(local_geometry) = Geometry.Covariant12Vector(0.0, 0.0)

    # vertical momentum vector
    w(local_geometry) = Geometry.Covariant3Vector(0.0)

    if thermovar == :ρθ
        return (ρ = ρ, ρθ = ρθ, uh = uh, w = w)
    elseif thermovar == :ρe_tot
        return (ρ = ρ, ρe_tot = ρe_tot, uh = uh, w = w)
    else
        throw(ArgumentError("thermovar $thermovar unknown."))
    end
end
