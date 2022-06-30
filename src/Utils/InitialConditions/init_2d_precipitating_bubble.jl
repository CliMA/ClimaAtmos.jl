"""
    init_2d_dry_bubble(params, thermovar = :ρθ)

    Rising bubble initial condition for 2D box benchmarking.
        Reference: https://journals.ametsoc.org/view/journals/mwre/140/4/mwr-d-10-05073.1.xml, Section 5a    Reference parameter values:
        - x_c = 0 m
        - z_c = 350 m
        - r_c = 250 m
        - θ_b = 300 K
        - θ_c = 0.5 K
        - p_0 = 1e5 Pa
        - g = 9.80616 m s⁻²
"""
function init_2d_precipitating_bubble(
    ::Type{FT},
    params;
    thermovar = :ρθ,
) where {FT}
    # initial condition specific parameters
    x_c::FT = 0.0
    z_c::FT = 350.0
    r_c::FT = 250.0
    θ_b::FT = 300.0
    θ_c::FT = 0.5
    q_tot_c::FT = 1e-3

    thermo_params = CAP.thermodynamics_params(params)
    # initial phase partition (specific humidity)
    q = TD.PhasePartition(q_tot_c, FT(0.0), FT(0.0))
    # physics parameters
    p_0::FT = CAP.MSLP(params)
    cp_m::FT = TD.cp_m(thermo_params, q)
    cv_m::FT = TD.cv_m(thermo_params, q)
    R_m::FT = TD.gas_constant_air(thermo_params, q)
    g::FT = CAP.grav(params)

    # auxiliary quantities
    # potential temperature perturbation
    θ_p(local_geometry) = begin
        @unpack x, z = local_geometry.coordinates
        r = sqrt((x - x_c)^2 + (z - z_c)^2)
        return r < r_c ? FT(0.5) * θ_c * (FT(1) + cospi(r / r_c)) : FT(0)
    end

    # potential temperature
    θ(local_geometry) = θ_b + θ_p(local_geometry)

    # exner function
    π_exn(local_geometry) = begin
        @unpack z = local_geometry.coordinates
        return FT(1) - g * z / cp_m / θ(local_geometry)
    end

    # temperature
    T(local_geometry) = π_exn(local_geometry) * θ(local_geometry)

    # pressure
    p(local_geometry) = p_0 * π_exn(local_geometry)^(cp_m / R_m)

    # internal energy
    e_int(local_geometry) = cv_m * T(local_geometry)

    # kintetic energy
    e_kin(local_geometry) = FT(0)

    # potential energy
    e_pot(local_geometry) = begin
        @unpack z = local_geometry.coordinates
        return g * z
    end

    # moisture
    q_tot(local_geometry) = begin
        @unpack x, z = local_geometry.coordinates
        r = sqrt((x - x_c)^2 + (z - z_c)^2)
        return r < r_c ? FT(0.5) * q_tot_c * (FT(1) + cospi(r / r_c)) : FT(0)
    end

    # precipitation
    q_rai(local_geometry) = FT(0)
    q_sno(local_geometry) = FT(0)

    # total energy
    e_tot(local_geometry) =
        e_int(local_geometry) + e_kin(local_geometry) + e_pot(local_geometry)

    # prognostic quantities
    # density
    ρ(local_geometry) =
        TD.air_density(thermo_params, T(local_geometry), p(local_geometry), q)

    # potential temperature density
    ρθ(local_geometry) = ρ(local_geometry) * θ(local_geometry)

    # total energy density
    ρe_tot(local_geometry) = ρ(local_geometry) * e_tot(local_geometry)

    # specific humidity density
    ρq_tot(local_geometry) = ρ(local_geometry) * q_tot(local_geometry)

    # specific precipitation humidity density
    ρq_rai(local_geometry) = ρ(local_geometry) * q_rai(local_geometry)
    ρq_sno(local_geometry) = ρ(local_geometry) * q_sno(local_geometry)

    # horizontal momentum vector
    ρuh(local_geometry) = Geometry.UVector(FT(0))

    # vertical momentum vector
    ρw(local_geometry) = Geometry.WVector(FT(0))

    if thermovar == :ρθ
        return (
            ρ = ρ,
            ρθ = ρθ,
            ρuh = ρuh,
            ρw = ρw,
            ρq_tot = ρq_tot,
            ρq_rai = ρq_rai,
            ρq_sno = ρq_sno,
        )
    elseif thermovar == :ρe_tot
        return (
            ρ = ρ,
            ρe_tot = ρe_tot,
            ρuh = ρuh,
            ρw = ρw,
            ρq_tot = ρq_tot,
            ρq_rai = ρq_rai,
            ρq_sno = ρq_sno,
        )
    else
        throw(ArgumentError("thermovar $thermovar unknown."))
    end
end
