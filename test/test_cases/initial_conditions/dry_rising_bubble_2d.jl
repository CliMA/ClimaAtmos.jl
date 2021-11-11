"""
    init_dry_rising_bubble_2d(params)

    Dry rising bubble initial condition for 2D box benchmarking.
    Reference: https://journals.ametsoc.org/view/journals/mwre/140/4/mwr-d-10-05073.1.xml, Section 5a
    Reference parameter values:
        - x_c = 0 m
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
function init_dry_rising_bubble_2d(
    FT::DataType = Float64,
    params = nothing,
    thermovar = :ρθ,
)
    if params isa Nothing
        x_c = FT(0.0)
        z_c = FT(350.0)
        r_c = FT(250.0)
        θ_b = FT(300.0)
        θ_c = FT(0.5)
        p_0 = FT(1e5)
        cp_d = FT(1004.0)
        cv_d = FT(717.5)
        R_d = FT(287.0)
        g = FT(9.80616)
    else
        @unpack x_c, z_c, r_c, θ_b, θ_c, p_0, cp_d, cv_d, R_d, g = params
    end

    # auxiliary quantities
    # potential temperature perturbation
    θ_p(local_geometry) = begin
        @unpack x, z = local_geometry.coordinates
        r = sqrt((x - x_c)^2 + (z - z_c)^2)
        return r < r_c ? FT(0.5) * θ_c * (1 + cospi(r / r_c)) : FT(0)
    end

    # potential temperature    
    θ(local_geometry) = θ_b + θ_p(local_geometry)

    # exner function
    π_exn(local_geometry) = begin
        @unpack z = local_geometry.coordinates
        return 1 - g * z / cp_d / θ(local_geometry)
    end

    # temperature
    T(local_geometry) = π_exn(local_geometry) * θ(local_geometry)

    # pressure
    p(local_geometry) = p_0 * π_exn(local_geometry)^(cp_d / R_d)

    # internal energy
    e_int(local_geometry) = cv_d * T(local_geometry)

    # kintetic energy
    e_kin(local_geometry) = 0.0

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
    ρuh(local_geometry) = Geometry.UVector(0.0)

    # vertical momentum vector
    ρw(local_geometry) = Geometry.WVector(0.0)

    if thermovar == :ρθ
        return (ρ = ρ, ρθ = ρθ, ρuh = ρuh, ρw = ρw)
    elseif thermovar == :ρe_tot
        return (ρ = ρ, ρe_tot = ρe_tot, ρuh = ρuh, ρw = ρw)
    else
        throw(ArgumentError("thermovar $thermovar unknown."))
    end
end
