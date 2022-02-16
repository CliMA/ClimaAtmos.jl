"""
    init_2d_moist_bubble(params, thermovar = :ρθ)

    Rising bubble initial condition for 2D box benchmarking.
    Reference: https://github.com/CliMA/ClimateMachine.jl/blob/master/experiments/TestCase/risingbubble.jl, Section 5a
    Reference parameter values:
        - x_c = 5000 m
        - z_c = 2000 m
        - r_c = 2000 m
        - θ_b = 300 K
        - θ_c = 2.0 K
        - q_tot_c = 1e-3 kg/kg
        - p_0 = MSLP
        - g = 9.80616 m s⁻²
"""

function init_2d_moist_bubble(::Type{FT}, params; thermovar = :ρe_tot) where {FT}
    # initial condition specific parameters
    x_c::FT = 5000.0
    z_c::FT = 2000.0
    r_c::FT = 2000.0
    θ_b::FT = 300.0
    θ_c::FT = 2.0
    q_tot_c::FT = 1e-3

    # initial phase partition (specific humidity)
    q = Thermodynamics.PhasePartition(q_tot_c, FT(0.0), FT(0.0))

    # physics parameters
    p_0::FT = CLIMAParameters.Planet.MSLP(params)
    cp_m::FT = Thermodynamics.cp_m(params, q)
    cv_m::FT = Thermodynamics.cv_m(params, q)
    R_m::FT = Thermodynamics.gas_constant_air(params, q)
    g::FT = CLIMAParameters.Planet.grav(params)

    # auxiliary quantities
    # potential temperature perturbation
    θ_p(local_geometry) = begin
        @unpack x, z = local_geometry.coordinates
        r = sqrt((x - x_c)^2 + (z - z_c)^2)
        return r < r_c ? θ_c * (FT(1.0) - r / r_c) : FT(0)
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
    e_int(local_geometry) = Thermodynamics.internal_energy(params, T(local_geometry), q)

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
        return r < r_c ? q_tot_c * (FT(1.0) - r / r_c) : FT(0)
    end
    
    # total energy
    e_tot(local_geometry) =
        e_int(local_geometry) + e_kin(local_geometry) + e_pot(local_geometry)

    # prognostic quantities
    # density
    ρ(local_geometry) = Thermodynamics.air_density(params, T(local_geometry), p(local_geometry), q) 

    # potential temperature density
    ρθ(local_geometry) = ρ(local_geometry) * θ(local_geometry)

    # total energy density
    ρe_tot(local_geometry) = ρ(local_geometry) * e_tot(local_geometry)

    # specific humidity density
    ρq_tot(local_geometry) = ρ(local_geometry) * q_tot(local_geometry)

    # horizontal momentum vector
    ρuh(local_geometry) = Geometry.UVector(FT(0))

    # vertical momentum vector
    ρw(local_geometry) = Geometry.WVector(FT(0))

    if thermovar == :ρθ
        return (ρ = ρ, ρθ = ρθ, ρuh = ρuh, ρw = ρw, ρq_tot = ρq_tot)
    elseif thermovar == :ρe_tot
        return (ρ = ρ, ρe_tot = ρe_tot, ρuh = ρuh, ρw = ρw, ρq_tot = ρq_tot)
    else
        throw(ArgumentError("thermovar $thermovar unknown."))
    end
end
