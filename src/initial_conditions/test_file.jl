"""
    BuoyantBubble(; perturb = false)

An `InitialCondition` with an isothermal background profile, with a 
buoyant bubble, and with an optional perturbation to the temperature.
"""
Base.@kwdef struct BuoyantBubble <: InitialCondition
    perturb::Bool = false
end

function (initial_condition::BuoyantBubble)(params)
    (; perturb) = initial_condition
    function local_state(local_geometry)
        FT = eltype(params)
        grav = CAP.grav(params)
        thermo_params = CAP.thermodynamics_params(params)
        ndims = length(propertynames(local_geometry.coordinates))
        (; x, z) = local_geometry.coordinates
        x_c = FT(10e3)
        x_r = FT(2e3)
        z_c = FT(2e3)
        z_r = FT(2e3)
        r_c = FT(1)
        θ_b = FT(300)
        θ_c = FT(2)
        cp_d = CAP.cp_d(params)
        cv_d = CAP.cv_d(params)
        p_0 = CAP.p_ref_theta(params)
        R_d = CAP.R_d(params)
        T_0 = CAP.T_0(params)

        # auxiliary quantities
        r² = FT(0)
        r² += ((x - x_c) / x_r)^2 + ((z - z_c) / z_r)^2
        if ndims == 3
            (; y) = local_geometry.coordinates
            y_c = FT(10e3)
            y_r = FT(2e3)
            r² += ((y - y_c) / y_r)^2
        end
        θ_p =
            sqrt(r²) < r_c ? FT(1 / 2) * θ_c * (FT(1) + cospi(sqrt(r²) / r_c)) :
            FT(0) # potential temperature perturbation
        θ = θ_b + θ_p # potential temperature
        π_exn = FT(1) - grav * z / cp_d / θ # exner function
        T = π_exn * θ # temperature
        p = p_0 * π_exn^(cp_d / R_d) # pressure
        ρ = p / R_d / T # density
        q = FT(0.0196) # constant moisture

        return LocalState(;
            params,
            geometry = local_geometry,
            thermo_state = TD.PhaseEquil_pTq(thermo_params, p, T, q),
        )
    end
    return local_state
end
