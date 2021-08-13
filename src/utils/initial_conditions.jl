module InitialConditions

using UnPack

export init_cosine_bell_2d
export init_dry_rising_bubble_2d
export init_straka_density_current_2d
export init_cosine_bell_shallow_water

"""
    init_cosine_bell_2d(x, z, parameters)

    Cosine bell initial condition for 2D box benchmarking.
    Reference parameter values:
        - x_c = 500 m
        - z_c = 500 m
        - r_c = 250 m
        - u_0 = 10.0
        - θ_b = 300 K
        - θ_c = 0.5 K
        - p_0 = 1e5 Pa
        - cp_d = 1004 J kg⁻¹ K⁻¹
        - cv_d = 717.5 J kg⁻¹ K⁻¹
        - R_d = 287.0 J kg⁻¹ K⁻¹
        - g = 0.0 m s⁻²
"""
function init_cosine_bell_2d(x, z, parameters)
    UnPack.@unpack x_c, z_c, r_c, θ_b, θ_c, p_0, cp_d, cv_d, R_d, g = parameters
    @assert x_c = 500.0
    @assert z_c = 500.0
    @assert r_c = 250.0
    @assert u_0 = 10.0
    @assert θ_b = 300.0
    @assert θ_c = 0.5
    @assert p_0 = 1e5
    @assert cp_d = 1004.0
    @assert cv_d = 717.5
    @assert R_d = 287.0
    @assert g = 0.0

    # auxiliary quantities
    r = sqrt((x - x_c)^2 + (z - z_c)^2)
    θ_p = r < r_c ? 0.5 * θ_c * (1.0 + cospi(r / r_c)) : 0.0 # potential temperature perturbation
    θ = θ_b + θ_p # potential temperature
    p_exn = 1.0 - g * z / cp_d / θ # exner function
    T = θ * p_exn # temperature

    # density
    ρ = p_0 * p_exn^(cv_d / R_d) / R_d / θ
    
    # velocity
    u = [u_0, 0.0] # u⃗ = (u, w) 
    
    # total energy
    e_int = cv_d * T # may need to be adjusted depending on thermodynamics package
    e_kin = 0.0
    e_pot = g * z # may need to be adjusted depending on definition of z
    e_tot = e_int + e_kin + e_pot

    return (ρ = ρ, u = u, ρe = ρ * e_tot,)
end

"""
    init_dry_rising_bubble_2d(x, z, parameters)

    Dry rising bubble initial condition for 2D box benchmarking.
    Reference: https://journals.ametsoc.org/view/journals/mwre/140/4/mwr-d-10-05073.1.xml, Section 5a
    Reference parameter values:
        - x_c = 500 m
        - z_c = 350 m
        - r_c = 250 m
        - θ_b = 300 K
        - θ_c = 0.5 K
        - p_0 = 1e5 Pa
        - cp_d = 1004 J kg⁻¹ K⁻¹
        - cv_d = 717.5 J kg⁻¹ K⁻¹
        - R_d = 287.0 J kg⁻¹ K⁻¹
        - g = 9.80616 m s⁻²
"""
function init_dry_rising_bubble_2d(x, z, parameters)
    UnPack.@unpack x_c, z_c, r_c, θ_b, θ_c, p_0, cp_d, cv_d, R_d, g = parameters
    @assert x_c = 500.0
    @assert z_c = 350.0
    @assert r_c = 250.0
    @assert θ_b = 300.0
    @assert θ_c = 0.5
    @assert p_0 = 1e5
    @assert cp_d = 1004.0
    @assert cv_d = 717.5
    @assert R_d = 287.0
    @assert g = 9.80616

    # auxiliary quantities
    r = sqrt((x - x_c)^2 + (z - z_c)^2)
    θ_p = r < r_c ? 0.5 * θ_c * (1.0 + cospi(r / r_c)) : 0.0 # potential temperature perturbation
    θ = θ_b + θ_p # potential temperature
    p_exn = 1.0 - g * z / cp_d / θ # exner function
    T = θ * p_exn # temperature

    # density
    ρ = p_0 * p_exn^(cv_d / R_d) / R_d / θ
    
    # velocity
    u = [0.0, 0.0] # u⃗ = (u, w) 
    
    # total energy
    e_int = cv_d * T # may need to be adjusted depending on thermodynamics package
    e_kin = 0.0
    e_pot = g * z # may need to be adjusted depending on definition of z
    e_tot = e_int + e_kin + e_pot

    return (ρ = ρ, u = u, ρe = ρ * e_tot,)
end

"""
    init_straka_density_current_2d(x, z, parameters)

    Straka density current initial condition for 2D box benchmarking.
    Reference: https://journals.ametsoc.org/view/journals/mwre/140/4/mwr-d-10-05073.1.xml, Section 5b
    Reference parameter values:
        -
        - p_0 = 1e5 Pa
        - cp_d = 1004 J kg⁻¹ K⁻¹
        - cv_d = 717.5 J kg⁻¹ K⁻¹
        - R_d = 287.0 J kg⁻¹ K⁻¹
        - g = 9.80616 m s⁻²
"""
function init_straka_density_current_2d(x, z, parameters)
    UnPack.@unpack x_c, z_c, r_c, θ_b, θ_c, p_0, cp_d, cv_d, R_d, g = parameters
    @assert x_c = 0.0
    @assert y_c = 0.0
    @assert z_c = 3000.0
    @assert r_x = 4000.0
    @assert r_z = 2000.0
    @assert θ_b = 300.0
    @assert θ_c = -15.0
    @assert p_0 = 1e5
    @assert cp_d = 1004.0
    @assert cv_d = 717.5
    @assert R_d = 287.0
    @assert g = 9.80616

    ## Define bubble center and background potential temperature
    x_c = 0.0
    y_c = 0.0
    z_c = 3000.0
    r_x = 4000.0
    r_z = 2000.0
    θ_b = ref_state.virtual_temperature_profile.T_surface
    θ_c = -15.0

    # auxiliary quantities
    r = sqrt(((x - x_c)^2) / r_x^2 + ((z - z_c)^2) / r_z^2)
    θ_p = r <= 1.0 ? 0.5 * θ_c * (1 + cospi(r)) : 0.0 # potential temperature perturbation
    θ = θ_b + θ_p # potential temperature
    p_exn = 1.0 - g * z / cp_d / θ # exner function
    T = θ * p_exn # temperature

    # density
    ρ = p_0 * p_exn^(cv_d / R_d) / R_d / θ
    
    # velocity
    u = [0.0, 0.0] # u⃗ = (u, w) 
    
    # total energy
    e_int = cv_d * T # may need to be adjusted depending on thermodynamics package
    e_kin = 0.0
    e_pot = g * z # may need to be adjusted depending on definition of z
    e_tot = e_int + e_kin + e_pot

    return (ρ = ρ, u = u, ρe = ρ * e_tot,)
end

"""
    init_cosine_bell_shallow_water(λ, ϕ, parameters)

    Cosine bell initial condition for 2D shallow water benchmarking.
    Reference: https://climate.ucdavis.edu/pubs/UJL2010JCP.pdf, section 7.1
    Reference parameter values:
        - λ_c = 3π/2
        - ϕ_c = 0.0
        - h_0 = 1e3 m
        - α ∈ [0.0, 0.05, π/2 - 0.05, π/2]
        - a = 6.37122e6 m
        - g = 9.80616 m s⁻²
"""
function init_cosine_bell_shallow_water(λ, ϕ, parameters)
    UnPack.@unpack λ_c, ϕ_c, h_0, α, a, g = parameters
    @assert λ_c = 3π/2
    @assert ϕ_c = 0.0
    @assert h_0 = 1e3
    @assert α ∈ [0.0, 0.05, π/2 - 0.05, π/2]
    @assert a = 6.37122e6
    @assert g = 9.80616 

    # auxiliary quantities
    r = a * arccos(sin(ϕ_c) * sin(ϕ) + cos(ϕ_c) * cos(ϕ) * cos(λ - λ_c)) # great circle distance
    R = a / 3.0

    # height field
    h = r < R ? 0.5 * h_0 * (1.0 + cospi(r / R)) : 0.0
    
    # velocity
    u_0 = 2π * a / 12.0 / 86400.0
    u = u_0 .* [cos(ϕ) * cos(α) + cos(λ) * sin(ϕ) * cos(α), -sin(λ) * sin(α)] # u⃗ = (u, v) 
    
    return (h = h, u = u,)
end

end # Module