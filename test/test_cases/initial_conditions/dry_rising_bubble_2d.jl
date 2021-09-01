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

    return (ρ = ρ, u = u, ρe = ρ * e_tot)
end
