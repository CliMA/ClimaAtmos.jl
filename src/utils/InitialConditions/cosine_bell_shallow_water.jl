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