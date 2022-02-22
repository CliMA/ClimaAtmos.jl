"""
    init_2d_mountain_advection(params, thermovar = :ρθ) 
    Flow over a mountain, 
    Reference https://journals.ametsoc.org/view/journals/mwre/140/4/mwr-d-10-05073.1.xml, Section 5d
    Reference parameter values: 
      - 𝒩       = 0.01 s⁻¹     [Brunt-Väisälä frequency]
      - u̅       = 10 ms⁻¹      [Background flow speed]
      - h_c     = 400 m        [Mountain peak height]
      - a_c     = 10000 m      [Mountain shape parameter]
      - Tₛ      = 300 K        [Surface Temperature]
      - Lx      = [-14a_c, 14a_c] m [X domain] 
      - Lz      = [0, 21000] m [Z domain]
      - z_s     = 16000 m      [Sponge height]
      - ν₂      = 0 m²s⁻¹      [Kinematic viscosity]
      - Δt      = 0.006a_c/u̅ s [Timestep] 
      - 𝒻       = 1e-4 s⁻¹     [Coriolis parameter]
"""
function init_2d_mountain_advection(::Type{FT}, params; thermovar = :ρθ) where {FT}
    θ₀ = 300.0
    p_0::FT = CLIMAParameters.Planet.MSLP(params)
    cp_d::FT = CLIMAParameters.Planet.cp_d(params)
    cv_d::FT = CLIMAParameters.Planet.cv_d(params)
    R_d::FT = CLIMAParameters.Planet.R_d(params)
    g::FT = CLIMAParameters.Planet.grav(params)
    γ = cp_d / cv_d

    𝒩 = 0.01
    π_exner(local_geometry) = begin 
      @unpack z = local_geometry.coordinates
      return exp(-g * z / (cp_d * θ₀))
    end 
    θ(local_geometry) = begin
      @unpack z = local_geometry.coordinates
      return θ₀ * exp(𝒩 ^2 * z / g)
    end

    ρ(local_geometry) = p_0 / (R_d * θ(local_geometry)) * (π_exner(local_geometry))^(cp_d/R_d)
    ρθ(local_geometry)  = ρ(local_geometry) * θ
    ρuh(local_geometry) = ρ(local_geometry) * Geometry.UVector.(10.0)
    
    if thermovar == :ρθ
        return (ρ = ρ, ρθ = ρθ, ρuh = ρuh, ρw = ρw)
    else
        throw(ArgumentError("thermovar $thermovar unknown."))
    end
    # Currently only supports ρθ form.

end

"""
   warp_mountain(coord;)
   Function prescribing shape of bottom boundary.
"""
function warp_mountain(
    coord;
    h_c = 400,
    a_c = 10_000,
    x_c = 0,
)
    x = coord.x
    return h_c / (1 + (x - x_c)^2/a_c^2)
end
