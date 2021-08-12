# dry rising bubble
function dry_rising_bubble(x,z,parameters)
    r(p,x,z)       = sqrt((x - p.xc)^2 + (z - p.zc)^2)
    Δθ(p,x,y,z)    = (r(p,x,z) < p.rc) ? ( p.θₐ * (1.0 - r(p,x,z) / p.rc) ) : 0
    θ₀(p,x,y,z)    = 300.0 + Δθ(p,x,y,z)
    π_exn(p,x,y,z) = 1.0 - p.g / (p.cp_d * θ₀(p,x,y,z) ) * z  

    e_pot(p,x,y,z) = p.g * z
    e_int(p,x,y,z) = p.cv_d * (θ₀(p,x,y,z) * π_exn(p,x,y,z) - p.T_0 )
    e_kin(p,x,y,z) = 0.0

    ρ₀(p,x,y,z)    = p.pₒ / (p.R_d * θ₀(p,x,y,z)) * (π_exn(p,x,y,z))^(p.cv_d / p.R_d)
    ρu₀(p,x,y,z)   = @SVector [0.0, 0.0, 0.0]
    ρe₀(p,x,y,z)   = ρ₀(p,x,y,z) * (e_kin(p,x,y,z) + e_int(p,x,y,z) + e_pot(p,x,y,z))

    nothing
end