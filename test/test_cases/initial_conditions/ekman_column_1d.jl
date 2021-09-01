function init_ekman_column_1d_c(z, params)
    @unpack grav, C_p, MSLP, R_d, T_surf, T_min_ref, u0, v0 = params

    # auxiliary quantities
    Γ = grav / C_p
    T = max(T_surf - Γ * z, T_min_ref)
    p = MSLP * (T / T_surf)^(grav / (R_d * Γ))
    if T == T_min_ref
        z_top = (T_surf - T_min_ref) / Γ
        H_min = R_d * T_min_ref / grav
        p *= exp(-(z - z_top) / H_min)
    end

    θ = T_surf # potential temperature
    ρ = p / (R_d * θ * (p / MSLP)^(R_d / C_p)) # density
    uv = Geometry.Cartesian12Vector(u0, v0) # velocity

    return (ρ = ρ, uv = uv, ρθ = ρ * θ)
end

function init_ekman_column_1d_f(z, params)
    @unpack w0 = params

    return (; w = Geometry.Cartesian3Vector(w0))
end
