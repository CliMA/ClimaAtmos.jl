function init_ekman_column_1d(params)
    @unpack grav, C_p, MSLP, R_d, T_surf, T_min_ref, u0, v0, w0 = params

    # density
    ρ(local_geometry) = begin
        @unpack z = local_geometry.coordinates

        Γ = grav / C_p
        T = max(T_surf - Γ * z, T_min_ref)
        p = MSLP * (T / T_surf)^(grav / (R_d * Γ))
        if T == T_min_ref
            z_top = (T_surf - T_min_ref) / Γ
            H_min = R_d * T_min_ref / grav
            p *= exp(-(z - z_top) / H_min)
        end
        θ = T_surf # potential temperature

        return p / (R_d * θ * (p / MSLP)^(R_d / C_p))
    end

    # velocity
    uv(local_geometry) = Geometry.Cartesian12Vector(u0, v0) # u, v components
    w(local_geometry) = Geometry.Cartesian3Vector(w0) # w component

    # potential temperature
    ρθ(local_geometry) = ρ(local_geometry) * T_surf

    return (ρ = ρ, uv = uv, w = w, ρθ = ρθ)
end
