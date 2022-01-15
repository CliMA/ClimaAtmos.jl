function init_1d_ekman_column(::Type{FT}, params) where {FT}
    # physics parameters
    p_0::FT = CLIMAParameters.Planet.MSLP(params)
    grav::FT = CLIMAParameters.Planet.grav(params)
    R_d::FT = CLIMAParameters.Planet.R_d(params)
    cp_d::FT = CLIMAParameters.Planet.cp_d(params)

    # initial condition specific parameters
    T_surf = FT(300)
    T_min_ref = FT(230)
    u0 = FT(1)
    v0 = FT(0)
    w0 = FT(0)

    # density
    ρ(local_geometry) = begin
        @unpack z = local_geometry.coordinates

        Γ = grav / cp_d
        T = max(T_surf - Γ * z, T_min_ref)
        p = p_0 * (T / T_surf)^(grav / (R_d * Γ))
        if T == T_min_ref
            z_top = (T_surf - T_min_ref) / Γ
            H_min = R_d * T_min_ref / grav
            p *= exp(-(z - z_top) / H_min)
        end
        θ = T_surf # potential temperature

        return p / (R_d * θ * (p / p_0)^(R_d / cp_d))
    end

    # velocity
    uv(local_geometry) = Geometry.UVVector(u0, v0) # u, v components
    w(local_geometry) = Geometry.WVector(w0) # w component

    # potential temperature
    ρθ(local_geometry) = ρ(local_geometry) * T_surf

    return (ρ = ρ, uv = uv, w = w, ρθ = ρθ)
end
