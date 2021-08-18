# S. 4.4.1
"""
    create_rhs
"""
function create_rhs(::ClimaCoreBackend, model::CompressibleFluidModel, function_space::Tuple{CenterFiniteDifferenceSpace,FaceFiniteDifferenceSpace})
    function rhs!(dY, Y, _, t)
        UnPack.@unpack Cd, f, ν, ug, vg, C_p, MSLP, R_d, R_m, C_v, grav = model.parameters
        (Yc, Yf) = Y.x
        (dYc, dYf) = dY.x
        UnPack.@unpack ρ, u, v, ρθ = Yc
        UnPack.@unpack w = Yf
        dρ = dYc.ρ
        du = dYc.u
        dv = dYc.v
        dρθ = dYc.ρθ
        dw = dYf.w

        # density (centers)
        flux_btm = 0.0
        flux_top = 0.0
        ∇c = Operators.GradientC2F()
        ∇f = Operators.GradientF2C(bottom = Operators.SetValue(flux_btm), top = Operators.SetValue(flux_top))

        If = Operators.InterpolateC2F(bottom = Operators.Extrapolate(), top = Operators.Extrapolate())
        @. dρ = ∇f( -w * If(ρ) ) # Eq. 4.11

        # potential temperature (centers)
        flux_btm = 0.0
        flux_top = 0.0
        ∇c = Operators.GradientC2F()
        ∇f = Operators.GradientF2C(bottom = Operators.SetValue(flux_btm), top = Operators.SetValue(flux_top)) # Eq. 4.20, 4.21

        @. dρθ = ∇f( -w * If(ρθ) + ν * ∇c(ρθ/ρ) ) # Eq. 4.12

        # u velocity (centers)
        center_space = axes(Yc)
        dz_top = center_space.face_local_geometry.WJ[end]
        u_top, u_btm = parent(u)[end], parent(u)[1]
        v_top, v_btm = parent(v)[end], parent(v)[1]
        u_wind = sqrt(u_btm^2 + v_btm^2)
        flux_btm = Cd * u_wind * u_btm
        flux_top = ν * (u_top - ug)/ dz_top / 2
        ∇c = Operators.GradientC2F() # Eq. 4.18
        ∇f = Operators.GradientF2C(bottom = Operators.SetValue(flux_btm), top = Operators.SetValue(flux_top)) # Eq. 4.16

        A = Operators.AdvectionC2C(bottom = Operators.SetValue(0.0), top = Operators.SetValue(0.0))
        @. du = ∇f(ν * ∇c(u)) + f * (v - vg) - A(w, u) # Eq. 4.8

        # v velocity (centers)
        center_space = axes(Yc)
        dz_top = center_space.face_local_geometry.WJ[end]
        u_top, u_btm = parent(u)[end], parent(u)[1]
        v_top, v_btm = parent(v)[end], parent(v)[1]
        u_wind = sqrt(u_btm^2 + v_btm^2)
        flux_btm = Cd * u_wind * v_btm
        flux_top = ν * (v_top - vg) / dz_top / 2
        ∇c = Operators.GradientC2F() # Eq. 4.18
        ∇f = Operators.GradientF2C(bottom = Operators.SetValue(flux_btm), top = Operators.SetValue(flux_top)) # Eq. 4.16
        
        A = Operators.AdvectionC2C(bottom = Operators.SetValue(0.0), top = Operators.SetValue(0.0))
        @. dv = ∇f(ν * ∇c(v)) - f * (u - ug) - A(w, v) # Eq. 4.9

        # w velocity (faces)
        flux_btm = 0.0
        flux_top = 0.0
        ∇c = Operators.GradientC2F()
        ∇f = Operators.GradientF2C(bottom = Operators.SetValue(flux_btm), top = Operators.SetValue(flux_top))

        If = Operators.InterpolateC2F(bottom = Operators.Extrapolate(), top = Operators.Extrapolate())
        B = Operators.SetBoundaryOperator(bottom = Operators.SetValue(0.0), top = Operators.SetValue(0.0))
        Π(ρθ) = C_p .* (R_d .* ρθ ./ MSLP).^(R_m ./ C_v)
        @. dw = B( -(If(ρθ / ρ) * ∇c(Π(ρθ))) - grav + ∇c(ν * ∇f(w)) - w * If(∇f(w))) # Eq. 4.10

        return dY
    end

    return rhs!
end