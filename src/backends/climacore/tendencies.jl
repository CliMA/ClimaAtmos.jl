function create_rhs(::ClimaCoreBackend, model::BarotropicFluidModel, function_space::SpectralElementSpace2D)
    function rhs!(dydt, y, _, t)
        UnPack.@unpack D₄, g = model.parameters

        # operators
        sdiv = Operators.Divergence()
        wdiv = Operators.WeakDivergence()
        sgrad = Operators.Gradient()
        wgrad = Operators.WeakGradient()
        scurl = Operators.Curl()
        wcurl = Operators.WeakCurl()

        # compute hyperviscosity first because it requires a direct stiffness summation
        @. dydt.u =
            wgrad(sdiv(y.u)) -
            Cartesian12Vector(wcurl(Geometry.Covariant3Vector(scurl(y.u))))
        @. dydt.ρθ = wdiv(sgrad(y.ρθ))

        Spaces.weighted_dss!(dydt)
        @. dydt.u =
            -D₄ * (
                wgrad(sdiv(dydt.u)) -
                Cartesian12Vector(wcurl(Geometry.Covariant3Vector(scurl(dydt.u))))
            )
        @. dydt.ρθ = -D₄ * wdiv(sgrad(dydt.ρθ))

        # add in advection pieces
        J = Fields.Field(function_space.local_geometry.J, function_space)
        @. begin
            dydt.ρ = -wdiv(y.ρ * y.u)
            dydt.u +=
                -sgrad(g * y.ρ + norm(y.u)^2 / 2) +
                Cartesian12Vector(J * (y.u × scurl(y.u)))
            dydt.ρθ += -wdiv(y.ρθ * y.u)
        end
        Spaces.weighted_dss!(dydt)

        return dydt
    end

    return rhs!
end

# S. 4.4.1
function create_rhs(::ClimaCoreBackend, model::HydrostaticModel, function_space::Tuple{CenterFiniteDifferenceSpace,FaceFiniteDifferenceSpace})
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

        # auxiliary calculations
        center_space = axes(Yc)
        dz_top = center_space.face_local_geometry.WJ[end]
        u_top, v_top = parent(u)[end], parent(v)[end]
        u_btm, v_btm = parent(u)[1], parent(v)[1]
        u_wind = sqrt(u_btm^2 + v_btm^2)

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

        # u velocity (center
        flux_btm = Cd * u_wind * u_btm
        flux_top = ν * (v_top - vg) / dz_top / 2
        ∇c = Operators.GradientC2F() # Eq. 4.18
        ∇f = Operators.GradientF2C(bottom = Operators.SetValue(flux_btm), top = Operators.SetValue(flux_top)) # Eq. 4.16

        A = Operators.AdvectionC2C(bottom = Operators.SetValue(0.0), top = Operators.SetValue(0.0))
        @. du = ∇f(ν * ∇c(u)) + f * (v - vg) - A(w, u) # Eq. 4.8

        # v velocity (center
        flux_btm = Cd * u_wind * v_btm
        flux_top = ν * (u_top - ug)/ dz_top / 2
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