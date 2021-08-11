using ClimaCore.Spaces: SpectralElementSpace2D, CenterFiniteDifferenceSpace, FaceFiniteDifferenceSpace

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

        # S 4.4.1: potential temperature density
        # Mass conservation
        If = Operators.InterpolateC2F(;
            bottom = Operators.Extrapolate(),
            top = Operators.Extrapolate(),
        )
        ∂c = Operators.GradientF2C()
        @. dρ = -(∂c(w * If(ρ)))  # Eq. 4.11

        # advection
        u_1 = parent(u)[1]
        v_1 = parent(v)[1]
        u_wind = sqrt(u_1^2 + v_1^2)
        A = Operators.AdvectionC2C(
            bottom = Operators.SetValue(0.0),
            top = Operators.SetValue(0.0),
        )

        # u-momentum
        bcs_bottom = Operators.SetValue(Cd * u_wind * u_1)  # Eq. 4.16
        bcs_top = Operators.SetValue(ug)  # Eq. 4.18
        gradc2f = Operators.GradientC2F(top = bcs_top)
        gradf2c = Operators.GradientF2C(bottom = bcs_bottom)
        @. du = gradf2c(ν * gradc2f(u)) + f * (v - vg) - A(w, u) # Eq. 4.8

        # v-momentum
        bcs_bottom = Operators.SetValue(Cd * u_wind * v_1)  # Eq. 4.17
        bcs_top = Operators.SetValue(vg)  # Eq. 4.19
        gradc2f = Operators.GradientC2F(top = bcs_top)
        gradf2c = Operators.GradientF2C(bottom = bcs_bottom)
        @. dv = gradf2c(ν * gradc2f(v)) - f * (u - ug) - A(w, v) # Eq. 4.9

        # w-momentum
        B = Operators.SetBoundaryOperator(
            bottom = Operators.SetValue(0.0),
            top = Operators.SetValue(0.0),
        )
        # gradc2f = Operators.GradientC2F(;bottom =  Operators.SetValue(0.0), top = Operators.SetValue(0.0))
        gradc2f = Operators.GradientC2F()
        If = Operators.InterpolateC2F(;
            bottom = Operators.Extrapolate(),
            top = Operators.Extrapolate(),
        )
        gradf2c = Operators.GradientF2C()
        Π = C_p .* (R_d .* ρθ ./ MSLP).^(R_m ./ C_v)
        @. dw = B(
            -(If(ρθ / ρ) * gradc2f(Π)) - grav + gradc2f(ν * gradf2c(w)) -
            w * If(gradf2c(w)),
        )   # Eq. 4.10


        # potential temperature density conservation
        bcs_bottom = Operators.SetValue(0.0)  # Eq. 4.20
        bcs_top = Operators.SetValue(0.0)  # Eq. 4.21
        gradc2f = Operators.GradientC2F()
        gradf2c = Operators.GradientF2C(bottom = bcs_bottom, top = bcs_top)
        @. dρθ = -(∂c(w * If(ρθ))) + gradf2c(ν * gradc2f(ρθ / ρ))  # Eq. 4.12

        return dY
    end

    return rhs!
end