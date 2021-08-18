"""
    create_rhs
"""
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