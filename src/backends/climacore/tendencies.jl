const sdiv = Operators.Divergence()
const wdiv = Operators.WeakDivergence()
const sgrad = Operators.Gradient()
const wgrad = Operators.WeakGradient()
const scurl = Operators.Curl()
const wcurl = Operators.WeakCurl()

function create_rhs(::ClimaCoreBackend, model::BarotropicFluidModel, function_space)
    function rhs!(dydt, y, _, t)
        @unpack D₄, g = model.parameters

        # compute hyperviscosity first
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

        # add in pieces
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