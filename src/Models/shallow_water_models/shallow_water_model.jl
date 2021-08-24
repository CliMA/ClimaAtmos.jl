"""
    ShallowWaterModel <: AbstractModel
"""
Base.@kwdef struct ShallowWaterModel{
    DT <: AbstractHorizontalDomain,
    BCT,
    ICT,
    PT,
} <: AbstractModel
    domain::DT
    boundary_conditions::BCT
    initial_conditions::ICT
    parameters::PT
end

"""
    make_ode_function(model::ShallowWaterModel)
"""
function make_ode_function(model::ShallowWaterModel)
    function rhs!(dY, Y, _, t)
        @unpack D₄, g = model.parameters

        # function space
        function_space = axes(Y)

        # instantiate operators
        sdiv = Operators.Divergence()
        wdiv = Operators.WeakDivergence()
        sgrad = Operators.Gradient()
        wgrad = Operators.WeakGradient()
        scurl = Operators.Curl()
        wcurl = Operators.WeakCurl()

        # compute hyperviscosity first because it requires a direct stiffness summation
        @. dY.u =
            wgrad(sdiv(Y.u)) -
            Geometry.Cartesian12Vector(wcurl(Geometry.Covariant3Vector(scurl(Y.u))))
        @. dY.ρθ = wdiv(sgrad(Y.ρθ))
        Spaces.weighted_dss!(dY)
        @. dY.u =
            -D₄ * (
                wgrad(sdiv(dY.u)) -
                Geometry.Cartesian12Vector(wcurl(Geometry.Covariant3Vector(scurl(dY.u))))
            )
        @. dY.ρθ = -D₄ * wdiv(sgrad(dY.ρθ))

        # add in advection terms
        J = Fields.Field(function_space.local_geometry.J, function_space)
        @. begin
            dY.ρ = -wdiv(Y.ρ * Y.u)
            dY.u +=
                -sgrad(g * Y.ρ + norm(Y.u)^2 / 2) +
                Geometry.Cartesian12Vector(J * (Y.u × scurl(Y.u)))
            dY.ρθ += -wdiv(Y.ρθ * Y.u)
        end
        Spaces.weighted_dss!(dY)

        return dY
    end

    return rhs!
end
