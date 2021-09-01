"""
    ShallowWaterModel <: AbstractModel
"""
Base.@kwdef struct ShallowWaterModel{FT, BCT, ICT, PT} <: AbstractModel
    domain::AbstractHorizontalDomain{FT}
    boundary_conditions::BCT
    initial_conditions::ICT
    parameters::PT
end

function Models.make_initial_conditions(model::ShallowWaterModel{FT}) where {FT}
    space = make_function_space(model.domain)
    local_geometry = Fields.local_geometry_field(space)
    Y_init = model.initial_conditions.(local_geometry, Ref(model.parameters))

    return Y_init
end

function Models.make_ode_function(model::ShallowWaterModel{FT}) where {FT}
    function rhs!(dY, Y, _, t)
        @unpack D₄, g = model.parameters

        # function space
        space = axes(Y)

        sdiv = Operators.Divergence()
        wdiv = Operators.WeakDivergence()
        grad = Operators.Gradient()
        wgrad = Operators.WeakGradient()
        curl = Operators.Curl()
        wcurl = Operators.WeakCurl()

        # compute hyperviscosity first
        @. dY.u =
            wgrad(sdiv(Y.u)) -
            Geometry.Covariant12Vector(wcurl(Geometry.Covariant3Vector(curl(Y.u))))
        @. dY.ρθ = wdiv(grad(Y.ρθ))
        Spaces.weighted_dss!(dY)
        @. dY.u =
            -D₄ * (
                wgrad(sdiv(dY.u)) -
                Geometry.Covariant12Vector(wcurl(Geometry.Covariant3Vector(curl(dY.u))),)
            )
        @. dY.ρθ = -D₄ * wdiv(grad(dY.ρθ))

        # add in advection terms
        J = Fields.Field(space.local_geometry.J, space)
        @. begin
            dY.ρ = -wdiv(Y.ρ * Y.u)
            dY.u +=
                -grad(g * Y.ρ + norm(Y.u)^2 / 2) +
                Geometry.Covariant12Vector((J * (Y.u × curl(Y.u))))
            dY.ρθ += -wdiv(Y.ρθ * Y.u)
        end
        Spaces.weighted_dss!(dY)

        return dY
    end
end
