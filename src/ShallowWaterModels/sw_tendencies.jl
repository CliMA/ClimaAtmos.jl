abstract type AbstractEquation end
abstract type AbstractTendency end

struct Height <: AbstractEquation end
struct Velocity <: AbstractEquation end
struct Tracer <: AbstractEquation end

struct FluxFormAdvection <: AbstractTendency end
struct CurlFormAdvection <: AbstractTendency end
struct PressureGradient{FT} <: AbstractTendency
    g::FT
end

function (::CurlFormAdvection)(dY, Y, Ya, t, ::Velocity)
    @unpack u = Y.swm
    grad = Operators.Gradient()
    curl = Operators.Curl()
    cov12 = Geometry.Covariant12Vector

    # TODO!: Should not need to use Jacobian determinant here -> ClimaCore
    space = axes(u)
    J = Fields.Field(space.local_geometry.J, space)
    @. dY.swm.u += -grad(norm(u)^2 / 2) + cov12((J * (u × curl(u))))
end

function (tend::PressureGradient)(dY, Y, Ya, t, ::Velocity)
    @unpack h = Y.swm
    grad = Operators.Gradient()

    @. dY.swm.u += -grad(tend.g * h)
end

# function (::FluxFormAdvection)(dY, Y, _, _, ::Height)
#     @unpack h, u = Y.swm
#     wdiv = Operators.WeakDivergence()
#     dss! = Spaces.weighted_dss!

#     @. dY.swm.h += -wdiv(h * u)
#     dss!(dY.swm.h)
# end

# function (::FluxFormAdvection)(dY, Y, _, _, ::Tracer)
#     @unpack c, u = Y.swm
#     wdiv = Operators.WeakDivergence()
#     dss! = Spaces.weighted_dss!

#     @. dY.swm.c += -wdiv(c * u)
#     dss!(dY.swm.c)
# end
