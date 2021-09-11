"""
    SingleColumnModel <: AbstractModel

Construct a single column model on `domain`
with `parameters`.
`boundary_conditions` contains field boundary conditions.
"""
Base.@kwdef struct SingleColumnModel{FT, BCT, PT} <: AbstractModel
    domain::AbstractVerticalDomain{FT}
    boundary_conditions::BCT
    parameters::PT
    name::Symbol = :scm
    varnames::Tuple = (:ρ, :uv, :w, :ρθ)
end

function Models.default_initial_conditions(
    model::SingleColumnModel{FT},
) where {FT}
    space_c, space_f = make_function_space(model.domain)
    local_geometry_c = Fields.local_geometry_field(space_c)
    local_geometry_f = Fields.local_geometry_field(space_f)

    # functions that make zeros for this model
    zero_scalar(lg) = zero(FT)
    zero_12vector(lg) = Geometry.Cartesian12Vector(zero(FT), zero(FT))
    zero_3vector(lg) = Geometry.Cartesian3Vector(zero(FT))

    ρ = zero_scalar.(local_geometry_c)
    uv = zero_12vector.(local_geometry_c)
    w = zero_3vector.(local_geometry_f) # faces
    ρθ = zero_scalar.(local_geometry_c)

    return Fields.FieldVector(
        scm = Fields.FieldVector(ρ = ρ, uv = uv, w = w, ρθ = ρθ),
    )
end

function Models.make_ode_function(model::SingleColumnModel{FT}) where {FT}
    rhs!(dY, Y, Ya, t) = begin
        @unpack Cd, f, ν, uvg, C_p, MSLP, R_d, R_m, C_v, grav = model.parameters

        # unpack tendencies and state
        dYm = dY.scm
        dρ = dYm.ρ
        duv = dYm.uv
        dw = dYm.w
        dρθ = dYm.ρθ
        Ym = Y.scm
        ρ = Ym.ρ
        uv = Ym.uv
        w = Ym.w
        ρθ = Ym.ρθ

        # density
        If = Operators.InterpolateC2F()
        ∂f = Operators.GradientC2F()
        ∂c = Operators.DivergenceF2C(
            bottom = Operators.SetValue(Geometry.Cartesian3Vector(zero(FT))),
            top = Operators.SetValue(Geometry.Cartesian3Vector(zero(FT))),
        )
        @. dρ = -∂c(w * If(ρ))

        # potential temperature
        If = Operators.InterpolateC2F()
        ∂f = Operators.GradientC2F()
        ∂c = Operators.DivergenceF2C(
            bottom = Operators.SetValue(Geometry.Cartesian3Vector(zero(FT))),
            top = Operators.SetValue(Geometry.Cartesian3Vector(zero(FT))),
        )
        # TODO!: Undesirable casting to vector required
        @. dρθ =
            -∂c(w * If(ρθ)) + ρ * ∂c(Geometry.CartesianVector(ν * ∂f(ρθ / ρ)))

        uv_1 = Operators.getidx(uv, Operators.Interior(), 1)
        u_wind = LinearAlgebra.norm(uv_1)

        A = Operators.AdvectionC2C(
            bottom = Operators.SetValue(Geometry.Cartesian12Vector(0.0, 0.0)),
            top = Operators.SetValue(Geometry.Cartesian12Vector(0.0, 0.0)),
        )

        # uv
        bcs_bottom = Operators.SetValue(
            Geometry.Cartesian3Vector(Cd * u_wind) ⊗ uv_1,
        )
        bcs_top = Operators.SetValue(uvg)
        ∂c = Operators.DivergenceF2C(bottom = bcs_bottom)
        ∂f = Operators.GradientC2F(top = bcs_top)
        duv .= (uv .- Ref(uvg)) .× Ref(Geometry.Cartesian3Vector(f))
        @. duv += ∂c(ν * ∂f(uv)) - A(w, uv)

        # w
        If = Operators.InterpolateC2F(
            bottom = Operators.Extrapolate(),
            top = Operators.Extrapolate(),
        )
        ∂f = Operators.GradientC2F()
        ∂c = Operators.GradientF2C()
        Af = Operators.AdvectionF2F()
        divf = Operators.DivergenceC2F()
        B = Operators.SetBoundaryOperator(
            bottom = Operators.SetValue(Geometry.Cartesian3Vector(zero(FT))),
            top = Operators.SetValue(Geometry.Cartesian3Vector(zero(FT))),
        )
        Φ(z) = grav * z
        Π(ρθ) = C_p * (R_d * ρθ / MSLP)^(R_m / C_v)
        zc = Fields.coordinate_field(axes(ρ))
        @. dw = B(
            Geometry.CartesianVector(-(If(ρθ / ρ) * ∂f(Π(ρθ))) - ∂f(Φ(zc))) + divf(ν * ∂c(w)) - Af(w, w),
        )

        return dY
    end

    return rhs!
end
