"""
    SingleColumnModel <: AbstractModel
"""
Base.@kwdef struct SingleColumnModel{FT, BCT, ICT, PT} <: AbstractModel
    domain::AbstractVerticalDomain{FT}
    boundary_conditions::BCT
    initial_conditions::ICT
    parameters::PT
end

function Models.make_initial_conditions(model::SingleColumnModel{FT}) where {FT}
    center_space, face_space = make_function_space(model.domain)
    z_centers = Fields.coordinate_field(center_space)
    z_faces = Fields.coordinate_field(face_space)
    Y_c = model.initial_conditions.centers.(z_centers, Ref(model.parameters))
    Y_f = model.initial_conditions.faces.(z_faces, Ref(model.parameters))

    Y_init = ArrayPartition(Y_c, Y_f)

    return Y_init
end

function Models.make_ode_function(model::SingleColumnModel{FT}) where {FT}
    rhs!(dY, Y, _, t) = begin
        @unpack Cd, f, ν, uvg, C_p, MSLP, R_d, R_m, C_v, grav = model.parameters

        # unpack prognostic state
        (Yc, Yf) = Y.x
        @unpack ρ, uv, ρθ = Yc
        @unpack w = Yf

        # unpack prognostic state tendency
        (dYc, dYf) = dY.x
        dρ = dYc.ρ
        duv = dYc.uv
        dw = dYf.w
        dρθ = dYc.ρθ

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
        @. dρθ = -∂c(w * If(ρθ) + Geometry.CartesianVector(ν * ∂f(ρθ / ρ)))

        uv_1 = Operators.getidx(uv, Operators.Interior(), 1)
        u_wind = LinearAlgebra.norm(uv_1)
        A = Operators.AdvectionC2C(
            bottom = Operators.SetValue(Geometry.Cartesian12Vector(0.0, 0.0)),
            top = Operators.SetValue(Geometry.Cartesian12Vector(0.0, 0.0)),
        )

        # uv
        bcs_bottom = Operators.SetValue(Geometry.Cartesian3Vector(
            Cd * u_wind * uv_1,
        ))
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
            Geometry.CartesianVector(
                -(If(Yc.ρθ / Yc.ρ) * ∂f(Π(Yc.ρθ))) - ∂f(Φ(zc)),
            ) + divf(ν * ∂c(w)) - Af(w, w),
        )

        return dY
    end

    return rhs!
end
