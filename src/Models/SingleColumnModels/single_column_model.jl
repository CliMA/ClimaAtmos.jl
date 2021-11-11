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
    zero_12vector(lg) = Geometry.UVVector(zero(FT), zero(FT))
    zero_3vector(lg) = Geometry.WVector(zero(FT))

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

        # gather boundary conditions
        bc_ρ = model.boundary_conditions.ρ
        bc_uv = model.boundary_conditions.uv
        bc_w = model.boundary_conditions.w
        bc_ρθ = model.boundary_conditions.ρθ

        # density
        flux_bottom = get_boundary_flux(model, bc_ρ.bottom, ρ, Ym, Ya)
        flux_top = get_boundary_flux(model, bc_ρ.top, ρ, Ym, Ya)
        If = Operators.InterpolateC2F()
        ∂f = Operators.GradientC2F()
        ∂c = Operators.DivergenceF2C(
            bottom = Operators.SetValue(flux_bottom),
            top = Operators.SetValue(flux_top),
        )
        @. dρ = -∂c(w * If(ρ))

        # potential temperature
        flux_bottom = get_boundary_flux(model, bc_ρθ.bottom, ρθ, Ym, Ya)
        flux_top = get_boundary_flux(model, bc_ρθ.top, ρθ, Ym, Ya)
        If = Operators.InterpolateC2F()
        ∂f = Operators.GradientC2F()
        ∂c = Operators.DivergenceF2C(
            bottom = Operators.SetValue(flux_bottom),
            top = Operators.SetValue(flux_top),
        )
        # TODO!: Undesirable casting to vector required
        @. dρθ = -∂c(w * If(ρθ)) + ρ * ∂c(Geometry.WVector(ν * ∂f(ρθ / ρ)))

        A = Operators.AdvectionC2C(
            bottom = Operators.SetValue(Geometry.UVVector(0.0, 0.0)),
            top = Operators.SetValue(Geometry.UVVector(0.0, 0.0)),
        )

        # uv
        flux_bottom = get_boundary_flux(model, bc_uv.bottom, uv, Ym, Ya)

        bcs_bottom = Operators.SetValue(flux_bottom)
        bcs_top = Operators.SetValue(uvg) # this needs abstraction
        ∂c = Operators.DivergenceF2C(bottom = bcs_bottom)
        ∂f = Operators.GradientC2F(top = bcs_top)
        duv .= (uv .- Ref(uvg)) .× Ref(Geometry.WVector(f))
        @. duv += ∂c(ν * ∂f(uv)) - A(w, uv)

        # w
        flux_bottom = get_boundary_flux(model, bc_w.bottom, w, Ym, Ya)
        flux_top = get_boundary_flux(model, bc_w.top, w, Ym, Ya)
        If = Operators.InterpolateC2F(
            bottom = Operators.Extrapolate(),
            top = Operators.Extrapolate(),
        )
        ∂f = Operators.GradientC2F()
        ∂c = Operators.GradientF2C()
        Af = Operators.AdvectionF2F()
        divf = Operators.DivergenceC2F()
        B = Operators.SetBoundaryOperator(
            bottom = Operators.SetValue(flux_bottom),
            top = Operators.SetValue(flux_top),
        )
        Φ(z) = grav * z
        Π(ρθ) = C_p * (R_d * ρθ / MSLP)^(R_m / C_v)
        zc = Fields.coordinate_field(axes(ρ)).z
        @. dw = B(
            Geometry.WVector(-(If(ρθ / ρ) * ∂f(Π(ρθ))) - ∂f(Φ(zc))) +
            divf(ν * ∂c(w)) - Af(w, w),
        )

        return dY
    end

    return rhs!
end
