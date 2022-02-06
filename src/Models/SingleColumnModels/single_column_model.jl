"""
    SingleColumnModel <: AbstractModel

A single column model. Required fields are `domain`, `boundary_conditions`, and
`parameters`.
"""
Base.@kwdef struct SingleColumnModel{D, BC, P} <:
                   Models.AbstractSingleColumnModel
    domain::D
    boundary_conditions::BC
    parameters::P
end

function Models.variable_names(::SingleColumnModel)
    base_vars = (:ρ, :uv, :w, :ρθ)
    thermo_vars = (:ρθ,)
    return (base = base_vars, thermodynamics = thermo_vars)
end

function Models.default_initial_conditions(model::SingleColumnModel)
    space_c, space_f = make_function_space(model.domain)
    local_geometry_c = CC.Fields.local_geometry_field(space_c)
    local_geometry_f = CC.Fields.local_geometry_field(space_f)

    # functions that make zeros for this model
    zero_val = zero(CC.Spaces.undertype(space_c))
    zero_scalar(lg) = zero_val
    zero_12vector(lg) = CC.Geometry.UVVector(zero_val, zero_val)
    zero_3vector(lg) = CC.Geometry.WVector(zero_val)

    # base components
    ρ = zero_scalar.(local_geometry_c)
    uv = zero_12vector.(local_geometry_c)
    w = zero_3vector.(local_geometry_f) # on faces

    # thermodynamics components
    ρθ = zero_scalar.(local_geometry_c)

    return CC.Fields.FieldVector(
        base = CC.Fields.FieldVector(ρ = ρ, uv = uv, w = w),
        thermodynamics = CC.Fields.FieldVector(ρθ = ρθ),
    )
end

function Models.make_ode_function(model::SingleColumnModel)
    FT = eltype(model.domain)

    rhs!(dY, Y, Ya, t) = begin
        # physics parameters
        C_p::FT = CLIMAParameters.Planet.cp_d(model.parameters)
        MSLP::FT = CLIMAParameters.Planet.MSLP(model.parameters)
        R_d::FT = CLIMAParameters.Planet.R_d(model.parameters)
        R_m::FT = R_d
        C_v::FT = CLIMAParameters.Planet.cv_d(model.parameters)
        grav::FT = CLIMAParameters.Planet.grav(model.parameters)

        # model specific parameters
        f = model.parameters.f
        uvg = model.parameters.uvg
        ν = model.parameters.ν

        # base components
        dYm = dY.base
        dρ = dYm.ρ
        duv = dYm.uv
        dw = dYm.w
        Ym = Y.base
        ρ = Ym.ρ
        uv = Ym.uv
        w = Ym.w

        # thermodynamics components
        dρθ = dY.thermodynamics.ρθ
        ρθ = Y.thermodynamics.ρθ
        wvec = CC.Geometry.WVector
        uvvec = CC.Geometry.UVector

        # gather boundary conditions
        bc_ρ = model.boundary_conditions.ρ
        bc_uv = model.boundary_conditions.uv
        bc_w = model.boundary_conditions.w
        bc_ρθ = model.boundary_conditions.ρθ

        # density
        flux_bottom = get_boundary_flux(model, bc_ρ.bottom, ρ, Y, Ya)
        flux_top = get_boundary_flux(model, bc_ρ.top, ρ, Y, Ya)
        If = CCO.InterpolateC2F()
        ∂f = CCO.GradientC2F()
        ∂c = CCO.DivergenceF2C(
            bottom = CCO.SetValue(flux_bottom),
            top = CCO.SetValue(flux_top),
        )
        @. dρ = -∂c(w * If(ρ))

        # potential temperature
        flux_bottom = get_boundary_flux(model, bc_ρθ.bottom, ρθ, Y, Ya)
        flux_top = get_boundary_flux(model, bc_ρθ.top, ρθ, Y, Ya)
        If = CCO.InterpolateC2F()
        ∂f = CCO.GradientC2F()
        ∂c = CCO.DivergenceF2C(
            bottom = CCO.SetValue(flux_bottom),
            top = CCO.SetValue(flux_top),
        )
        # TODO!: Undesirable casting to vector required
        @. dρθ = -∂c(w * If(ρθ)) + ρ * ∂c(wvec(ν * ∂f(ρθ / ρ)))

        FT = eltype(Y)
        A = CCO.AdvectionC2C(
            bottom = CCO.SetValue(uvvec(FT(0), FT(0))),
            top = CCO.SetValue(uvvec(FT(0), FT(0))),
        )

        # uv
        flux_bottom = get_boundary_flux(model, bc_uv.bottom, uv, Y, Ya)

        bcs_bottom = CCO.SetValue(flux_bottom)
        bcs_top = CCO.SetValue(uvg) # this needs abstraction
        ∂c = CCO.DivergenceF2C(bottom = bcs_bottom)
        ∂f = CCO.GradientC2F(top = bcs_top)
        duv .= (uv .- Ref(uvg)) .× Ref(wvec(f))
        @. duv += ∂c(ν * ∂f(uv)) - A(w, uv)

        # w
        flux_bottom = get_boundary_flux(model, bc_w.bottom, w, Y, Ya)
        flux_top = get_boundary_flux(model, bc_w.top, w, Y, Ya)
        If = CCO.InterpolateC2F(
            bottom = CCO.Extrapolate(),
            top = CCO.Extrapolate(),
        )
        ∂f = CCO.GradientC2F()
        ∂c = CCO.GradientF2C()
        Af = CCO.AdvectionF2F()
        divf = CCO.DivergenceC2F()
        B = CCO.SetBoundaryOperator(
            bottom = CCO.SetValue(flux_bottom),
            top = CCO.SetValue(flux_top),
        )
        Φ(z) = grav * z
        Π(ρθ) = C_p * (R_d * ρθ / MSLP)^(R_m / C_v)
        zc = CC.Fields.coordinate_field(axes(ρ)).z
        @. dw = B(
            wvec(-(If(ρθ / ρ) * ∂f(Π(ρθ))) - ∂f(Φ(zc))) + divf(ν * ∂c(w)) - Af(w, w),
        )

        return dY
    end

    return rhs!
end

function Models.get_velocities(Y, model::SingleColumnModel)
    w = Y.base.w
    return w
end
