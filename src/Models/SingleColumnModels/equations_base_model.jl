@inline function rhs_base_model!(dY, Y, Ya, t, _...)
    error("not implemented for this model configuration.")
end

function rhs_base_model!(
    dY,
    Y,
    Ya,
    t,
    # Π, # just for now, incremental tests -> climacore ekman, p,
    p,
    Φ,
    ::AdvectiveForm,
    params,
    flux_correction,
    FT,
)
    # base components
    dρ = dY.base.ρ # scalar on centers
    duh = dY.base.uh # Covariant12Vector on centers
    dw = dY.base.w # Covariant3Vector on faces
    ρ = Y.base.ρ
    uh = Y.base.uh
    w = Y.base.w

    # experiment specif parameters
    # TODO: given lat in params, f = @. Geometry.Contravariant3Vector(Geometry.WVector(2*Ω*params.lat,)) 
    f = params.f
    uh_g = params.uh_g

    # # gather boundary conditions:
    # TODO: need to unpack outside rhs function and only pass in base bc
    # bc_ρ = model.boundary_conditions.ρ
    # bc_uh = model.boundary_conditions.uh
    # bc_w = model.boundary_conditions.w

    # density
    # bc as hard coded for now; will later on make it work with the bc module
    # flux_bottom = get_boundary_flux(model, bc_ρ.bottom, ρ, Y, Ya)
    # flux_top = get_boundary_flux(model, bc_ρ.top, ρ, Y, Ya)
    flux_bottom = Geometry.WVector(FT(0))
    flux_top = Geometry.WVector(FT(0))
    interp_c2f = Operators.InterpolateC2F()
    vector_vdiv_f2c = Operators.DivergenceF2C(
        top = Operators.SetValue(flux_top),
        bottom = Operators.SetValue(flux_bottom),
    )
    @. dρ = -vector_vdiv_f2c(interp_c2f(ρ) * w)

    # horizontal velocity
    vadv_c2c = Operators.AdvectionC2C(
        bottom = Operators.SetValue(Geometry.UVVector(FT(0), FT(0))),
        top = Operators.SetValue(Geometry.UVVector(FT(0), FT(0))),
    )

    duh .= (uh .- Ref(uh_g)) .× Ref(Geometry.WVector(f))
    @. duh -= vadv_c2c(w, uh)

    # vertical velocity
    # flux_bottom = get_boundary_flux(model, bc_w.bottom, w, Y, Ya)
    # flux_top = get_boundary_flux(model, bc_w.top, w, Y, Ya)
    flux_bottom = Geometry.WVector(FT(0))
    flux_top = Geometry.WVector(FT(0))

    wbc = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(flux_bottom),
        top = Operators.SetValue(flux_top),
    )
    interp_c2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    scalar_vgrad_c2f = Operators.GradientC2F()
    vadv_f2f = Operators.AdvectionF2F()
    @. dw = wbc(
        Geometry.WVector(
            -(scalar_vgrad_c2f(p) / interp_c2f(ρ)) - scalar_vgrad_c2f(Φ),
        ) - vadv_f2f(w, w),
    )

end

function rhs_base_model!(
    dY,
    Y,
    Ya,
    t,
    # Π, # just for now, incremental tests -> climacore ekman, p,
    p,
    Φ,
    ::AnelasticAdvectiveForm,
    params,
    flux_correction,
    FT,
)
    # base components
    dρ = dY.base.ρ
    duh = dY.base.uh
    uh = Y.base.uh

    # experiment specif parameters
    # TODO: given lat in params, f = @. Geometry.Contravariant3Vector(Geometry.WVector(2*Ω*params.lat,)) 
    f = params.f
    uh_g = params.uh_g

    # density
    # anelastic
    dρ .= FT(0)

    # horizontal velocity
    duh .= (uh .- Ref(uh_g)) .× Ref(Geometry.WVector(f))

end
