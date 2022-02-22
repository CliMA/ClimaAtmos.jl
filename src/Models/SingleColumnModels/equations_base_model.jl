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
    ρθ = Y.thermodynamics.ρθ

    # experiment specif parameters
    # TODO: given lat in params, f = @. Geometry.Contravariant3Vector(Geometry.WVector(2*Ω*params.lat,)) 
    ν = params.ν
    Cd = params.Cd
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
    # bc for uh is very tricky, we want to support only flux boundary conditions in ClimaAtmos
    # but will all for user-customized function to provide their only flux and hooked up to the 
    # flux boundary condition interface

    # flux_bottom = get_boundary_flux(model, bc_uh.bottom, uv, Y, Ya)
    # bulk formula for bottom boundary condition
    uh_1 = Operators.getidx(uh, Operators.Interior(), 1)
    uh_wind = norm(uh_1)
    flux_bottom = Geometry.WVector(Cd * uh_wind) ⊗ uh_1
    bcs_bottom = Operators.SetValue(flux_bottom)

    bcs_top = Operators.SetValue(uh_g)

    # a hacky way to make this Direchlet BC a flux BC
    # TODO: wrap it up in bc module; this might also be helpful in thinking about EDMF bc
    # uv_top = parent(uv)[end]
    # dz_top = axes(uv).center_space.face_local_geometry.WJ[end]
    # flux_top = ν * (uv_top - uvg) / dz / 2
    # bcs_top = ClimaCore.Operators.SetValue(flux_top)
    # vector_vdiv_f2c = Operators.DivergenceF2C(bottom = bcs_bottom, top = bcs_top)
    # scalar_vgrad_c2f = Operators.GradientC2F()

    scalar_vgrad_c2f = Operators.GradientC2F(top = bcs_top)
    vector_vdiv_f2c = Operators.DivergenceF2C(bottom = bcs_bottom)

    # this is a separate issue, need the get_boundary_flux function being able to dispath
    # on different FieldVector, e.g., Scalar, Contravariant3Vector, Contravariant12Vector, etc...
    vadv_c2c = Operators.AdvectionC2C(
        bottom = Operators.SetValue(Geometry.UVVector(FT(0), FT(0))),
        top = Operators.SetValue(Geometry.UVVector(FT(0), FT(0))),
    )

    duh .= (uh .- Ref(uh_g)) .× Ref(Geometry.WVector(f))
    @. duh -= vadv_c2c(w, uh)
    @. duh += vector_vdiv_f2c(ν * scalar_vgrad_c2f(uh))

    # @. duh += ∂c(ν * ∂f(uh)) - vadv_c2c(w, uh)

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
    scalar_vgrad_f22 = Operators.GradientF2C()
    vdiv_c2f = Operators.DivergenceC2F()
    vadv_f2f = Operators.AdvectionF2F()
    @. dw = wbc(
        Geometry.WVector(
            -(scalar_vgrad_c2f(p) / interp_c2f(ρ)) - scalar_vgrad_c2f(Φ),
        ) + vdiv_c2f(ν * scalar_vgrad_f22(w)) - vadv_f2f(w, w),
    )

end
