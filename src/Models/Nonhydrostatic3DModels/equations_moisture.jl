@inline rhs_moisture!(
    dY,
    Y,
    Ya,
    t,
    p,
    ::AbstractBaseModelStyle,
    ::Dry,
    bc_moisture,
    params,
    hyperdiffusivity,
    flux_correction,
    FT,
) = nothing

@inline function rhs_moisture!(
    dY,
    Y,
    Ya,
    t,
    p,
    ::AdvectiveForm,
    ::EquilibriumMoisture,
    bc_moisture,
    params,
    hyperdiffusivity,
    flux_correction,
    FT,
)
    # parameters
    κ₄ = hyperdiffusivity

    # unpack bc
    bc_ρq_tot = bc_moisture.ρq_tot

    # unpack needed variables
    dρq_tot = dY.moisture.ρq_tot
    ρ = Y.base.ρ
    uh = Y.base.uh
    w = Y.base.w
    ρq_tot = Y.moisture.ρq_tot

    # operators /w boundary conditions
    hdiv = Operators.Divergence()
    hwdiv = Operators.WeakDivergence()
    hgrad = Operators.Gradient()
    
    interp_c2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    interp_f2c = Operators.InterpolateF2C()
    flux_correction_center = Operators.FluxCorrectionC2C(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )

    # auxiliary variables
    uvw = @. Geometry.Covariant123Vector(uh) +
       Geometry.Covariant123Vector(interp_f2c(w))

    # hyperdiffusion
    χq = @. dρq_tot = hwdiv(hgrad(ρq_tot / ρ))
    Spaces.weighted_dss!(dρq_tot)
    @. dρq_tot = -κ₄ * hwdiv(ρ * hgrad(χq))

    # advection
    flux_top = get_boundary_flux(bc_ρq_tot.top, ρq_tot, Y, Ya)
    flux_bottom = get_boundary_flux(bc_ρq_tot.bottom, ρq_tot, Y, Ya)
    vector_vdiv_f2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(flux_bottom),
        top = Operators.SetValue(flux_top),
    )

    @. dρq_tot -= hdiv(uvw * (ρq_tot))
    @. dρq_tot -= vector_vdiv_f2c(w * interp_c2f(ρq_tot))
    @. dρq_tot -= vector_vdiv_f2c(interp_c2f(uh * (ρq_tot)))

    # direct stiffness summation
    Spaces.weighted_dss!(dρq_tot)
end
