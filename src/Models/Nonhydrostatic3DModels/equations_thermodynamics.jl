@inline function rhs_thermodynamics!(dY, Y, Ya, t, _...)
    error("not implemented for this model configuration.")
end

@inline function rhs_thermodynamics!(
    dY,
    Y,
    Ya,
    t,
    p,
    ::AdvectiveForm,
    ::PotentialTemperature,
    bc_thermo,
    params,
    hyperdiffusivity,
    flux_correction,
    FT,
)
    # parameters
    κ₄ = hyperdiffusivity

    # unpack bc
    bc_ρθ = bc_thermo.ρθ

    # unpack needed variables
    dρθ = dY.thermodynamics.ρθ
    ρ = Y.base.ρ
    uh = Y.base.uh
    w = Y.base.w
    ρθ = Y.thermodynamics.ρθ

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
    χe = @. dρθ = hwdiv(hgrad(ρθ / ρ))
    Spaces.weighted_dss!(dρθ)
    @. dρθ = -κ₄ * hwdiv(ρ * hgrad(χe))

    # advection
    flux_bottom = get_boundary_flux(bc_ρθ.bottom, ρθ, Y, Ya)
    flux_top = get_boundary_flux(bc_ρθ.top, ρθ, Y, Ya)
    vector_vdiv_f2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(flux_bottom),
        top = Operators.SetValue(flux_top),
    )
    @. dρθ -= hdiv(uvw * (ρθ))
    @. dρθ -= vector_vdiv_f2c(w * interp_c2f(ρθ))
    @. dρθ -= vector_vdiv_f2c(interp_c2f(uh * (ρθ)))

    # flux correction
    if flux_correction
        @. dρθ += flux_correction_center(w, ρθ)
    end

    # direct stiffness summation
    Spaces.weighted_dss!(dρθ)
end

@inline function rhs_thermodynamics!(
    dY,
    Y,
    Ya,
    t,
    p,
    ::AdvectiveForm,
    ::TotalEnergy,
    bc_thermo,
    params,
    hyperdiffusivity,
    flux_correction,
    FT,
)
    # parameters
    κ₄ = hyperdiffusivity

    # unpack bc
    bc_ρe_tot = bc_thermo.ρe_tot

    # unpack needed variables
    dρe_tot = dY.thermodynamics.ρe_tot
    ρ = Y.base.ρ
    uh = Y.base.uh
    w = Y.base.w
    ρe_tot = Y.thermodynamics.ρe_tot

    # operators /w boundary conditions
    hdiv = Operators.Divergence()
    hwdiv = Operators.Divergence()
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
    χe = @. dρe_tot = hwdiv(hgrad(ρe_tot / ρ))
    Spaces.weighted_dss!(dρe_tot)
    @. dρe_tot = -κ₄ * hwdiv(ρ * hgrad(χe))

    # advection
    flux_bottom = get_boundary_flux(bc_ρe_tot.bottom, ρe_tot, Y, Ya)
    flux_top = get_boundary_flux(bc_ρe_tot.top, ρe_tot, Y, Ya)
    vector_vdiv_f2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(flux_bottom),
        top = Operators.SetValue(flux_top),
    )
    @. dρe_tot -= hdiv(uvw * (ρe_tot + p))
    @. dρe_tot -= vector_vdiv_f2c(w * interp_c2f(ρe_tot + p))
    @. dρe_tot -= vector_vdiv_f2c(interp_c2f(uh * (ρe_tot + p)))

    # flux correction
    if flux_correction
        @. dρe_tot += flux_correction_center(w, ρe_tot)
    end

    # direct stiffness summation
    Spaces.weighted_dss!(dρe_tot)
end
