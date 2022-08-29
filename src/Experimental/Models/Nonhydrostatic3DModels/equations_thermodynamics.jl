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
    params,
    hyperdiffusivity,
    flux_correction,
    FT,
)
    # parameters
    κ₄ = hyperdiffusivity

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
    vector_vdiv_f2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.WVector(FT(0))),
        top = Operators.SetValue(Geometry.WVector(FT(0))),
    )
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
    params,
    hyperdiffusivity,
    flux_correction,
    FT,
)
    # parameters
    κ₄ = hyperdiffusivity

    # unpack needed variables
    dρe_tot = dY.thermodynamics.ρe_tot
    ρ = Y.base.ρ
    uh = Y.base.uh
    w = Y.base.w
    ρe_tot = Y.thermodynamics.ρe_tot

    # operators /w boundary conditions
    hdiv = Operators.Divergence()
    hwdiv = Operators.WeakDivergence()
    hgrad = Operators.Gradient()
    vector_vdiv_f2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.WVector(FT(0))),
        top = Operators.SetValue(Geometry.WVector(FT(0))),
    )
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
    χe = @. dρe_tot = hwdiv(hgrad((ρe_tot + p) / ρ))
    Spaces.weighted_dss!(dρe_tot)
    @. dρe_tot = -κ₄ * hwdiv(ρ * hgrad(χe))

    # advection 
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

@inline function rhs_thermodynamics!(
    dY,
    Y,
    Ya,
    t,
    p,
    ::AdvectiveForm,
    ::InternalEnergy,
    params,
    hyperdiffusivity,
    flux_correction,
    FT,
)
    # parameters
    κ₄ = hyperdiffusivity

    # unpack needed variables
    dρe_int = dY.thermodynamics.ρe_int
    ρ = Y.base.ρ
    uh = Y.base.uh
    w = Y.base.w
    ρe_int = Y.thermodynamics.ρe_int

    # operators /w boundary conditions
    hdiv = Operators.Divergence()
    hwdiv = Operators.WeakDivergence()
    hgrad = Operators.Gradient()
    scalar_vgrad_c2f = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.Covariant3Vector(FT(0))),
        top = Operators.SetGradient(Geometry.Covariant3Vector(FT(0))),
    )
    vector_vdiv_f2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.WVector(FT(0))),
        top = Operators.SetValue(Geometry.WVector(FT(0))),
    )
    interp_c2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    interp_f2c = Operators.InterpolateF2C()
    flux_correction_center = Operators.FluxCorrectionC2C(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )

    # hyperdiffusion
    χe = @. dρe_int = hwdiv(hgrad((ρe_int + p) / ρ))
    Spaces.weighted_dss!(dρe_int)
    @. dρe_int = -κ₄ * hwdiv(ρ * hgrad(χe))

    # advection 
    uvw = @. Geometry.Covariant123Vector(uh) +
       Geometry.Covariant123Vector(interp_f2c(w))
    @. dρe_int -=
        hdiv(uvw * (ρe_int + p)) -
        dot(uh, Geometry.Contravariant12Vector(hgrad(p)))
    @. dρe_int -=
        vector_vdiv_f2c(interp_c2f(uh * (ρe_int + p))) -
        interp_f2c(dot(w, Geometry.Contravariant3Vector(scalar_vgrad_c2f(p))))

    # flux correction
    if flux_correction
        @. dρe_int += flux_correction_center(w, ρe_int)
    end

    # direct stiffness summation
    Spaces.weighted_dss!(dρe_int)
end
