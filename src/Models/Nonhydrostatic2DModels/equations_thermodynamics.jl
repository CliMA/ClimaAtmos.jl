@inline function rhs_thermodynamics!(dY, Y, Ya, t, _...)
    error("not implemented for this model configuration.")
end

# TODO: dispatched on base model
@inline function rhs_thermodynamics!(
    dY,
    Y,
    Ya,
    t,
    p,
    ::ConservativeForm,
    ::PotentialTemperature,
    params,
    hyperdiffusivity,
    FT,
)
    ρ = Y.base.ρ
    ρuh = Y.base.ρuh
    ρw = Y.base.ρw
    ρθ = Y.thermodynamics.ρθ

    # operators /w boundary conditions
    hdiv = Operators.Divergence()
    vector_vdiv_f2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.WVector(FT(0))),
        top = Operators.SetValue(Geometry.WVector(FT(0))),
    )
    scalar_interp_c2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )

    @. dY.thermodynamics.ρθ = -hdiv(ρuh * ρθ / ρ)
    @. dY.thermodynamics.ρθ -= vector_vdiv_f2c(ρw * scalar_interp_c2f(ρθ / ρ))
    Spaces.weighted_dss!(dY.thermodynamics.ρθ)
end

@inline function rhs_thermodynamics!(
    dY,
    Y,
    Ya,
    t,
    p,
    ::ConservativeForm,
    ::TotalEnergy,
    params,
    hyperdiffusivity,
    # flux_correction,
    FT,
)
    # parameters
    κ₄ = hyperdiffusivity

    # unpack needed variables
    ρ = Y.base.ρ
    ρuh = Y.base.ρuh
    ρw = Y.base.ρw
    ρe_tot = Y.thermodynamics.ρe_tot

    dρe_tot = dY.thermodynamics.ρe_tot

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
    uvw = @. Geometry.Covariant123Vector(ρuh / ρ) +
       Geometry.Covariant123Vector(interp_f2c(ρw) / ρ)

    # hyperdiffusion
    χe = @. dρe_tot = hwdiv(hgrad((ρe_tot + p) / ρ))
    Spaces.weighted_dss!(dρe_tot)
    @. dρe_tot = -κ₄ * hwdiv(ρ * hgrad(χe))

    # advection 
    @. dρe_tot -= hdiv(uvw * (ρe_tot + p))
    @. dρe_tot -= vector_vdiv_f2c(ρw / interp_c2f(ρ) * interp_c2f(ρe_tot + p))
    @. dρe_tot -= vector_vdiv_f2c(interp_c2f(ρuh / ρ * (ρe_tot + p)))

    # # flux correction
    # if flux_correction
    #     @. dρe_tot += flux_correction_center(w, ρe_tot)
    # end

    # direct stiffness summation
    Spaces.weighted_dss!(dρe_tot)
end
