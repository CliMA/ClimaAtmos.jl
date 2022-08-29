@inline function rhs_thermodynamics!(dY, Y, Ya, t, _...)
    error("not implemented for this model configuration.")
end

# TODO: dispatched on base model
@inline function rhs_thermodynamics!(
    dY,
    Y,
    Ya,
    t,
    ::ConservativeForm,
    ::PotentialTemperature,
    params,
    hyperdiffusivity,
    FT,
)
    κ₄::FT = hyperdiffusivity

    # operators
    # spectral horizontal operators
    hdiv = Operators.Divergence()
    hwdiv = Operators.WeakDivergence()
    hgrad = Operators.Gradient()

    # vertical FD operators with BC's
    # interpolations
    scalar_interp_c2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    # divergence
    vector_vdiv_f2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.WVector(FT(0))),
        top = Operators.SetValue(Geometry.WVector(FT(0))),
    )

    # unpack state and tendency
    ρ = Y.base.ρ
    ρuh = Y.base.ρuh
    ρw = Y.base.ρw
    ρθ = Y.thermodynamics.ρθ
    dρθ = dY.thermodynamics.ρθ

    # hyperdiffusion
    @. dρθ = hwdiv(hgrad(ρθ ./ ρ))
    Spaces.weighted_dss!(dρθ)
    @. dρθ = -κ₄ * hwdiv(ρ * hgrad(dρθ))

    @. dρθ -= vector_vdiv_f2c(ρw * scalar_interp_c2f(ρθ / ρ))
    @. dρθ -= hdiv(ρuh / ρ * ρθ)

    # ### UPWIND FLUX CORRECTION
    # if upwind_correction
    #     @. dρθ += fcc(w, ρθ)
    # end

    Spaces.weighted_dss!(dρθ)
end

@inline function rhs_thermodynamics!(
    dY,
    Y,
    Ya,
    t,
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

    p = Ya.p

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
