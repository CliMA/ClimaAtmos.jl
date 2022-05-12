@inline function rhs_base_model!(dY, Y, Ya, t, _...)
    error("not implemented for this model configuration.")
end

@inline function rhs_base_model!(dY, Y, Ya, t, params, hyperdiffusivity, FT)
    # unpack parameters
    κ₄::FT = hyperdiffusivity

    # operators
    # unity tensor for pressure term calculation
    # in horizontal spectral divergence
    Ih = Ref(
        Geometry.Axis2Tensor(
            (Geometry.UAxis(), Geometry.UAxis()),
            @SMatrix [FT(1)]
        ),
    )

    # spectral horizontal operators
    hdiv = Operators.Divergence()
    hwdiv = Operators.WeakDivergence()
    hgrad = Operators.Gradient()

    # vertical FD operators with BC's
    # interpolators
    interp_f2c = Operators.InterpolateF2C()
    interp_c2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )

    # gradients
    grad_c2f = Operators.GradientC2F()
    B = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(Geometry.WVector(FT(0))),
        top = Operators.SetValue(Geometry.WVector(FT(0))),
    )

    # divergences
    vector_vdiv_f2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.WVector(FT(0))),
        top = Operators.SetValue(Geometry.WVector(FT(0))),
    )
    tensor_vdiv_c2f = Operators.DivergenceC2F(
        bottom = Operators.SetDivergence(Geometry.WVector(FT(0))),
        top = Operators.SetDivergence(Geometry.WVector(FT(0))),
    )
    tensor_vdiv_f2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(
            Geometry.WVector(FT(0)) ⊗ Geometry.UVector(FT(0)),
        ),
        top = Operators.SetValue(
            Geometry.WVector(FT(0)) ⊗ Geometry.UVector(FT(0)),
        ),
    )


    # upwinding flux correction
    fcc = Operators.FluxCorrectionC2C(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    fcf = Operators.FluxCorrectionF2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )

    # unpack state and tendency
    dρ = dY.base.ρ
    dρuh = dY.base.ρuh
    dρw = dY.base.ρw
    ρ = Y.base.ρ
    ρuh = Y.base.ρuh
    ρw = Y.base.ρw

    # unpack cache
    p = Ya.p
    Φ = Ya.Φ

    uh = @. ρuh / ρ
    w = @. ρw / interp_c2f(ρ)
    wc = @. interp_f2c(ρw) / ρ
    fρ = @. interp_c2f(ρ)

    # hyperdiffusion
    @. dρuh = hwdiv(hgrad(uh))
    @. dρw = hwdiv(hgrad(w))
    Spaces.weighted_dss!(dρuh)
    Spaces.weighted_dss!(dρw)

    @. dρuh = -κ₄ * hwdiv(ρ * hgrad(dρuh))
    @. dρw = -κ₄ * hwdiv(fρ * hgrad(dρw))

    # density equation
    @. dρ = -vector_vdiv_f2c(ρw)
    @. dρ -= hdiv(ρuh)
    Spaces.weighted_dss!(dρ)

    # horizontal momentum equation
    @. dρuh += -tensor_vdiv_f2c(ρw ⊗ interp_c2f(uh))
    @. dρuh -= hdiv(ρuh ⊗ uh + p * Ih)
    Spaces.weighted_dss!(dρuh)

    # vertical momentum
    @. dρw += B(
        Geometry.transform(
            Geometry.WAxis(),
            -(grad_c2f(p)) - interp_c2f(ρ) * grad_c2f(Φ),
        ) - tensor_vdiv_c2f(interp_f2c(ρw ⊗ w)),
    )
    uh_f = @. interp_c2f(ρuh / ρ)
    @. dρw -= hdiv(uh_f ⊗ ρw)

    # ### UPWIND FLUX CORRECTION
    # upwind_correction = true
    # if upwind_correction
    #     @. dρ += fcc(w, ρ)
    #     @. dρuh += fcc(w, ρuh)
    #     @. dρw += fcf(wc, ρw)
    # end

    Spaces.weighted_dss!(dρw)
    return dY
end
