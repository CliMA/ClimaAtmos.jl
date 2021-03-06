@inline function rhs_base_model!(dY, Y, Ya, t, _...)
    error("not implemented for this model configuration.")
end

@inline function rhs_base_model!(dY, Y, Ya, t, params, hyperdiffusivity, FT)
    # unpack parameters
    Îșâ::FT = hyperdiffusivity

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
            Geometry.WVector(FT(0)) â Geometry.UVector(FT(0)),
        ),
        top = Operators.SetValue(
            Geometry.WVector(FT(0)) â Geometry.UVector(FT(0)),
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
    dÏ = dY.base.Ï
    dÏuh = dY.base.Ïuh
    dÏw = dY.base.Ïw
    Ï = Y.base.Ï
    Ïuh = Y.base.Ïuh
    Ïw = Y.base.Ïw

    # unpack cache
    p = Ya.p
    ÎŠ = Ya.ÎŠ

    uh = @. Ïuh / Ï
    w = @. Ïw / interp_c2f(Ï)
    wc = @. interp_f2c(Ïw) / Ï
    fÏ = @. interp_c2f(Ï)

    # hyperdiffusion
    @. dÏuh = hwdiv(hgrad(uh))
    @. dÏw = hwdiv(hgrad(w))
    Spaces.weighted_dss!(dÏuh)
    Spaces.weighted_dss!(dÏw)

    @. dÏuh = -Îșâ * hwdiv(Ï * hgrad(dÏuh))
    @. dÏw = -Îșâ * hwdiv(fÏ * hgrad(dÏw))

    # density equation
    @. dÏ = -vector_vdiv_f2c(Ïw)
    @. dÏ -= hdiv(Ïuh)
    Spaces.weighted_dss!(dÏ)

    # horizontal momentum equation
    @. dÏuh += -tensor_vdiv_f2c(Ïw â interp_c2f(uh))
    @. dÏuh -= hdiv(Ïuh â uh + p * Ih)
    Spaces.weighted_dss!(dÏuh)

    # vertical momentum
    @. dÏw += B(
        Geometry.transform(
            Geometry.WAxis(),
            -(grad_c2f(p)) - interp_c2f(Ï) * grad_c2f(ÎŠ),
        ) - tensor_vdiv_c2f(interp_f2c(Ïw â w)),
    )
    uh_f = @. interp_c2f(Ïuh / Ï)
    @. dÏw -= hdiv(uh_f â Ïw)

    # ### UPWIND FLUX CORRECTION
    # upwind_correction = true
    # if upwind_correction
    #     @. dÏ += fcc(w, Ï)
    #     @. dÏuh += fcc(w, Ïuh)
    #     @. dÏw += fcf(wc, Ïw)
    # end

    Spaces.weighted_dss!(dÏw)
    return dY
end
