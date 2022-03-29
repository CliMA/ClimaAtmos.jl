@inline function rhs_base_model!(dY, Y, Ya, t, _...)
    error("not implemented for this model configuration.")
end

@inline function rhs_base_model!(
    dY,
    Y,
    Ya,
    t,
    p,
    Φ,
    params,
    hyperdiffusivity,
    FT,
)
    # relevant parameters 
    g::FT = CLIMAParameters.Planet.grav(params)
    κ₄::FT = hyperdiffusivity

    # unity tensor for pressure term calculation 
    # in horizontal spectral divergence
    Ih = Ref(Geometry.Axis2Tensor(
        (Geometry.UAxis(), Geometry.UAxis()),
        @SMatrix [FT(1)]
    ),)

    # operators
    # spectral horizontal operators
    hdiv = Operators.Divergence()
    hwdiv = Operators.WeakDivergence()
    hgrad = Operators.Gradient()

    # vertical FD operators with BC's
    # interpolators
    scalar_interp_c2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    vector_interp_c2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    tensor_interp_f2c = Operators.InterpolateF2C()

    # gradients
    scalar_grad_c2f = Operators.GradientC2F()
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

    # unpack state and tendency
    dρ = dY.base.ρ
    dρuh = dY.base.ρuh
    dρw = dY.base.ρw
    ρ = Y.base.ρ
    ρuh = Y.base.ρuh
    ρw = Y.base.ρw

    fρ = @. scalar_interp_c2f(ρ)
    uh = @. ρuh / ρ
    w = @. ρw / fρ

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
    @. dρuh -= tensor_vdiv_f2c(ρw ⊗ vector_interp_c2f(ρuh / ρ))
    @. dρuh -= hdiv(ρuh ⊗ ρuh / ρ + p * Ih)
    Spaces.weighted_dss!(dρuh)

    # vertical momentum equation
    @. dρw += B(
        Geometry.transform(
            Geometry.WAxis(),
            -(scalar_grad_c2f(p)) - scalar_interp_c2f(ρ) * scalar_grad_c2f(Φ),
        ) -
        tensor_vdiv_c2f(tensor_interp_f2c(ρw ⊗ ρw / scalar_interp_c2f(ρ))),
    )
    uh_f = @. vector_interp_c2f(ρuh / ρ)
    @. dρw = -hdiv(uh_f ⊗ ρw)
    Spaces.weighted_dss!(dρw)

    return dY
end
