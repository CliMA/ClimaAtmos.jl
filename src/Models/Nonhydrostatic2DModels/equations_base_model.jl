@inline function rhs_base_model!(dY, Y, Ya, t, _...)
    error("not implemented for this model configuration.")
end

@inline function rhs_base_model!(dY, Y, Ya, t, p, params, hyperdiffusivity, FT)
    # relevant parameters 
    g::FT = CLIMAParameters.Planet.grav(params)
    κ₄::FT = hyperdiffusivity

    # unity tensor for pressure term calculation 
    # in horizontal spectral divergence
    I = Ref(Geometry.Axis2Tensor(
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
        bottom = Operators.SetValue(Geometry.UVector(FT(0))),
        top = Operators.SetValue(Geometry.UVector(FT(0))),
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

    # unpack state
    dYm = dY.base
    Ym = Y.base
    dρ = dYm.ρ
    dρuh = dYm.ρuh
    dρw = dYm.ρw
    ρ = Ym.ρ
    ρuh = Ym.ρuh
    ρw = Ym.ρw

    # density equation
    @. dρ = -hdiv(ρuh)
    @. dρ -= vector_vdiv_f2c(ρw)
    Spaces.weighted_dss!(dρ)

    # hyperdiffusion
    @. dρuh = hwdiv(hgrad(ρuh / ρ))
    Spaces.weighted_dss!(dρuh)
    @. dρuh = -κ₄ * hwdiv(ρ * hgrad(dρuh))

    # horizontal momentum equation
    @. dρuh -= hdiv(ρuh ⊗ ρuh / ρ + p * I)
    @. dρuh -= tensor_vdiv_f2c(ρw ⊗ vector_interp_c2f(ρuh / ρ))
    Spaces.weighted_dss!(dρuh)

    # vertical momentum equation
    uh_f = @. vector_interp_c2f(ρuh / ρ)
    @. dρw = -hdiv(uh_f ⊗ ρw)
    @. dρw += B(
        Geometry.transform(
            Geometry.WAxis(),
            -(scalar_grad_c2f(p)) +
            scalar_interp_c2f(ρ) * Geometry.Covariant3Vector(-g), # TODO!: Not generally a Covariant3Vector
        ) -
        tensor_vdiv_c2f(tensor_interp_f2c(ρw ⊗ ρw / scalar_interp_c2f(ρ))),
    )
    Spaces.weighted_dss!(dρw)

    return dY
end
