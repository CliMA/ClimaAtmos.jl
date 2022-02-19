@inline function rhs_base_model!(dY, Y, Ya, t, _...)
    error("not implemented for this model configuration.")
end

@inline function rhs_base_model!(dY, Y, Ya, t, p, bc_base, params, FT)
    # relevant parameters 
    g::FT = CLIMAParameters.Planet.grav(params)

    # unity tensor for pressure term calculation 
    # in horizontal spectral divergence
    I = Ref(Geometry.Axis2Tensor(
        (Geometry.UAxis(), Geometry.UAxis()),
        @SMatrix [FT(1)]
    ),)

    # unpack boundary boundary_conditions
    bc_ρ = bc_base.ρ
    bc_ρuh = bc_base.ρuh
    bc_ρw = bc_base.ρw

    # operators
    # spectral horizontal operators
    hdiv = Operators.Divergence()

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
    flux_bottom = get_boundary_flux(bc_ρ.bottom, ρ, Y, Ya)
    flux_top = get_boundary_flux(bc_ρ.top, ρ, Y, Ya)
    vector_vdiv_f2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(flux_bottom),
        top = Operators.SetValue(flux_top),
    )
    @. dρ = -hdiv(ρuh)
    @. dρ -= vector_vdiv_f2c(ρw)
    Spaces.weighted_dss!(dρ)

    # horizontal momentum equation
    flux_bottom = get_boundary_flux(bc_ρuh.bottom, ρuh, Y, Ya)
    flux_top = get_boundary_flux(bc_ρuh.top, ρuh, Y, Ya)
    # TODO: need to double check
    tensor_vdiv_f2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(
            (flux_bottom ⊗ Geometry.UVector(FT(0))),
        ),
        top = Operators.SetValue(
            (flux_top ⊗ Geometry.UVector(FT(0))),
        ),
    )
    @. dρuh = -hdiv(ρuh ⊗ ρuh / ρ + p * I)
    @. dρuh -= tensor_vdiv_f2c(ρw ⊗ vector_interp_c2f(ρuh / ρ))
    Spaces.weighted_dss!(dρuh)

    # vertical momentum equation
    flux_bottom = get_boundary_flux(bc_ρw.bottom, ρw, Y, Ya)
    flux_top    = get_boundary_flux(bc_ρw.top, ρw, Y, Ya)
    B = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(flux_bottom),
        top = Operators.SetValue(flux_top),
    )
    scalar_grad_c2f = Operators.GradientC2F()
    # TODO: double check how to get this bc from the flux bc set up for ρw
    tensor_vdiv_c2f = Operators.DivergenceC2F(
        bottom = Operators.SetDivergence(Geometry.WVector(FT(0))),
        top = Operators.SetDivergence(Geometry.WVector(FT(0))),
    )

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
