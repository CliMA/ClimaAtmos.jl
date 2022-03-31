@inline rhs_moisture!(
    dY,
    Y,
    Ya,
    t,
    ::AbstractBaseModelStyle,
    ::Dry,
    ::NoPrecipitation,
    params,
    hyperdiffusivity,
    FT,
) = nothing

@inline function rhs_moisture!(
    dY,
    Y,
    Ya,
    t,
    ::AdvectiveForm,
    ::EquilibriumMoisture,
    ::NoPrecipitation,
    params,
    hyperdiffusivity,
    FT,
)
    # parameters
    κ₄ = hyperdiffusivity

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
    vector_vdiv_f2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.WVector(FT(0))),
        top = Operators.SetValue(Geometry.WVector(FT(0))),
    )
    interp_c2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    interp_f2c = Operators.InterpolateF2C()

    # auxiliary variables
    uvw = @. Geometry.Covariant123Vector(uh) +
       Geometry.Covariant123Vector(interp_f2c(w))

    # hyperdiffusion
    χq = @. dρq_tot = hwdiv(hgrad(ρq_tot / ρ))
    Spaces.weighted_dss!(dρq_tot)
    @. dρq_tot = -κ₄ * hwdiv(ρ * hgrad(χq))

    # advection
    @. dρq_tot -= hdiv(uvw * (ρq_tot))
    @. dρq_tot -= vector_vdiv_f2c(w * interp_c2f(ρq_tot))
    @. dρq_tot -= vector_vdiv_f2c(interp_c2f(uh * (ρq_tot)))

    # direct stiffness summation
    Spaces.weighted_dss!(dρq_tot)
end

@inline function rhs_moisture!(
    dY,
    Y,
    Ya,
    t,
    ::ConservativeForm,
    ::EquilibriumMoisture,
    ::NoPrecipitation,
    params,
    hyperdiffusivity,
    FT,
)
    # parameters
    κ₄ = hyperdiffusivity

    # unpack needed variables
    dρq_tot = dY.moisture.ρq_tot
    ρ = Y.base.ρ
    ρuh = Y.base.ρuh
    ρw = Y.base.ρw
    ρq_tot = Y.moisture.ρq_tot

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

    # auxiliary variables
    uvw = @. Geometry.Covariant123Vector(ρuh / ρ) +
       Geometry.Covariant123Vector(interp_f2c(ρw) / ρ)

    # hyperdiffusion
    χq = @. dρq_tot = hwdiv(hgrad(ρq_tot / ρ))
    Spaces.weighted_dss!(dρq_tot)
    @. dρq_tot = -κ₄ * hwdiv(ρ * hgrad(χq))
    # dρq_tot .= FT(0)

    # advection
    @. dρq_tot -= hdiv(uvw * (ρq_tot))
    @. dρq_tot -= vector_vdiv_f2c(ρw / interp_c2f(ρ) * interp_c2f(ρq_tot))
    @. dρq_tot -= vector_vdiv_f2c(interp_c2f(ρuh / ρ * (ρq_tot)))

    # direct stiffness summation
    Spaces.weighted_dss!(dρq_tot)
end


@inline function rhs_moisture!(
    dY,
    Y,
    Ya,
    t,
    ::ConservativeForm,
    ::EquilibriumMoisture,
    ::Union{PrecipitationRemoval, OneMoment},
    params,
    hyperdiffusivity,
    FT,
)
    # parameters
    κ₄ = hyperdiffusivity

    # unpack needed variables
    dρq_tot = dY.moisture.ρq_tot
    dρe_tot = dY.thermodynamics.ρe_tot
    dρ = dY.base.ρ

    ρ = Y.base.ρ
    ρuh = Y.base.ρuh
    ρw = Y.base.ρw
    ρq_tot = Y.moisture.ρq_tot
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

    # auxiliary variables
    uvw = @. Geometry.Covariant123Vector(ρuh / ρ) +
       Geometry.Covariant123Vector(interp_f2c(ρw) / ρ)

    # # hyperdiffusion
    χq = @. dρq_tot = hwdiv(hgrad(ρq_tot / ρ))
    Spaces.weighted_dss!(dρq_tot)
    @. dρq_tot = -κ₄ * hwdiv(ρ * hgrad(χq))

    # advection
    @. dρq_tot -= hdiv(uvw * (ρq_tot))
    @. dρq_tot -= vector_vdiv_f2c(ρw / interp_c2f(ρ) * interp_c2f(ρq_tot))
    @. dρq_tot -= vector_vdiv_f2c(interp_c2f(ρuh / ρ * (ρq_tot)))

    # remove precipitation rhs:
    @. dρq_tot += ρ * Ya.microphysics_cache.S_q_tot
    @. dρe_tot += ρ * Ya.microphysics_cache.S_e_tot
    @. dρ += ρ * Ya.microphysics_cache.S_q_tot

    # direct stiffness summation
    Spaces.weighted_dss!(dρq_tot)
    Spaces.weighted_dss!(dρe_tot)
    Spaces.weighted_dss!(dρ)
end
