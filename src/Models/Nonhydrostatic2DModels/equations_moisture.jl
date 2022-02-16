@inline rhs_moisture!(
    dY,
    Y,
    Ya,
    t,
    p,
    Φ,
    K,
    ::AbstractBaseModelStyle,
    ::Dry,
    ::NoPrecipitation,
    params,
    # hyperdiffusivity,
    FT,
) = nothing

@inline function rhs_moisture!(
    dY,
    Y,
    Ya,
    t,
    p,
    Φ,
    K,
    ::AdvectiveForm,
    ::EquilibriumMoisture,
    ::NoPrecipitation,
    params,
    # hyperdiffusivity,
    FT,
)
    # parameters
    κ₄ = FT(0) #hyperdiffusivity

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
    # flux_correction_center = Operators.FluxCorrectionC2C(
    #     bottom = Operators.Extrapolate(),
    #     top = Operators.Extrapolate(),
    # )

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
    p,
    Φ,
    K,
    ::ConservativeForm,
    ::EquilibriumMoisture,
    ::NoPrecipitation,
    params,
    # hyperdiffusivity,
    FT,
)
    # parameters
    #κ₄ = FT(0) #hyperdiffusivity

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
    # flux_correction_center = Operators.FluxCorrectionC2C(
    #     bottom = Operators.Extrapolate(),
    #     top = Operators.Extrapolate(),
    # )

    # auxiliary variables
    uvw = @. Geometry.Covariant123Vector(ρuh/ρ) +
       Geometry.Covariant123Vector(interp_f2c(ρw)/ρ)

    # # hyperdiffusion
    # χq = @. dρq_tot = hwdiv(hgrad(ρq_tot / ρ))
    # Spaces.weighted_dss!(dρq_tot)
    # @. dρq_tot = -κ₄ * hwdiv(ρ * hgrad(χq))
    dρq_tot .= FT(0)

    # advection
    @. dρq_tot -= hdiv(uvw * (ρq_tot))
    @. dρq_tot -= vector_vdiv_f2c(ρw / interp_c2f(ρ) * interp_c2f(ρq_tot))
    @. dρq_tot -= vector_vdiv_f2c(interp_c2f(ρuh/ρ * (ρq_tot)))

    # direct stiffness summation
    Spaces.weighted_dss!(dρq_tot)
end


@inline function rhs_moisture!(
    dY,
    Y,
    Ya,
    t,
    p,
    Φ,
    K,
    ::ConservativeForm,
    ::EquilibriumMoisture,
    ::PrecipitationRemoval,
    params,
    # hyperdiffusivity,
    FT,
)
    # parameters
    # κ₄ = hyperdiffusivity

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
    # flux_correction_center = Operators.FluxCorrectionC2C(
    #     bottom = Operators.Extrapolate(),
    #     top = Operators.Extrapolate(),
    # )

    # auxiliary variables
    uvw = @. Geometry.Covariant123Vector(ρuh/ρ) +
       Geometry.Covariant123Vector(interp_f2c(ρw)/ρ)
   
    # # hyperdiffusion
    # χq = @. dρq_tot = hwdiv(hgrad(ρq_tot / ρ))
    # Spaces.weighted_dss!(dρq_tot)
    # @. dρq_tot = -κ₄ * hwdiv(ρ * hgrad(χq))
    dρq_tot .= FT(0)
 

    # advection
    @. dρq_tot -= hdiv(uvw * (ρq_tot))
    @. dρq_tot -= vector_vdiv_f2c(ρw /interp_c2f(ρ) * interp_c2f(ρq_tot))
    @. dρq_tot -= vector_vdiv_f2c(interp_c2f(ρuh / ρ * (ρq_tot)))

    # remove precipitation rhs:

    # TODO move \Phi and K to Ya
    e_int = @. ρe_tot / ρ - Φ - K
    q_tot = @. ρq_tot / ρ
   
    # saturation adjustment (repeated in pressure) TODO - move to Ya
    # (we could cache the temperature, then creating ts would not be expensive)
    ts = Thermodynamics.PhaseEquil_ρeq.(Ref(params), ρ, e_int, q_tot)

    # precipitation removal
    q = @. Thermodynamics.PhasePartition(ts)
    λ = @. Thermodynamics.liquid_fraction(ts)
    I_l = @. Thermodynamics.internal_energy_liquid(ts)
    I_i = @. Thermodynamics.internal_energy_ice(ts)
    S_qt = @. CloudMicrophysics.Microphysics_0M.remove_precipitation(params, q)
    S_e = @. (λ * I_l + (1 - λ) * I_i + Φ) * S_qt

    @. dρq_tot += ρ * S_qt
    @. dρe_tot += ρ * S_e
    @. dρ = ρ * S_qt

    # direct stiffness summation
    Spaces.weighted_dss!(dρq_tot)
    Spaces.weighted_dss!(dρe_tot)
    Spaces.weighted_dss!(dρ)
end