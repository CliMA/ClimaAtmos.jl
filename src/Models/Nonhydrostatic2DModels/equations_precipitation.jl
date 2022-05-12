@inline rhs_precipitation!(
    dY,
    Y,
    Ya,
    t,
    ::AbstractBaseModelStyle,
    ::AbstractMoistureStyle,
    ::Union{NoPrecipitation, PrecipitationRemoval},
    params,
    hyperdiffusivity,
    FT,
) = nothing

@inline function rhs_precipitation!(
    dY,
    Y,
    Ya,
    t,
    ::ConservativeForm,
    ::EquilibriumMoisture,
    ::OneMoment,
    params,
    hyperdiffusivity,
    FT,
)
    # parameters
    κ₄ = hyperdiffusivity

    # unpack needed variables
    dρq_rai = dY.precipitation.ρq_rai
    dρq_sno = dY.precipitation.ρq_sno

    ρ = Y.base.ρ
    ρuh = Y.base.ρuh
    ρw = Y.base.ρw
    ρq_tot = Y.moisture.ρq_tot
    ρq_rai = Y.precipitation.ρq_rai
    ρq_sno = Y.precipitation.ρq_sno
    ρe_tot = Y.thermodynamics.ρe_tot

    S_q_rai = Ya.microphysics_cache.S_q_rai
    S_q_sno = Ya.microphysics_cache.S_q_sno

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
    v_rai = @. Geometry.WVector(
        interp_c2f(
            CM1M.terminal_velocity(params, CM1M.RainType(), ρ, ρq_rai / ρ),
        ) * FT(-1),
    )
    v_sno = @. Geometry.WVector(
        interp_c2f(
            CM1M.terminal_velocity(params, CM1M.SnowType(), ρ, ρq_sno / ρ),
        ) * FT(-1),
    )
    uvw_rai = @. Geometry.Covariant123Vector(ρuh / ρ) +
       Geometry.Covariant123Vector(interp_f2c(ρw) / ρ) +
       Geometry.Covariant123Vector(interp_f2c(v_rai))
    uvw_sno = @. Geometry.Covariant123Vector(ρuh / ρ) +
       Geometry.Covariant123Vector(interp_f2c(ρw) / ρ) +
       Geometry.Covariant123Vector(interp_f2c(v_sno))

    # hyperdiffusion of q_rai
    χq = @. dρq_rai = hwdiv(hgrad(ρq_rai / ρ))
    Spaces.weighted_dss!(dρq_rai)
    @. dρq_rai = -κ₄ * hwdiv(ρ * hgrad(χq))

    # hyperdiffusion of q_sno
    χq = @. dρq_sno = hwdiv(hgrad(ρq_sno / ρ))
    Spaces.weighted_dss!(dρq_sno)
    @. dρq_sno = -κ₄ * hwdiv(ρ * hgrad(χq))

    # advection of q_rai
    @. dρq_rai -= hdiv(uvw_rai * (ρq_rai))
    @. dρq_rai -= vector_vdiv_f2c(ρw / interp_c2f(ρ) * interp_c2f(ρq_rai))
    @. dρq_rai -= vector_vdiv_f2c(v_rai * interp_c2f(ρq_rai))
    @. dρq_rai -= vector_vdiv_f2c(interp_c2f(ρuh / ρ * (ρq_rai)))

    # advection of q_sno
    @. dρq_sno -= hdiv(uvw_sno * (ρq_sno))
    @. dρq_sno -= vector_vdiv_f2c(ρw / interp_c2f(ρ) * interp_c2f(ρq_sno))
    @. dρq_sno -= vector_vdiv_f2c(v_sno * interp_c2f(ρq_sno))
    @. dρq_sno -= vector_vdiv_f2c(interp_c2f(ρuh / ρ * (ρq_sno)))

    # remove precipitation rhs:
    @. dρq_rai += ρ * S_q_rai
    @. dρq_sno += ρ * S_q_sno

    # direct stiffness summation
    Spaces.weighted_dss!(dρq_rai)
    Spaces.weighted_dss!(dρq_sno)
end
