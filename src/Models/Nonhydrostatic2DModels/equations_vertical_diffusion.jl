@inline function rhs_vertical_diffusion!(dY, Y, Ya, t, _...)
    error("not implemented for this model configuration.")
end

@inline rhs_vertical_diffusion!(
    dY,
    Y,
    Ya,
    t,
    ::AbstractBaseModelStyle,
    ::AbstractThermodynamicsStyle,
    ::AbstractMoistureStyle,
    vert_diffusion_style::NoVerticalDiffusion,
    params,
    FT,
) = nothing

@inline function rhs_vertical_diffusion!(
    dY,
    Y,
    Ya,
    t,
    ::AdvectiveForm,
    ::PotentialTemperature,
    ::Dry,
    vert_diffusion_style::ConstantViscosity,
    params,
    FT,
)

    # viscosity for ConstantViscosity turbulence scheme
    ν = vert_diffusion_style.ν

    # base components
    ρ = Y.base.ρ
    uh = Y.base.uh
    w = Y.base.w

    duh = dY.base.uh
    dw = dY.base.w

    # thermodynamics components
    ρθ = Y.thermodynamics.ρθ
    dρθ = dY.thermodynamics.ρθ

    # horizontal velocity
    flux_bottom = Geometry.WVector(FT(0))
    flux_top = Geometry.WVector(FT(0))
    scalar_vgrad_c2f = Operators.GradientC2F()
    vector_vdiv_f2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(flux_bottom),
        top = Operators.SetValue(flux_top),
    )
    @. duh += vector_vdiv_f2c(ν * scalar_vgrad_c2f(uh))

    # vertical velocity
    flux_bottom = Geometry.WVector(FT(0))
    flux_top = Geometry.WVector(FT(0))

    wbc = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(flux_bottom),
        top = Operators.SetValue(flux_top),
    )
    vdiv_c2f = Operators.DivergenceC2F()
    scalar_vgrad_f2c = Operators.GradientF2C()
    @. dw += wbc(vdiv_c2f(ν * scalar_vgrad_f2c(w)),)

    # thermodynamics
    flux_bottom = Geometry.WVector(FT(0))
    flux_top = Geometry.WVector(FT(0))
    scalar_vgrad_c2f = Operators.GradientC2F()
    vector_vdiv_f2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(flux_bottom),
        top = Operators.SetValue(flux_top),
    )
    scalar_interp_c2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    @. dρθ +=
        vector_vdiv_f2c(ν * scalar_interp_c2f(ρ) * scalar_vgrad_c2f(ρθ / ρ))
end

@inline function rhs_vertical_diffusion!(
    dY,
    Y,
    Ya,
    t,
    ::AdvectiveForm,
    ::TotalEnergy,
    ::Dry,
    vert_diffusion_style::ConstantViscosity,
    params,
    FT,
)
    # viscosity for ConstantViscosity turbulence scheme
    ν = vert_diffusion_style.ν

    # base components
    ρ = Y.base.ρ
    uh = Y.base.uh
    w = Y.base.w

    duh = dY.base.uh
    dw = dY.base.w

    # thermodynamics components
    ρe_tot = Y.thermodynamics.ρe_tot
    dρe_tot = dY.thermodynamics.ρe_tot

    p = Ya.p

    # horizontal velocity
    flux_bottom = Geometry.WVector(FT(0))
    flux_top = Geometry.WVector(FT(0))
    scalar_vgrad_c2f = Operators.GradientC2F()
    vector_vdiv_f2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(flux_bottom),
        top = Operators.SetValue(flux_top),
    )
    @. duh += vector_vdiv_f2c(ν * scalar_vgrad_c2f(uh))

    # vertical velocity
    flux_bottom = Geometry.WVector(FT(0))
    flux_top = Geometry.WVector(FT(0))

    wbc = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(flux_bottom),
        top = Operators.SetValue(flux_top),
    )
    vdiv_c2f = Operators.DivergenceC2F()
    scalar_vgrad_f2c = Operators.GradientF2C()
    @. dw += wbc(vdiv_c2f(ν * scalar_vgrad_f2c(w)),)

    # thermodynamics: diffusion on enthalpy
    flux_bottom = Geometry.WVector(FT(0))
    flux_top = Geometry.WVector(FT(0))
    scalar_vgrad_c2f = Operators.GradientC2F()
    vector_vdiv_f2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(flux_bottom),
        top = Operators.SetValue(flux_top),
    )
    scalar_interp_c2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    @. dρe_tot += vector_vdiv_f2c(
        ν * scalar_interp_c2f(ρ) * scalar_vgrad_c2f((ρe_tot + p) / ρ),
    )
end

@inline function rhs_vertical_diffusion!(
    dY,
    Y,
    Ya,
    t,
    ::ConservativeForm,
    ::PotentialTemperature,
    ::Dry,
    vert_diffusion_style::ConstantViscosity,
    params,
    FT,
)
    # viscosity for ConstantViscosity turbulence scheme
    ν = vert_diffusion_style.ν

    # base components
    ρ = Y.base.ρ
    ρuh = Y.base.ρuh
    ρw = Y.base.ρw

    dρuh = dY.base.ρuh
    dρw = dY.base.ρw

    # thermodynamics components
    ρθ = Y.thermodynamics.ρθ
    dρθ = dY.thermodynamics.ρθ

    # horizontal velocity
    flux_bottom = Geometry.WVector(FT(0))
    flux_top = Geometry.WVector(FT(0))
    scalar_vgrad_c2f = Operators.GradientC2F()
    vector_vdiv_f2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(flux_bottom),
        top = Operators.SetValue(flux_top),
    )
    scalar_interp_c2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    @. dρuh +=
        vector_vdiv_f2c(ν * scalar_interp_c2f(ρ) * scalar_vgrad_c2f(ρuh / ρ))

    # vertical velocity
    flux_bottom = Geometry.WVector(FT(0))
    flux_top = Geometry.WVector(FT(0))

    wbc = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(flux_bottom),
        top = Operators.SetValue(flux_top),
    )
    vdiv_c2f = Operators.DivergenceC2F()
    scalar_vgrad_f2c = Operators.GradientF2C()
    @. dρw += wbc(vdiv_c2f(ν * ρ * scalar_vgrad_f2c(ρw / ρ)),)

    # thermodynamics
    flux_bottom = Geometry.WVector(FT(0))
    flux_top = Geometry.WVector(FT(0))
    scalar_vgrad_c2f = Operators.GradientC2F()
    vector_vdiv_f2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(flux_bottom),
        top = Operators.SetValue(flux_top),
    )
    scalar_interp_c2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    @. dρθ +=
        vector_vdiv_f2c(ν * scalar_interp_c2f(ρ) * scalar_vgrad_c2f(ρθ / ρ))
end

@inline function rhs_vertical_diffusion!(
    dY,
    Y,
    Ya,
    t,
    ::ConservativeForm,
    ::TotalEnergy,
    ::Dry,
    vert_diffusion_style::ConstantViscosity,
    params,
    FT,
)
    # viscosity for ConstantViscosity turbulence scheme
    ν = vert_diffusion_style.ν

    # base components
    ρ = Y.base.ρ
    ρuh = Y.base.ρuh
    ρw = Y.base.ρw

    dρuh = dY.base.ρuh
    dρw = dY.base.ρw

    # thermodynamics components
    ρe_tot = Y.thermodynamics.ρe_tot
    dρe_tot = dY.thermodynamics.ρe_tot

    p = Ya.p

    # horizontal velocity
    flux_bottom = Geometry.WVector(FT(0))
    flux_top = Geometry.WVector(FT(0))
    scalar_vgrad_c2f = Operators.GradientC2F()
    vector_vdiv_f2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(flux_bottom),
        top = Operators.SetValue(flux_top),
    )
    scalar_interp_c2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    @. dρuh +=
        vector_vdiv_f2c(ν * scalar_interp_c2f(ρ) * scalar_vgrad_c2f(ρuh / ρ))

    # vertical velocity
    flux_bottom = Geometry.WVector(FT(0))
    flux_top = Geometry.WVector(FT(0))

    wbc = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(flux_bottom),
        top = Operators.SetValue(flux_top),
    )
    vdiv_c2f = Operators.DivergenceC2F()
    scalar_vgrad_f2c = Operators.GradientF2C()
    @. dρw += wbc(vdiv_c2f(ν * ρ * scalar_vgrad_f2c(ρw / ρ)),)

    # thermodynamics: diffusion on enthalpy
    flux_bottom = Geometry.WVector(FT(0))
    flux_top = Geometry.WVector(FT(0))
    scalar_vgrad_c2f = Operators.GradientC2F()
    vector_vdiv_f2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(flux_bottom),
        top = Operators.SetValue(flux_top),
    )
    scalar_interp_c2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    @. dρe_tot += vector_vdiv_f2c(
        ν * scalar_interp_c2f(ρ) * scalar_vgrad_c2f((ρe_tot + p) / ρ),
    )
end
