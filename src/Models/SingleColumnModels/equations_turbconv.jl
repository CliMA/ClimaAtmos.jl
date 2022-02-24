@inline function rhs_turbconv!(dY, Y, Ya, t, _...)
    error("not implemented for this model configuration.")
end

@inline function rhs_turbconv!(
    dY,
    Y,
    Ya,
    t,
    p,
    ::AdvectiveForm,
    ::PotentialTemperature,
    ::Dry,
    turbconv_style::ConstantViscosity,
    params,
    FT,
)

    # viscosity for ConstantViscosity turbulence scheme
    ν = turbconv_style.ν

    # experiment specific parameters
    uh_g = params.uh_g
    Cd = params.Cd

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
    # bc for uh is very tricky, we want to support only flux boundary conditions in ClimaAtmos
    # but will all for user-customized function to provide their only flux and hooked up to the 
    # flux boundary condition interface

    # flux_bottom = get_boundary_flux(model, bc_uh.bottom, uv, Y, Ya)
    # bulk formula for bottom boundary condition
    uh_1 = Operators.getidx(uh, Operators.Interior(), 1)
    uh_wind = norm(uh_1)
    flux_bottom = Geometry.WVector(Cd * uh_wind) ⊗ uh_1
    bcs_bottom = Operators.SetValue(flux_bottom)

    bcs_top = Operators.SetValue(uh_g)

    # a hacky way to make this Direchlet BC a flux BC
    # TODO: wrap it up in bc module; this might also be helpful in thinking about EDMF bc
    # uv_top = parent(uv)[end]
    # dz_top = axes(uv).center_space.face_local_geometry.WJ[end]
    # flux_top = ν * (uv_top - uvg) / dz / 2
    # bcs_top = ClimaCore.Operators.SetValue(flux_top)
    # vector_vdiv_f2c = Operators.DivergenceF2C(bottom = bcs_bottom, top = bcs_top)
    # scalar_vgrad_c2f = Operators.GradientC2F()

    scalar_vgrad_c2f = Operators.GradientC2F(top = bcs_top)
    vector_vdiv_f2c = Operators.DivergenceF2C(bottom = bcs_bottom)
    @. duh += vector_vdiv_f2c(ν * scalar_vgrad_c2f(uh))

    # vertical velocity
    # flux_bottom = get_boundary_flux(model, bc_w.bottom, w, Y, Ya)
    # flux_top = get_boundary_flux(model, bc_w.top, w, Y, Ya)
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
    # flux_bottom = get_boundary_flux(bc_ρθ.bottom, ρθ, Y, Ya)
    # flux_top = get_boundary_flux(bc_ρθ.top, ρθ, Y, Ya)
    flux_bottom = Geometry.WVector(FT(0.0))
    flux_top = Geometry.WVector(FT(0.0))
    scalar_vgrad_c2f = Operators.GradientC2F()
    vector_vdiv_f2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(flux_bottom),
        top = Operators.SetValue(flux_top),
    )
    @. dρθ += ρ * vector_vdiv_f2c(ν * scalar_vgrad_c2f(ρθ / ρ))
end

@inline function rhs_turbconv!(
    dY,
    Y,
    Ya,
    t,
    p,
    ::AnelasticAdvectiveForm,
    ::PotentialTemperature,
    ::Dry,
    turbconv_style::ConstantViscosity,
    params,
    FT,
)

    # viscosity for ConstantViscosity turbulence scheme
    ν = turbconv_style.ν

    # experiment specific parameters
    uh_g = params.uh_g
    Cd = params.Cd

    # base components
    ρ = Y.base.ρ
    uh = Y.base.uh
    duh = dY.base.uh

    # thermodynamics components
    ρθ = Y.thermodynamics.ρθ
    dρθ = dY.thermodynamics.ρθ

    # horizontal velocity
    # bc for uh is very tricky, we want to support only flux boundary conditions in ClimaAtmos
    # but will all for user-customized function to provide their only flux and hooked up to the 
    # flux boundary condition interface

    # flux_bottom = get_boundary_flux(model, bc_uh.bottom, uv, Y, Ya)
    # bulk formula for bottom boundary condition
    uh_1 = Operators.getidx(uh, Operators.Interior(), 1)
    uh_wind = norm(uh_1)
    flux_bottom = Geometry.WVector(Cd * uh_wind) ⊗ uh_1
    bcs_bottom = Operators.SetValue(flux_bottom)

    bcs_top = Operators.SetValue(uh_g)

    # a hacky way to make this Direchlet BC a flux BC
    # TODO: wrap it up in bc module; this might also be helpful in thinking about EDMF bc
    # uv_top = parent(uv)[end]
    # dz_top = axes(uv).center_space.face_local_geometry.WJ[end]
    # flux_top = ν * (uv_top - uvg) / dz / 2
    # bcs_top = ClimaCore.Operators.SetValue(flux_top)
    # vector_vdiv_f2c = Operators.DivergenceF2C(bottom = bcs_bottom, top = bcs_top)
    # scalar_vgrad_c2f = Operators.GradientC2F()

    scalar_vgrad_c2f = Operators.GradientC2F(top = bcs_top)
    vector_vdiv_f2c = Operators.DivergenceF2C(bottom = bcs_bottom)
    @. duh += vector_vdiv_f2c(ν * scalar_vgrad_c2f(uh))

    # thermodynamics
    # flux_bottom = get_boundary_flux(bc_ρθ.bottom, ρθ, Y, Ya)
    # flux_top = get_boundary_flux(bc_ρθ.top, ρθ, Y, Ya)
    flux_bottom = Geometry.WVector(FT(0.0))
    flux_top = Geometry.WVector(FT(0.0))
    scalar_vgrad_c2f = Operators.GradientC2F()
    vector_vdiv_f2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(flux_bottom),
        top = Operators.SetValue(flux_top),
    )
    @. dρθ += ρ * vector_vdiv_f2c(ν * scalar_vgrad_c2f(ρθ / ρ))
end
