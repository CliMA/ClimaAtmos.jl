@inline function rhs_thermodynamics!(dY, Y, Ya, t, _...)
    error("not implemented for this model configuration.")
end

@inline function rhs_thermodynamics!(
    dY,
    Y,
    Ya,
    t,
    p,
    ::AdvectiveForm,
    ::PotentialTemperature,
    params,
    flux_correction,
    FT,
)
    # base components
    w = Y.base.w

    # thermodynamics components
    dρθ = dY.thermodynamics.ρθ
    ρθ = Y.thermodynamics.ρθ

    # # gather boundary condition 
    # bc_ρθ = thermo_bc.ρθ

    # potential temperature
    # flux_bottom = get_boundary_flux(bc_ρθ.bottom, ρθ, Y, Ya)
    # flux_top = get_boundary_flux(bc_ρθ.top, ρθ, Y, Ya)
    flux_bottom = FT(0.0)
    flux_top = FT(0.0)
    interp_c2f = Operators.InterpolateC2F()
    vector_vdiv_f2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.WVector(flux_bottom)),
        top = Operators.SetValue(Geometry.WVector(flux_top)),
    )
    # TODO!: Undesirable casting to vector required
    @. dρθ = -vector_vdiv_f2c(w * interp_c2f(ρθ))

end

@inline function rhs_thermodynamics!(
    dY,
    Y,
    Ya,
    t,
    p,
    ::AnelasticAdvectiveForm,
    ::PotentialTemperature,
    params,
    flux_correction,
    FT,
)
    # thermodynamics components
    dρθ = dY.thermodynamics.ρθ

    dρθ .= FT(0)

end
