@inline function rhs_thermodynamics!(dY, Y, Ya, t, _...)
    error("not implemented for this model configuration.")
end

@inline function rhs_thermodynamics!(
    dY,
    Y,
    Ya,
    t,
    p,
    ::PotentialTemperature,
    params,
    FT,
)
    ρ = Y.base.ρ
    ρuh = Y.base.ρuh
    ρw = Y.base.ρw
    ρθ = Y.thermodynamics.ρθ

    # operators /w boundary conditions
    hdiv = Operators.Divergence()
    vector_vdiv_f2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.WVector(FT(0))),
        top = Operators.SetValue(Geometry.WVector(FT(0))),
    )
    scalar_interp_c2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )

    @. dY.thermodynamics.ρθ = -hdiv(ρuh * ρθ / ρ)
    @. dY.thermodynamics.ρθ -= vector_vdiv_f2c(ρw * scalar_interp_c2f(ρθ / ρ))
    Spaces.weighted_dss!(dY.thermodynamics.ρθ)
end
