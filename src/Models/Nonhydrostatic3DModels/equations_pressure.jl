@inline function calculate_pressure(Y, Ya, _...)
    error("not implemented for this model configuration.")
end

@inline function calculate_pressure(
    Y,
    Ya,
    ::AbstractBaseModelStyle,
    ::PotentialTemperature,
    ::Dry,
    params,
    FT,
)
    ρ = Y.base.ρ
    ρθ = Y.thermodynamics.ρθ

    p = @. Thermodynamics.air_pressure(Thermodynamics.PhaseDry_ρθ(
        params,
        ρ,
        ρθ / ρ,
    ))

    return p
end

@inline function calculate_pressure(
    Y,
    Ya,
    ::AdvectiveForm,
    ::TotalEnergy,
    ::Dry,
    params,
    FT,
)
    ρ = Y.base.ρ
    uh = Y.base.uh
    w = Y.base.w
    ρe_tot = Y.thermodynamics.ρe_tot

    interp_f2c = Operators.InterpolateF2C()

    z = Fields.coordinate_field(axes(ρ)).z
    uvw = @. Geometry.Covariant123Vector(uh) +
       Geometry.Covariant123Vector(interp_f2c(w))
    Φ = calculate_gravitational_potential(Y, Ya, params, FT)

    e_int = @. ρe_tot / ρ - Φ - norm(uvw)^2 / 2
    p = Thermodynamics.air_pressure.(Thermodynamics.PhaseDry.(params, e_int, ρ))

    return p
end

@inline function calculate_pressure(
    Y,
    Ya,
    ::AdvectiveForm,
    ::TotalEnergy,
    ::EquilibriumMoisture,
    params,
    FT,
)
    ρ = Y.base.ρ
    uh = Y.base.uh
    w = Y.base.w
    ρe_tot = Y.thermodynamics.ρe_tot
    ρq_tot = Y.moisture.ρq_tot

    interp_f2c = Operators.InterpolateF2C()

    #TODO - why do we need z here?
    z = Fields.coordinate_field(axes(ρ)).z
    uvw = @. Geometry.Covariant123Vector(uh) +
       Geometry.Covariant123Vector(interp_f2c(w))
    Φ = calculate_gravitational_potential(Y, Ya, params, FT)
    e_int = @. ρe_tot / ρ - Φ - norm(uvw)^2 / 2
    q_tot = @. ρq_tot / ρ

    # saturation adjustment
    p =
        Thermodynamics.air_pressure.(Thermodynamics.PhaseEquil_ρeq.(
            Ref(params),
            ρ,
            e_int,
            q_tot,
        ))

    return p
end
