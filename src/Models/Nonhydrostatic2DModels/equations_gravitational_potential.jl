@inline function calculate_gravitational_potential(Y, Ya, params, FT)
    g::FT = CLIMAParameters.Planet.grav(params)
    ρ = Y.base.ρ
    z = Fields.coordinate_field(axes(ρ)).z

    return @. g * z
end

@inline function calculate_kinetic_energy(Y, Ya, params, FT)
    cρ = Y.base.ρ
    cρuₕ = Y.base.ρuh # Covariant12Vector on centers
    fρw = Y.base.ρw # Covariant3Vector on faces
    If2c = Operators.InterpolateF2C()

    cuvw =
        Geometry.Covariant123Vector.(cρuₕ ./ cρ) .+
        Geometry.Covariant123Vector.(If2c.(fρw) ./ cρ)
    cK = @. norm_sqr(cuvw) / 2
    return cK
end
