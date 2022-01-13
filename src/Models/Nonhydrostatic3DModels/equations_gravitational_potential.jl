@inline function calculate_gravitational_potential(Y, Ya, params, FT)
    g::FT = CLIMAParameters.Planet.grav(params)
    ρ = Y.base.ρ
    z = Fields.coordinate_field(axes(ρ)).z

    return @. g * z
end
