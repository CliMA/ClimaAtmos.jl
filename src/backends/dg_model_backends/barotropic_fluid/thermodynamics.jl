@inline function calc_pressure(::BarotropicFluid, state, aux, params)
    ρ  = state.ρ
    cₛ = params.cₛ
    ρₒ = params.ρₒ

    return (cₛ * ρ)^2 / (2 * ρₒ)
end

@inline function calc_sound_speed(::BarotropicFluid, state, aux, params)
    ρ = state.ρ
    cₛ = params.cₛ 
    ρₒ = params.ρₒ
    
    return cₛ * sqrt(ρ / ρₒ) 
end