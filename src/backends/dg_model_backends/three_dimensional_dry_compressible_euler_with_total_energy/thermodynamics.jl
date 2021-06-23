

@inline function calc_pressure(eos::DryIdealGas, state, aux, params)
    ρ  = state.ρ

    R_d = calc_gas_constant(eos, state, params)
    T = calc_air_temperature(eos, state, aux, params)

    return ρ * R_d * T
end

@inline function calc_linear_pressure(eos::DryIdealGas, state, aux, params)
    ρ  = state.ρ
    ρe = state.ρe
    Φ  = aux.Φ
    T_0  = params.T_0

    γ = calc_heat_capacity_ratio(eos, state, params)
    cv_d = calc_heat_capacity_at_constant_volume(eos, state, params)

    return (γ - 1) * (ρe - ρ * Φ + ρ * cv_d * T_0) 
end

@inline function calc_very_linear_pressure(eos::DryIdealGas, state, aux, params)
    ρ  = state.ρ
    ρu = state.ρu
    ρe = state.ρe
    Φ  = aux.Φ

    # Reference states
    ρᵣ  = aux.ref_state.ρ
    ρuᵣ = aux.ref_state.ρu

    γ  = calc_heat_capacity_ratio(eos, state, params)

    return (γ - 1) * (ρe - dot(ρuᵣ, ρu) / ρᵣ + ρ * dot(ρuᵣ, ρuᵣ) / (2*ρᵣ^2) - ρ * Φ)
end

@inline function calc_sound_speed(eos::DryIdealGas, state, aux, params)
    ρ  = state.ρ

    γ  = calc_heat_capacity_ratio(eos, state, params)
    p  = calc_pressure(eos, state, aux, params)

    return sqrt(γ * p / ρ)
end

@inline function calc_ref_sound_speed(eos::DryIdealGas, state, aux, params)
    p = aux.ref_state.p
    ρ = aux.ref_state.ρ

    γ = calc_heat_capacity_ratio(eos, state, params)

    return sqrt(γ * p / ρ)
end

@inline function calc_air_temperature(::DryIdealGas, state, aux, params)
  ρ = state.ρ
  ρu = state.ρu
  ρe = state.ρe
  Φ = aux.Φ
  T_0 = params.T_0

  cv_d = calc_heat_capacity_at_constant_volume(DryIdealGas(), state, params)

  e_int = (ρe - ρu' * ρu / 2ρ - ρ * Φ) / ρ
  T = T_0 + e_int / cv_d

  return T
end

@inline function calc_total_specific_enthalpy(eos::DryIdealGas, state, aux, params)
    ρ  = state.ρ
    ρe = state.ρe

    p  = calc_pressure(eos, state, aux, params)

    return (ρe + p) / ρ
end

@inline function calc_heat_capacity_at_constant_pressure(::DryIdealGas, state, params)
    return params.cp_d
end

@inline function calc_heat_capacity_at_constant_volume(::DryIdealGas, state, params)
    return params.cv_d 
end

@inline function calc_gas_constant(::DryIdealGas, state, params)
    return params.R_d
end

@inline function calc_heat_capacity_ratio(eos::AbstractEquationOfState, state, params)
    cp = calc_heat_capacity_at_constant_pressure(eos, state, params)
    cv = calc_heat_capacity_at_constant_volume(eos, state, params)
    γ  = cp/cv

    return γ
end