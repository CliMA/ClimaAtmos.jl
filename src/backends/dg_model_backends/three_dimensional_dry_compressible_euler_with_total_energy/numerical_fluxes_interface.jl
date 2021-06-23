struct RefanovFlux <: NumericalFluxFirstOrder end 
struct CentralVolumeFlux <: NumericalFluxFirstOrder end
struct KGVolumeFlux <: NumericalFluxFirstOrder end
struct LinearKGVolumeFlux <: NumericalFluxFirstOrder end
struct VeryLinearKGVolumeFlux <: NumericalFluxFirstOrder end

function numerical_volume_fluctuation_flux_first_order!(
    ::NumericalFluxFirstOrder,
    balance_law::ThreeDimensionalDryCompressibleEulerWithTotalEnergy,
    source::Grad,
    state_1::Vars,
    aux_1::Vars,
    state_2::Vars,
    aux_2::Vars,
)
    if haskey(balance_law.sources, :gravity)
        ρ_1, ρ_2 = state_1.ρ, state_2.ρ
        Φ_1, Φ_2 = aux_1.Φ, aux_2.Φ

        α = ave(ρ_1, ρ_2) * 0.5

        source.ρu -= α * (Φ_1 - Φ_2) * I
    end
end    

function numerical_volume_conservative_flux_first_order!(
    ::CentralVolumeFlux,
    m::ThreeDimensionalDryCompressibleEulerWithTotalEnergy,
    F::Grad,
    state_1::Vars,
    aux_1::Vars,
    state_2::Vars,
    aux_2::Vars,
)
    FT = eltype(F)
    F_1 = similar(F)
    flux_first_order!(m, F_1, state_1, aux_1, FT(0), EveryDirection())

    F_2 = similar(F)
    flux_first_order!(m, F_2, state_2, aux_2, FT(0), EveryDirection())

    parent(F) .= (parent(F_1) .+ parent(F_2)) ./ 2
end

function numerical_volume_conservative_flux_first_order!(
    ::KGVolumeFlux,
    balance_law::ThreeDimensionalDryCompressibleEulerWithTotalEnergy,
    F::Grad,
    state_1::Vars,
    aux_1::Vars,
    state_2::Vars,
    aux_2::Vars,
)
    eos = balance_law.equation_of_state
    parameters = balance_law.parameters

    ρ_1 = state_1.ρ
    ρu_1 = state_1.ρu
    ρe_1 = state_1.ρe
    u_1 = ρu_1 / ρ_1
    e_1 = ρe_1 / ρ_1
    p_1 = calc_pressure(eos, state_1, aux_1, parameters)

    ρ_2 = state_2.ρ
    ρu_2 = state_2.ρu
    ρe_2 = state_2.ρe
    u_2 = ρu_2 / ρ_2
    e_2 = ρe_2 / ρ_2
    p_2 = calc_pressure(eos, state_2, aux_2, parameters)

    ρ_avg = ave(ρ_1, ρ_2)
    u_avg = ave(u_1, u_2)
    e_avg = ave(e_1, e_2)
    p_avg = ave(p_1, p_2)

    F.ρ  = ρ_avg * u_avg
    F.ρu = p_avg * I + ρ_avg * u_avg .* u_avg'
    F.ρe = ρ_avg * u_avg * e_avg + p_avg * u_avg
end

function numerical_flux_first_order!(
    ::Nothing, 
    ::ThreeDimensionalDryCompressibleEulerWithTotalEnergy, 
    _...,
)
    return nothing
end

function numerical_flux_first_order!(
    ::RoeNumericalFlux,
    balance_law::ThreeDimensionalDryCompressibleEulerWithTotalEnergy,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
    direction,
) where {S, A}

    numerical_flux_first_order!(
        CentralNumericalFluxFirstOrder(),
        balance_law,
        fluxᵀn,
        normal_vector,
        state_prognostic⁻,
        state_auxiliary⁻,
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
        direction,
    )
    eos = balance_law.equation_of_state
    parameters = balance_law.parameters

    cv_d = parameters.cv_d
    T_0  = parameters.T_0

    Φ = state_auxiliary⁻.Φ #Φ⁻ and Φ⁺ have the same value
    ρ⁻ = state_prognostic⁻.ρ
    ρu⁻ = state_prognostic⁻.ρu
    u⁻ = ρu⁻ / ρ⁻

    p⁻ = calc_pressure(eos, state_prognostic⁻, state_auxiliary⁻, parameters)
    c⁻ = calc_sound_speed(eos, state_prognostic⁻, state_auxiliary⁻, parameters)
    h⁻ = calc_total_specific_enthalpy(eos, state_prognostic⁻, state_auxiliary⁻, parameters)

    ρ⁺ = state_prognostic⁺.ρ
    ρu⁺ = state_prognostic⁺.ρu
    u⁺ = ρu⁺ / ρ⁺

    p⁺ = calc_pressure(eos, state_prognostic⁺, state_auxiliary⁺, parameters)
    c⁺ = calc_sound_speed(eos, state_prognostic⁺, state_auxiliary⁺, parameters)
    h⁺ = calc_total_specific_enthalpy(eos, state_prognostic⁺, state_auxiliary⁺, parameters)

    ρ̃ = sqrt(ρ⁻ * ρ⁺)
    ũ = roe_average(ρ⁻, ρ⁺, u⁻, u⁺)
    h̃ = roe_average(ρ⁻, ρ⁺, h⁻, h⁺)
    c̃ = sqrt(roe_average(ρ⁻, ρ⁺, c⁻^2, c⁺^2))

    ũᵀn = ũ' * normal_vector

    Δρ = ρ⁺ - ρ⁻
    Δp = p⁺ - p⁻
    Δu = u⁺ - u⁻
    Δuᵀn = Δu' * normal_vector

    w1 = abs(ũᵀn - c̃) * (Δp - ρ̃ * c̃ * Δuᵀn) / (2 * c̃^2)
    w2 = abs(ũᵀn + c̃) * (Δp + ρ̃ * c̃ * Δuᵀn) / (2 * c̃^2)
    w3 = abs(ũᵀn) * (Δρ - Δp / c̃^2)
    w4 = abs(ũᵀn) * ρ̃

    fluxᵀn.ρ -= (w1 + w2 + w3) / 2
    fluxᵀn.ρu -=
        (
            w1 * (ũ - c̃ * normal_vector) +
            w2 * (ũ + c̃ * normal_vector) +
            w3 * ũ +
            w4 * (Δu - Δuᵀn * normal_vector)
        ) / 2
    fluxᵀn.ρe -=
        (
            w1 * (h̃ - c̃ * ũᵀn) +
            w2 * (h̃ + c̃ * ũᵀn) +
            w3 * (ũ' * ũ / 2 + Φ - T_0 * cv_d) +
            w4 * (ũ' * Δu - ũᵀn * Δuᵀn)
        ) / 2
end

function numerical_flux_first_order!(
    ::LMARSNumericalFlux,
    balance_law::ThreeDimensionalDryCompressibleEulerWithTotalEnergy,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
    direction,
) where {S, A}
    FT = eltype(fluxᵀn)
    eos = balance_law.equation_of_state
    parameters = balance_law.parameters

    ρ⁻ = state_prognostic⁻.ρ
    ρu⁻ = state_prognostic⁻.ρu
    u⁻ = ρu⁻ / ρ⁻
    uᵀn⁻ = u⁻' * normal_vector

    p⁻ = calc_pressure(eos, state_prognostic⁻, state_auxiliary⁻, parameters)
    # if the reference state is removed in the momentum equations (meaning p-p_ref is used for pressure gradient force) then we should remove the reference pressure
    #     p⁻ -= state_auxiliary⁻.ref_state.p
    # end
    c⁻ = calc_sound_speed(eos, state_prognostic⁻, state_auxiliary⁻, parameters)
    h⁻ = calc_total_specific_enthalpy(eos, state_prognostic⁻, state_auxiliary⁻, parameters)

    ρ⁺ = state_prognostic⁺.ρ
    ρu⁺ = state_prognostic⁺.ρu
    u⁺ = ρu⁺ / ρ⁺
    uᵀn⁺ = u⁺' * normal_vector

    p⁺ = calc_pressure(eos, state_prognostic⁺, state_auxiliary⁺, parameters)
    # if the reference state is removed in the momentum equations (meaning p-p_ref is used for pressure gradient force) then we should remove the reference pressure
    #     p⁺ -= state_auxiliary⁺.ref_state.p
    # end
    # c⁺ = calc_sound_speed(eos, state_prognostic⁺, state_auxiliary⁺, parameters)
    h⁺ = calc_total_specific_enthalpy(eos, state_prognostic⁺, state_auxiliary⁺, parameters)

    # Eqn (49), (50), β the tuning parameter
    β = FT(1)
    u_half = 1 / 2 * (uᵀn⁺ + uᵀn⁻) - β * 1 / (ρ⁻ + ρ⁺) / c⁻ * (p⁺ - p⁻)
    p_half = 1 / 2 * (p⁺ + p⁻) - β * ((ρ⁻ + ρ⁺) * c⁻) / 4 * (uᵀn⁺ - uᵀn⁻)

    # Eqn (46), (47)
    ρ_b = u_half > FT(0) ? ρ⁻ : ρ⁺
    ρu_b = u_half > FT(0) ? ρu⁻ : ρu⁺
    ρh_b = u_half > FT(0) ? ρ⁻ * h⁻ : ρ⁺ * h⁺

    # Update fluxes Eqn (18)
    fluxᵀn.ρ = ρ_b * u_half
    fluxᵀn.ρu = ρu_b * u_half .+ p_half * normal_vector
    fluxᵀn.ρe = ρh_b * u_half
end

function numerical_flux_first_order!(
    ::RefanovFlux,
    model::ThreeDimensionalDryCompressibleEulerWithTotalEnergy,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state⁻::Vars{S},
    aux⁻::Vars{A},
    state⁺::Vars{S},
    aux⁺::Vars{A},
    t,
    direction,
) where {S, A}

    numerical_flux_first_order!(
        CentralNumericalFluxFirstOrder(),
        model,
        fluxᵀn,
        normal_vector,
        state⁻,
        aux⁻,
        state⁺,
        aux⁺,
        t,
        direction,
    )
    eos = model.equation_of_state
    parameters = balance_law.parameters
    
    c⁻ = calc_ref_sound_speed(eos, state⁻, aux⁻, parameters)
    c⁺ = calc_ref_sound_speed(eos, state⁺, aux⁺, parameters)
    c = max(c⁻, c⁺)

    # - states
    ρ⁻ = state⁻.ρ
    ρu⁻ = state⁻.ρu
    ρe⁻ = state⁻.ρe

    # + states
    ρ⁺ = state⁺.ρ
    ρu⁺ = state⁺.ρu
    ρe⁺ = state⁺.ρe

    Δρ = ρ⁺ - ρ⁻
    Δρu = ρu⁺ - ρu⁻
    Δρe = ρe⁺ - ρe⁻

    fluxᵀn.ρ  -= c * Δρ  * 0.5
    fluxᵀn.ρu -= c * Δρu * 0.5
    fluxᵀn.ρe -= c * Δρe * 0.5
end

function numerical_flux_second_order!(
    ::Nothing, 
    ::ThreeDimensionalDryCompressibleEulerWithTotalEnergy, 
    _...,
) 
    return nothing
end

# utils
roe_average(ρ⁻, ρ⁺, var⁻, var⁺) =
    (sqrt(ρ⁻) * var⁻ + sqrt(ρ⁺) * var⁺) / (sqrt(ρ⁻) + sqrt(ρ⁺))

function wavespeed(
    model::ThreeDimensionalDryCompressibleEulerWithTotalEnergy,
    n⁻,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    eos = model.equation_of_state
    parameters = model.parameters
    ρ = state.ρ
    ρu = state.ρu

    u = ρu / ρ
    u_norm = abs(dot(n⁻, u))
    return u_norm + calc_sound_speed(eos, state, aux, parameters)
end