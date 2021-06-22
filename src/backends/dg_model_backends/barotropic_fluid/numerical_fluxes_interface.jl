struct CentralVolumeFlux <: NumericalFluxFirstOrder end
struct KGVolumeFlux <: NumericalFluxFirstOrder end

function numerical_volume_fluctuation_flux_first_order!(
    ::NumericalFluxFirstOrder,
    balance_law::ThreeDimensionalCompressibleEulerWithBarotropicFluid,
    D::Grad,
    state_1::Vars,
    aux_1::Vars,
    state_2::Vars,
    aux_2::Vars,
)
end

function numerical_volume_conservative_flux_first_order!(
    ::CentralVolumeFlux,
    m::ThreeDimensionalCompressibleEulerWithBarotropicFluid,
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
    balance_law::ThreeDimensionalCompressibleEulerWithBarotropicFluid,
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
    ρθ_1 = state_1.ρθ
    u_1 = ρu_1 / ρ_1
    θ_1 = ρθ_1 / ρ_1
    p_1 = calc_pressure(eos, state_1, aux_1, parameters)

    ρ_2 = state_2.ρ
    ρu_2 = state_2.ρu
    ρθ_2 = state_2.ρθ
    u_2 = ρu_2 / ρ_2
    θ_2 = ρθ_2 / ρ_2
    p_2 = calc_pressure(eos, state_2, aux_2, parameters)

    ρ_avg = ave(ρ_1, ρ_2)
    u_avg = ave(u_1, u_2)
    θ_avg = ave(θ_1, θ_2)
    p_avg = ave(p_1, p_2)

    F.ρ  = ρ_avg * u_avg
    F.ρu = p_avg * I + ρ_avg * u_avg .* u_avg'
    F.ρθ = ρ_avg * u_avg * θ_avg
end

function numerical_flux_first_order!(
    ::Nothing, 
    ::ThreeDimensionalCompressibleEulerWithBarotropicFluid, 
    _...,
)
    return nothing
end

function numerical_flux_first_order!(
    ::RoeNumericalFlux,
    balance_law::ThreeDimensionalCompressibleEulerWithBarotropicFluid,
    fluxᵀn::Vars{S},
    n⁻::SVector,
    state⁻::Vars{S},
    aux⁻::Vars{A},
    state⁺::Vars{S},
    aux⁺::Vars{A},
    t,
    direction,
) where {S, A}
    numerical_flux_first_order!(
        CentralNumericalFluxFirstOrder(),
        balance_law,
        fluxᵀn,
        n⁻,
        state⁻,
        aux⁻,
        state⁺,
        aux⁺,
        t,
        direction,
    )
    eos = balance_law.equation_of_state
    parameters = balance_law.parameters 

    # - states
    ρ⁻ = state⁻.ρ
    ρu⁻ = state⁻.ρu
    ρθ⁻ = state⁻.ρθ

    # constructed states
    u⁻ = ρu⁻ / ρ⁻
    θ⁻ = ρθ⁻ / ρ⁻

    # in general thermodynamics
    p⁻ = calc_pressure(eos, state⁻, aux⁻, parameters)
    c⁻ = calc_sound_speed(eos, state⁻, aux⁻, parameters)

    # + states
    ρ⁺ = state⁺.ρ
    ρu⁺ = state⁺.ρu
    ρθ⁺ = state⁺.ρθ

    # constructed states
    u⁺ = ρu⁺ / ρ⁺
    θ⁺ = ρθ⁺ / ρ⁺

    # in general thermodynamics
    p⁺ = calc_pressure(eos, state⁺, aux⁺, parameters)
    c⁺ = calc_sound_speed(eos, state⁺, aux⁺, parameters)

    # construct roe averges
    ρ = sqrt(ρ⁻ * ρ⁺)
    u = roe_average(ρ⁻, ρ⁺, u⁻, u⁺)
    θ = roe_average(ρ⁻, ρ⁺, θ⁻, θ⁺)
    c = roe_average(ρ⁻, ρ⁺, c⁻, c⁺)

    # construct normal velocity
    uₙ = u' * n⁻

    # differences
    Δρ = ρ⁺ - ρ⁻
    Δp = p⁺ - p⁻
    Δu = u⁺ - u⁻
    Δρθ = ρθ⁺ - ρθ⁻
    Δuₙ = Δu' * n⁻

    # constructed values
    c⁻² = 1 / c^2
    w1 = abs(uₙ - c) * (Δp - ρ * c * Δuₙ) * 0.5 * c⁻²
    w2 = abs(uₙ + c) * (Δp + ρ * c * Δuₙ) * 0.5 * c⁻²
    w3 = abs(uₙ) * (Δρ - Δp * c⁻²)
    w4 = abs(uₙ) * ρ
    w5 = abs(uₙ) * (Δρθ - θ * Δp * c⁻²)

    # fluxes!!!
    fluxᵀn.ρ -= (w1 + w2 + w3) * 0.5
    fluxᵀn.ρu -=
        (
            w1 * (u - c * n⁻) +
            w2 * (u + c * n⁻) +
            w3 * u +
            w4 * (Δu - Δuₙ * n⁻)
        ) * 0.5
    fluxᵀn.ρθ -= ((w1 + w2) * θ + w5) * 0.5

    return nothing
end

function numerical_flux_first_order!(
    ::LMARSNumericalFlux,
    balance_law::ThreeDimensionalCompressibleEulerWithBarotropicFluid,
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
    ρθ⁻ = state_prognostic⁻.ρθ
    u⁻ = ρu⁻ / ρ⁻
    uᵀn⁻ = u⁻' * normal_vector
    p⁻ = calc_pressure(eos, state_prognostic⁻, state_auxiliary⁻, parameters)
    c⁻ = calc_sound_speed(eos, state_prognostic⁻, state_auxiliary⁻, parameters)

    ρ⁺ = state_prognostic⁺.ρ
    ρu⁺ = state_prognostic⁺.ρu
    ρθ⁺ = state_prognostic⁺.ρθ
    u⁺ = ρu⁺ / ρ⁺
    uᵀn⁺ = u⁺' * normal_vector
    p⁺ = calc_pressure(eos, state_prognostic⁺, state_auxiliary⁺, parameters)

    # Eqn (49), (50), β the tuning parameter
    β = FT(1)
    u_half = 1 / 2 * (uᵀn⁺ + uᵀn⁻) - β * 1 / (ρ⁻ + ρ⁺) / c⁻ * (p⁺ - p⁻)
    p_half = 1 / 2 * (p⁺ + p⁻) - β * ((ρ⁻ + ρ⁺) * c⁻) / 4 * (uᵀn⁺ - uᵀn⁻)

    # Eqn (46), (47)
    ρ_b = u_half > FT(0) ? ρ⁻ : ρ⁺
    ρu_b = u_half > FT(0) ? ρu⁻ : ρu⁺
    ρθ_b = u_half > FT(0) ? ρθ⁻ : ρθ⁺

    # Update fluxes Eqn (18)
    fluxᵀn.ρ = ρ_b * u_half
    fluxᵀn.ρu = ρu_b * u_half .+ p_half * normal_vector
    fluxᵀn.ρθ = ρθ_b * u_half
end

function numerical_flux_second_order!(
    ::Nothing, 
    ::ThreeDimensionalCompressibleEulerWithBarotropicFluid, 
    _...,
) 
    return nothing
end

# utils
roe_average(ρ⁻, ρ⁺, var⁻, var⁺) =
    (sqrt(ρ⁻) * var⁻ + sqrt(ρ⁺) * var⁺) / (sqrt(ρ⁻) + sqrt(ρ⁺))