function numerical_boundary_flux_first_order!(
    numerical_flux::NumericalFluxFirstOrder,
    ::DefaultBC,
    balance_law::ThreeDimensionalDryCompressibleEulerWithTotalEnergy,
    fluxᵀn::Vars{S},
    n̂::SVector,
    state⁻::Vars{S},
    aux⁻::Vars{A},
    state⁺::Vars{S},
    aux⁺::Vars{A},
    t,
    direction,
    state1⁻::Vars{S},
    aux1⁻::Vars{A},
) where {S, A}
    state⁺.ρ = state⁻.ρ
    state⁺.ρe = state⁻.ρe

    ρu⁻ = state⁻.ρu
    
    # project and reflect for impenetrable condition, but 
    # leave tangential component untouched
    state⁺.ρu = ρu⁻ - n̂ ⋅ ρu⁻ .* SVector(n̂) - n̂ ⋅ ρu⁻ .* SVector(n̂)
    numerical_flux_first_order!(
      numerical_flux,
      balance_law,
      fluxᵀn,
      n̂,
      state⁻,
      aux⁻,
      state⁺,
      aux⁺,
      t,
      direction,
    )
end

function numerical_boundary_flux_first_order!(
    numerical_flux::NumericalFluxFirstOrder,
    bctype::BulkFormulaTemperature,
    balance_law::ThreeDimensionalDryCompressibleEulerWithTotalEnergy,
    fluxᵀn::Vars{S},
    n̂::SVector,
    state⁻::Vars{S},
    aux⁻::Vars{A},
    state⁺::Vars{S},
    aux⁺::Vars{A},
    t,
    direction,
    state1⁻::Vars{S},
    aux1⁻::Vars{A},
) where {S, A}
    # Impenetrable free-slip condition to reflect and project momentum 
    # at the boundary
    numerical_boundary_flux_first_order!(
        numerical_flux,
        bctype::FreeSlip,
        balance_law,
        fluxᵀn,
        n̂,
        state⁻,
        aux⁻,
        state⁺,
        aux⁺,
        t,
        direction,
        state1⁻,
        aux1⁻,
    )
    
    # Apply bulk laws using the tangential velocity as energy flux
    ρ = state⁻.ρ
    ρu = state⁻.ρu
    eos = balance_law.equation_of_state
    parameters = balance_law.parameters

    # obtain surface fields from bcs
    ϕ = lat(aux⁻.x, aux⁻.y, aux⁻.z)
    Cₕ = bctype.drag_coef_temperature(parameters, ϕ)
    T_sfc = bctype.temperature(parameters, ϕ)

    # magnitude of tangential velocity (usually called speed)
    u = ρu / ρ
    speed_tangential = norm((I - n̂ ⊗ n̂) * u)
       
    # sensible heat flux
    cp = calc_cp(eos, state⁻, parameters)
    T = calc_air_temperature(eos, state⁻, aux⁻, parameters)
    H = ρ * Cₕ * speed_tangential * cp * (T - T_sfc)

    fluxᵀn.ρe = H
end

function numerical_boundary_flux_second_order!(
    ::Nothing, 
    a, 
    ::Union{ThreeDimensionalDryCompressibleEulerWithTotalEnergy,LinearThreeDimensionalDryCompressibleEulerWithTotalEnergy}, 
    _...
) 
    return nothing
end

function boundary_conditions(
    balance_law::Union{ThreeDimensionalDryCompressibleEulerWithTotalEnergy, LinearThreeDimensionalDryCompressibleEulerWithTotalEnergy}
) 
    return balance_law.boundary_conditions
end

function boundary_state!(_...)
    nothing
end