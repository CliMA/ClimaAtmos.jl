function numerical_boundary_flux_first_order!(
    numerical_flux::NumericalFluxFirstOrder,
    ::DefaultBC,
    balance_law::ThreeDimensionalCompressibleEulerWithBarotropicFluid,
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
    state⁺.ρθ = state⁻.ρθ
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

function numerical_boundary_flux_second_order!(
    ::Nothing, 
    a, 
    ::ThreeDimensionalCompressibleEulerWithBarotropicFluid, 
    _...
) 
    return nothing
end

function boundary_conditions(
    balance_law::ThreeDimensionalCompressibleEulerWithBarotropicFluid
) 
    return balance_law.boundary_conditions
end