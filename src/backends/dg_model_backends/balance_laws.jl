Base.@kwdef struct ThreeDimensionalCompressibleEulerWithBarotropicFluid{𝒜,ℬ,𝒞,𝒟,ℰ} <: BalanceLaw 
    equation_of_state::𝒜
    sources::ℬ
    boundary_conditions::𝒞
    initial_conditions::𝒟
    parameters::ℰ
end

Base.@kwdef struct ThreeDimensionalDryCompressibleEulerWithTotalEnergy{𝒜,ℬ,𝒞,𝒟} <: BalanceLaw 
    physics::𝒜
    boundary_conditions::ℬ
    initial_conditions::𝒞
    parameters::𝒟
end

Base.@kwdef struct ThreeDimensionalMoistCompressibleEulerWithTotalEnergy{𝒜,ℬ,𝒞,𝒟} <: BalanceLaw 
    physics::𝒜
    boundary_conditions::ℬ
    initial_conditions::𝒞
    parameters::𝒟
end

function create_balance_law(model::ModelSetup{𝒜}) where 
    {𝒜 <: ThreeDimensionalEuler{Density, BarotropicFluid, Compressible}}

    return ThreeDimensionalCompressibleEulerWithBarotropicFluid(
        equation_of_state = model.equations.equation_of_state,
        sources = model.equations.sources,
        boundary_conditions = model.boundary_conditions,
        initial_conditions = model.initial_conditions,  
        parameters = model.parameters, 
    )
end

function create_balance_law(model::ModelSetup{𝒜}) where 
    {𝒜 <: ThreeDimensionalEuler{TotalEnergy, DryIdealGas, Compressible}}

    return ThreeDimensionalDryCompressibleEulerWithTotalEnergy(
        physics = model.equations.physics,
        boundary_conditions = model.boundary_conditions,
        initial_conditions = model.initial_conditions,  
        parameters = model.parameters, 
    )
end

function create_balance_law(model::ModelSetup{𝒜}) where 
    {𝒜 <: ThreeDimensionalEuler{TotalEnergy, MoistIdealGas, Compressible}}

    return ThreeDimensionalMoistCompressibleEulerWithTotalEnergy(
        physics = model.equations.physics,
        boundary_conditions = model.boundary_conditions,
        initial_conditions = model.initial_conditions,  
        parameters = model.parameters,  
    )
end

function create_numerical_flux(surface_flux)
    if surface_flux == :lmars
        return LMARSNumericalFlux()
    elseif surface_flux == :roe
        return RoeNumericalFlux()
    elseif surface_flux == :refanov 
        return RefanovFlux()
    else
        return nothing
    end
end