Base.@kwdef struct ThreeDimensionalCompressibleEulerWithBarotropicFluid{𝒜,ℬ,𝒞,𝒟} <: BalanceLaw 
    physics::𝒜
    boundary_conditions::ℬ
    initial_conditions::𝒞
    parameters::𝒟
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
        physics = model.equations.physics,
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