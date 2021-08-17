abstract type AbstractPhysics end
abstract type AbstractEquationOfState end

# equation of state
struct BarotropicFluid <: AbstractEquationOfState end
struct DryIdealGas <: AbstractEquationOfState end
struct MoistIdealGas <: AbstractEquationOfState end

# coriolis force
struct DeepShellCoriolis <: AbstractPhysics end

# gravity
struct Gravity <: AbstractPhysics end

Base.@kwdef struct ModelPhysics{𝒜,ℬ,𝒞} 
    equation_of_state::𝒜 = nothing
    ref_state::ℬ = nothing
    sources::𝒞 = nothing
end