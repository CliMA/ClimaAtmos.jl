abstract type AbstractPhysics end

Base.@kwdef struct ModelPhysics{𝒜,ℬ,𝒞} <: AbstractPhysics
    equation_of_state::𝒜 = nothing
    ref_state::ℬ = nothing
    sources::𝒞 = nothing
end