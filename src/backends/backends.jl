abstract type AbstractBackend end

Base.@kwdef struct DiscontinuousGalerkinBackend{𝒜} <: AbstractBackend
    numerics::𝒜
end

Base.@kwdef struct CoreBackend{𝒜} <: AbstractBackend
    numerics::𝒜
end
