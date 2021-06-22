abstract type AbstractBackend end

Base.@kwdef struct DiscontinuousGalerkinBackend{𝒜,ℬ} <: AbstractBackend
    grid::𝒜
    numerics::ℬ
end

Base.@kwdef struct CoreBackend{𝒜,ℬ} <: AbstractBackend
    grid::𝒜
    numerics::ℬ
end


