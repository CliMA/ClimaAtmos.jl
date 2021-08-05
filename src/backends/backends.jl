abstract type AbstractBackend end

struct ClimaCoreBackend <: AbstractBackend end

Base.@kwdef struct DiscontinuousGalerkinBackend{𝒜} <: AbstractBackend
    numerics::𝒜
end