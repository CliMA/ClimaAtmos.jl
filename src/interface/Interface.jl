module Interface

using IntervalSets

# Exports
export ClimaCoreBackend, DiscontinuousGalerkinBackend

# Backends supported
abstract type AbstractBackend end
struct ClimaCoreBackend <: AbstractBackend end
Base.@kwdef struct DiscontinuousGalerkinBackend{𝒜} <: AbstractBackend
    numerics::𝒜
end

# includes
# include("simulations.jl")

# WIP includes
include("domains.jl")
include("models.jl")
include("timesteppers.jl")
include("boundary_conditions.jl")
include("simulations.jl")

end # end of module
