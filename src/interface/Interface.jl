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
include("WIP_domains.jl")
include("WIP_models.jl")
include("WIP_timesteppers.jl")
include("WIP_boundary_conditions.jl")
include("WIP_simulations.jl")
include("WIP_physics.jl")

end # end of module