module Domains

using IntervalSets
using Printf
using ClimaCore: ClimaCore, Geometry, Meshes, Topologies, Spaces

export make_function_space,
    AbstractDomain,
    AbstractVerticalDomain,
    AbstractHybridDomain,
    Column,
    HybridPlane

"""
    make_function_space(domain)

Convert an `AbstractDomain` into a `ClimaCore.Spaces.AbstactSpace`.
"""
function make_function_space end

"""
Supertype for all domains.
"""
abstract type AbstractDomain{FT} end

abstract type AbstractVerticalDomain{FT} <: AbstractDomain{FT} end
abstract type AbstractHybridDomain{FT} <: AbstractDomain{FT} end

include("domain.jl")
include("make_function_space.jl")

Base.eltype(::AbstractDomain{FT}) where {FT} = FT

end # module
