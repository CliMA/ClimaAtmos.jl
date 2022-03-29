module Domains

using IntervalSets
using Printf
using ClimaCore:
    ClimaCore, Geometry, Meshes, Topologies, Spaces, Fields, Hypsography

export make_function_space,
    AbstractDomain,
    AbstractVerticalDomain,
    AbstractHybridDomain,
    Column,
    HybridPlane,
    HybridBox,
    SphericalShell,
    WarpedSurface,
    CanonicalSurface

import ClimaCore.Meshes:
    StretchingRule,
    Uniform,
    ExponentialStretching,
    GeneralizedExponentialStretching

import ClimaCore.Hypsography: TerrainAdaption, LinearAdaption

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

include("topography.jl")
include("domain.jl")
include("make_function_space.jl")

Base.eltype(::AbstractDomain{FT}) where {FT} = FT

end # module
