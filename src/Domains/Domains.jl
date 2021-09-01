module Domains

using ClimaCore
using Printf

import ClimaCore: Meshes, Spaces, Topologies

import IntervalSets: Interval

"""
    AbstractDomain
"""
abstract type AbstractDomain end

"""
    AbstractHorizontalDomain
"""
abstract type AbstractHorizontalDomain{FT} <: AbstractDomain end

"""
    AbstractVerticalDomain
"""
abstract type AbstractVerticalDomain{FT} <: AbstractDomain end

include("domain.jl")
include("make_function_space.jl")

export AbstractDomain
export AbstractHorizontalDomain
export AbstractVerticalDomain
export Column
export PeriodicRectangle
export Rectangle

export make_function_space

end # module
