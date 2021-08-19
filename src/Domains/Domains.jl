module Domains

using ClimaCore
using IntervalSets

import ClimaCore:
    Meshes,
    Spaces,
    Topologies

"""
    AbstractDomain
"""
abstract type AbstractDomain end

"""
    AbstractHorizontalDomain
"""
abstract type AbstractHorizontalDomain <: AbstractDomain end

"""
    AbstractVerticalDomain
"""    
abstract type AbstractVerticalDomain <: AbstractDomain end

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