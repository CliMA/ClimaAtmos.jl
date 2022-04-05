module Callbacks

using DiffEqCallbacks
using JLD2
using UnPack
using ClimaCore: ClimaCore, Fields, Spaces
using ..Models

export generate_callback, AbstractCallback, JLD2Output, CFLAdaptive, StateDSS

"""
    generate_callback(callback; kwargs...)

Convert an `AbstractCallback` to a `SciMLBase.DECallback`.
"""
function generate_callback end

"""
Supertype for all callbacks.
"""
abstract type AbstractCallback end

include("callback.jl")

end # end module
