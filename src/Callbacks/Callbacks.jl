module Callbacks

using DiffEqCallbacks
using OrdinaryDiffEq: set_proposed_dt!
using UnPack
using JLD2

using ClimaCore

using ClimaAtmos.Models: AbstractModel

export generate_callback, AbstractCallback, JLD2Output, CFLAdaptive

"""
    AbstractCallback
    Abstract type for callback definitions. 
"""
abstract type AbstractCallback end

function generate_callback(::AbstractCallback) end

include("callback.jl")

end # end module
