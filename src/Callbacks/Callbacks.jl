module Callbacks

using DiffEqCallbacks
using UnPack
using JLD2

using ClimaAtmos.Models: AbstractModel

export generate_callback, AbstractCallback, JLD2Output

"""
    AbstractCallback
    Abstract type for callback definitions. 
"""
abstract type AbstractCallback end

function generate_callback(::AbstractCallback) end

include("callback.jl")

end # end module
