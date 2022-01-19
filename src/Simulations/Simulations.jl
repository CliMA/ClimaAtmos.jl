module Simulations

import DiffEqBase # avoid namespace conflict with step!

using JLD2
using Printf
using UnPack
using ClimaCore: Fields
using ..Models, ..Callbacks

export step!,
    run!, set!, get_spaces, Simulation, AbstractRestart, NoRestart, Restart

"""
Supertype for all restart modes.
"""
abstract type AbstractRestart end

include("simulation.jl")
include("restart.jl")

end # module
