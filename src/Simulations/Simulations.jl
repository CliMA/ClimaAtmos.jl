module Simulations

import OrdinaryDiffEq
const ODE = OrdinaryDiffEq
using OrdinaryDiffEq: CallbackSet

using JLD2
using Printf
using UnPack
using ClimaCore: Fields
using ..Models, ..Callbacks, ..Domains

export step!,
    run!, set!, get_spaces, Simulation, AbstractRestart, NoRestart, Restart, StateDSS

"""
Supertype for all restart modes.
"""
abstract type AbstractRestart end

include("simulation.jl")
include("restart.jl")

end # module
