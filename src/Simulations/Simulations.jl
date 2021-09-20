module Simulations

using DiffEqBase
using JLD2
using UnPack: @unpack
using Printf

using ClimaAtmos.Callbacks
using ClimaAtmos.Models:
    AbstractModel, default_initial_conditions, make_ode_function
using ClimaCore: Fields

import DiffEqBase: step!

"""
    AbstractSimulation
"""
abstract type AbstractSimulation end

"""
    AbstractRestart
"""
abstract type AbstractRestart end

include("simulation.jl")
include("restart.jl")

export Simulation
export AbstractRestart, Restart
export step!
export run!
export set!

end # module
