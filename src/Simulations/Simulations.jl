module Simulations

using DiffEqBase
using UnPack: @unpack

using ClimaAtmos.Models:
    AbstractModel, default_initial_conditions, make_ode_function
using ClimaCore: Fields

import DiffEqBase: step!

"""
    AbstractSimulation
"""
abstract type AbstractSimulation end

include("simulation.jl")

export Simulation
export step!
export run!
export set!

end # module
