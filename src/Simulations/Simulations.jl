module Simulations

using DiffEqBase
using UnPack: @unpack

using ClimaCore: Fields

import ClimaAtmos.Models: AbstractModel
import ClimaAtmos.Models: make_initial_conditions, make_ode_function
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
