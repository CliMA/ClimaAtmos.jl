module Simulations

using DiffEqBase

import ClimaAtmos.Models: AbstractModel, make_initial_conditions, make_ode_function
import ClimaAtmos.Timesteppers: AbstractTimestepper

"""
    AbstractSimulation
"""
abstract type AbstractSimulation end

include("simulation.jl")

export AbstractSimulation
export Simulation

export init
export run

end # module