# Introduction
The ```ClimaAtmos.jl``` code is structured around a few human-centered abstractions that allow for REPL-driven development of simulations. This allows the user to build progressively more complicated simulation setups.

## Domains
Computational domains in ```ClimaAtmos.jl``` are abstract objects that contain information about the shape and discretization type chosen. 

## Boundary Conditions 
ClimaAtmos.jl boundary conditions handle different types of boundary conditions.

## Models
Models in ```ClimaAtmos.jl``` aim to define a base set of partial differential equations that can be modified with a few bells and whistles. 

## TimeSteppers
TimeSteppers in ```ClimaAtmos.jl``` serve as a master structure that orchestrates the all things time stepping.

## Simulations
Simulations in ```ClimaAtmos.jl``` serve as a master structure that orchestrates the initialization and evolution of model equations.

## Utils
Utils in ```ClimaAtmos.jl``` have some precooked function that are useful for benchmarking, testing, etc.
