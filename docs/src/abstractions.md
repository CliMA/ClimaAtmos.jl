# Introduction
```Aleph.jl``` supports the following numerics backends 
1. [ClimateMachine](@ref sec:climate_machine) is a [Discontinuous Galerkin](https://github.com/CliMA/ClimateMachine.jl) backend
2. [ClimateMachineCore](@ref sec:climate_machine_core) is a [Continuous Galerkin](https://github.com/CliMA/ClimateMachineCore.jl) backend *(still under development)*

The code is structured around a few human-centered abstractions that allow for REPL-driven development of simulations. This allows the user to build progressively more complicated simulation setups.

## Parameter Sets
```Aleph.jl``` natively supports parameters sets as Julia NamedTuples, but the plan is to support more sophisticated solution in the future that are useful for data assimilation purposes.

## Grids
Computational grids in ```Aleph.jl``` are abstract objects that contain information about the shape and discretization type chosen. 

## Initial Conditions 
Aleph.jl initial conditions are based on Julia's NamedTuple structures. This allows for a functional way of setting up and testing initial conditions.

## Models
ModelSetups in ```Aleph.jl``` aim to define a base set of partial differential equations that can be modified with a few bells and whistles. As a consequence an ```Aleph.jl``` ModelSetup requires the specification of an equation set together with boundary conditions.

For example, a struct may be a placeholder for the following Euler Equations

```math
    \begin{align}
    \partial_t \rho + \nabla \cdot (\rho \vec{u})  &= 0 
    \label{eq:continuity}
    \\
    \partial_t (\rho \vec{u}) + \nabla \cdot (\rho \vec{u} \otimes \vec{u} + p I)  &= 0 
    \label{eq:momentum}
    \\
        \partial_t (\rho e) + \nabla \cdot ( \vec{u} [\rho e + p])  &= 0 
    \label{eq:energy}
    \\
    \rho e -  \frac{1}{2} \rho \vec{u} \cdot \vec{u} - \rho \phi  &= p
    \label{eq:pressure}
    \end{align}
```
where the continuity equation is (\ref{eq:continuity}), the momentum equation is (\ref{eq:momentum}), the energy equation is (\ref{eq:energy}), and the pressure equation is (\ref{eq:pressure}).

## Diagnostics
```Aleph.jl``` provides a few callback structures for monitoring and simulation diagnostics purposes. Examples include the monitoring of CFL numbers and VTK or JLD2 output.

## Simulations
Simulations in ```Aleph.jl``` serve as a master structure that orchestrates the initialization and evolution of model equations.