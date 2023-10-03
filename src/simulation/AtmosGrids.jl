# AbstractAtmosGrids.jl
#
# In this file we define an abstract type AbstractAtmosGrid which describes the
# computational domain over which the simulation is run. We provide definitions of concrete
# AbstractAtmosGrids commonly used in simulations.

module Grid

import ClimaComms
import ClimaCore: Domains, Geometry, Meshes, Spaces, Topologies

"""
    AbstractAtmosGrid


Computational domain over which the simulation is run. Thin wrapper around two
`ClimaCore.Space`s, one the cell centers and one for the cell faces, and possibly other
general information and parameters related to the given grid.

"""
abstract type AbstractAtmosGrid end

"""
    float_type(grid)


Return the floating-point type associated with the given `grid`.

"""
float_type(grid::AbstractAtmosGrid) = Spaces.undertype(grid.center_space)

"""
    context(grid)


Return the context of the computational device associated with the given `grid`.

"""
context(grid::AbstractAtmosGrid) = ClimaComms.context(grid.center_space)

include("atmos_grids.jl")

end
