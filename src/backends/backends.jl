module Backends

import ..Interface: AbstractModel, AbstractDomain, Rectangle, PeriodicRectangle, ClimaCoreBackend, SingleColumn

@info "error comes from ClimaCore"
using UnPack
using ClimaCore

using OrdinaryDiffEq: ODEProblem, solve, SSPRK33

using ClimaCore.Spaces
using ClimaCore.Geometry
import ClimaCore: Fields, Domains, Topologies, Meshes, Spaces
import ClimaCore.Operators
import ClimaCore.Geometry
import ClimaCore:
    Fields,
    Domains,
    Topologies,
    Meshes,
    DataLayouts,
    Operators,
    Geometry,
    Spaces

# include
include("climacore/function_spaces.jl")
include("climacore/initial_conditions.jl")
# include("climacore/ode_problems.jl")
# include("climacore/tendencies.jl")

end # end of module