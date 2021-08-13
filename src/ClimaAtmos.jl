module ClimaAtmos

using Pkg, LinearAlgebra

# convenience functions
export @boilerplate

# example 
export function_documentation_template

# include other modules
include("utils/Utils.jl")
include("interface/Interface.jl")
include("backends/backends.jl")


# convenience functions: definitions
"""
macro boilerplate()

# Description 
A convenience function that includes all the usual needed for a script
"""
macro boilerplate()
    boiler_block = :( 
        using ClimaAtmos.Utils;
        using ClimaAtmos.Interface;
        using ClimaAtmos.Backends;
        using IntervalSets, UnPack;
        using OrdinaryDiffEq: SSPRK33;

        # explicit imports from Interface
        import ClimaAtmos.Interface: PeriodicRectangle, SingleColumn;
        import ClimaAtmos.Interface: BarotropicFluidModel, HydrostaticModel;
        import ClimaAtmos.Interface: TimeStepper;
        import ClimaAtmos.Interface: DirichletBC, Simulation;
        import ClimaAtmos.Backends: create_ode_problem, evolve;

        # imports from Clima Core
        using ClimaCore.Geometry;
        import ClimaCore: Fields, Domains, Topologies, Meshes, Spaces
        )
    return boiler_block
end

# example: definition

"""
function_documentation_template(a; informative_keyword = "yes")

# Description
The goal of this function is to provide a template for all functions used
in ClimaAtmos.jl

# Arguments
- `a` | Type: Any | An example function argument 

# Keyword Arguments
- `informative_keyword`. Default: "yes"

# Return 
- π : a number

"""
function function_documentation_template(a; informative_keyword = "yes")
    println("This function is just an example. The input was ", a)
    println("The informative keyword is ", informative_keyword)
    return pi
end

end # module