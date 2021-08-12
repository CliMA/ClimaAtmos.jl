module ClimaAtmos

using Pkg

# convenience functions
export @boilerplate, add_climate_machine, add_clima_core

# example 
export function_documentation_template

# convenience functions: definitions
"""
macro boilerplate()

# Description 
A convenience function that includes all the usual packages
"""
macro boilerplate()
    boiler_block = :( 
        using JLD2; 
        using Plots;)
    return boiler_block
end

"""
function add_climate_machine()

# Description
Grabs the particular ClimateMachine branch used in ClimaAtmos
"""
function add_climate_machine()
       Pkg.add(url = "https://github.com/CliMA/ClimateMachine.jl.git#tb/refactoring_ans_sphere")
end

"""
function add_climate_machine()

# Description
Grabs the particular ClimaCore branch used in ClimaAtmos
"""
function add_clima_core()
       Pkg.add(url = "https://github.com/CliMA/ClimaCore.jl.git")
       Pkg.add("DiffEqBase")
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