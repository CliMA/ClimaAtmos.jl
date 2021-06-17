module Aleph

export function_documentation_template

"""
function_documentation_template(a; informative_keyword = "yes")

# Description
The goal of this function is to provide a template for all functions used
in Aleph.jl

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