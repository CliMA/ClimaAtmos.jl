module Utils

# dependencies
using StaticArrays, LinearAlgebra

# sphere_utils 
export r̂ⁿᵒʳᵐ, ϕ̂ⁿᵒʳᵐ, λ̂ⁿᵒʳᵐ, r̂, ϕ̂, λ̂

# operations 
export ⊗, ⋅

# includes
include("operations.jl")
include("sphere_utils.jl")

end # end of module