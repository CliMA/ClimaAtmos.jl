abstract type AbstractGrid end

struct Rectangle <: AbstractGrid
    left::FT
    right::FT
    front::FT
    back::FT
    n1::Int
    n2::Int
    polynomial_order::Int
    x1periodic::Bool
    x2periodic::Bool
end