abstract type AbstractDiscretizedDomain end
abstract type AbstractGrid end

"""
    DiscretizedDomain
"""
Base.@kwdef struct DiscretizedDomain{𝒜, ℬ} <: AbstractDiscretizedDomain
    domain::𝒜
    grid::ℬ
end

"""
    Grids
"""
Base.@kwdef struct SpectralElementGrid{𝒜,ℬ,𝒞} <: AbstractGrid 
    elements::𝒜
    polynomial_order::ℬ
    warping::𝒞
end

function SpectralElementGrid(; elements, polynomial_order)
    return SpectralElementGrid(elements, polynomial_order, nothing)
end

Base.@kwdef struct StaggeredGrid{𝒜,ℬ} <: AbstractGrid 
    cells::𝒜
    stretching::ℬ
end

function StaggeredGrid(; cells)
    return StaggeredGrid(cells, nothing)
end