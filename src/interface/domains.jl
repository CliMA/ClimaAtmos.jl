abstract type AbstractDomain end
abstract type AbstractVerticalDiscretization end
Base.@kwdef struct VerticalFiniteDifference{T} <: AbstractVerticalDiscretization 
    warping::T = nothing
end
Base.@kwdef struct VerticalDiscontinousGalerkin{T} <: AbstractVerticalDiscretization
    vpolynomial::Int
    warping::T = nothing
end

struct Rectangle <: AbstractDomain
    xlim::Interval  
    ylim::Interval
    nelements::Tuple{Int,Int}
    npolynomial::Int
    periodic::Tuple{Bool,Bool}
end

function Rectangle(; xlim, ylim, nelements, npolynomial, periodic)
    @assert xlim.left < xlim.right
    @assert ylim.left < ylim.right
    return Rectangle(xlim, ylim, nelements, npolynomial, periodic)
end

function PeriodicRectangle(; xlim, ylim, nelements, npolynomial)
    @assert xlim.left < xlim.right
    @assert ylim.left < ylim.right
    return Rectangle(
        xlim = xlim, 
        ylim = ylim, 
        nelements = nelements, 
        npolynomial = npolynomial, 
        periodic = (true, true),
    )
end

struct SingleColumn <: AbstractDomain
    zlim::Interval
    nelements::Int
end

function SingleColumn(; zlim, nelements)
    @assert zlim.left < zlim.right
    return SingleColumn(zlim, nelements)
end

struct SphericalShell{A,B} <: AbstractDomain
    radius::A
    height::A
    nelements::NamedTuple{(:horizontal, :vertical), Tuple{Int, Int}}
    npolynomial::NamedTuple{(:horizontal,), Tuple{Int}}
    vertical_discretization::B # default to FD 
end

function SphericalShell(; radius, height, nelements, npolynomial, vertical_discretization = VerticalFiniteDifference())
    if nelements isa Tuple
        nelements = (horizontal=nelements[1],vertical=nelements[2])
    elseif nelements isa Int
        nelements = (horizontal=nelements,vertical=nelements)
    end
    if npolynomial isa Int
        npolynomial = (horizontal = npolynomial, )
    end
    return SphericalShell(radius, height, nelements, npolynomial, vertical_discretization)
end

#= old domain interface for DG
struct SphericalShell{T} <: AbstractDomain
    radius::T
    height::T
end

function SphericalShell(; radius, height)
    @assert radius > 0 && height > 0
    return SphericalShell(radius, height)
end

Base.@kwdef struct DiscretizedDomain{𝒜, ℬ} <: AbstractDomain
    domain::𝒜
    discretization::ℬ
end

abstract type AbstractGrid end

Base.@kwdef struct SpectralElementGrid{𝒜,ℬ,𝒞} <: AbstractGrid 
    elements::𝒜
    polynomial_order::ℬ
    warping::𝒞
end

function SpectralElementGrid(; elements, polynomial_order)
    return SpectralElementGrid(elements, polynomial_order, nothing)
end
=#