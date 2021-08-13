abstract type AbstractDomain end

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