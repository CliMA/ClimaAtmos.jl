abstract type AbstractDomain end

struct Rectangle{FT} <: AbstractDomain
    xlim::Tuple{FT,FT}  
    ylim::Tuple{FT,FT}
    nelements::Tuple{Int,Int}
    npolynomial::Tuple{Int,Int}
    periodic::Tuple{Bool,Bool}
end

function Rectangle(; xlim, ylim, nelements, npolynomial, periodic)
    @assert xlim[1] < xlim[2]
    @assert ylim[1] < ylim[2]
    return Rectangle(xlim, ylim, nelements, npolynomial, periodic)
end

function Torus2D(; xlim, ylim, nelements, npolynomial)
    @assert xlim[1] < xlim[2]
    @assert ylim[1] < ylim[2]
    periodic = (true, true)
    return Rectangle(
        xlim = xlim, 
        ylim = ylim, 
        nelements = nelements, 
        npolynomial = npolynomial, 
        periodic = periodic,
    )
end

struct Box{FT} <: AbstractDomain
    xlim::Tuple{FT,FT}  
    ylim::Tuple{FT,FT}
    zlim::Tuple{FT,FT}
    nelements::Tuple{Int,Int,Int}
    npolynomial::Tuple{Int,Int,Int}
    periodic::Tuple{Bool,Bool,Bool}
    staggering::Bool
end

function Box(; xlim, ylim, zlim, nelements, npolynomial, periodic, staggering)
    @assert xlim[1] < xlim[2]
    @assert ylim[1] < ylim[2]
    @assert zlim[1] < zlim[2]
    return Box(xlim, ylim, zlim, nelements, npolynomial, periodic, staggering)
end

function Torus3D(; xlim, ylim, zlim, nelements, npolynomial, staggering)
    periodic = (true, true, true)
    return Box(
        xlim = xlim, 
        ylim = ylim, 
        zlim = zlim, 
        nelements = nelements, 
        npolynomial = npolynomial, 
        periodic = periodic,
        staggering = staggering,
    )
end