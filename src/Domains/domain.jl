"""
    Column <: AbstractVerticalDomain
"""
struct Column <: AbstractVerticalDomain
    zlim::Interval
    nelements::Int
end

# outer constructor
function Column(; zlim, nelements)
    @assert zlim.left < zlim.right
    return Column(zlim, nelements)
end

"""
    Plane <: AbstractHorizontalDomain
"""
struct Plane <: AbstractHorizontalDomain
    xlim::Interval  
    ylim::Interval
    nelements::Tuple{Int,Int}
    npolynomial::Int
    periodic::Tuple{Bool,Bool}
end

# outer constructor
function Plane(; xlim, ylim, nelements, npolynomial, periodic)
    @assert xlim.left < xlim.right
    @assert ylim.left < ylim.right
    return Plane(xlim, ylim, nelements, npolynomial, periodic)
end

"""
    PeriodicPlane
"""
function PeriodicPlane(; xlim, ylim, nelements, npolynomial)
    @assert xlim.left < xlim.right
    @assert ylim.left < ylim.right
    
    return Plane(
        xlim = xlim, 
        ylim = ylim, 
        nelements = nelements, 
        npolynomial = npolynomial, 
        periodic = (true, true),
    )
end