"""
    Column{FT} <: AbstractVerticalDomain
"""
struct Column{FT} <: AbstractVerticalDomain{FT}
    zlim::Tuple{FT, FT}
    nelements::Int32
end

function Column(FT::DataType = Float64; zlim, nelements)
    @assert zlim[1] < zlim[2]
    return Column{FT}(zlim, nelements)
end

"""
    Plane <: AbstractHorizontalDomain
"""
struct Plane{FT} <: AbstractHorizontalDomain{FT}
    xlim::Tuple{FT, FT}
    ylim::Tuple{FT, FT}
    nelements::Tuple{Integer, Integer}
    npolynomial::Integer
    periodic::Tuple{Bool, Bool}
end

function Plane(
    FT::DataType = Float64;
    xlim,
    ylim,
    nelements,
    npolynomial,
    periodic,
)
    @assert xlim[1] < xlim[2]
    @assert ylim[1] < ylim[2]
    return Plane{FT}(xlim, ylim, nelements, npolynomial, periodic)
end

"""
    PeriodicPlane
"""
function PeriodicPlane(
    FT::DataType = Float64;
    xlim,
    ylim,
    nelements,
    npolynomial,
)
    @assert xlim[1] < xlim[2]
    @assert ylim[1] < ylim[2]

    return Plane(
        FT,
        xlim = xlim,
        ylim = ylim,
        nelements = nelements,
        npolynomial = npolynomial,
        periodic = (true, true),
    )
end

Base.ndims(::Column) = 1
Base.ndims(::Plane) = 2

Base.length(domain::Column) = domain.zlim[2] - domain.zlim[1]

Base.size(domain::Column) = length(domain)
Base.size(domain::Plane) =
    (domain.xlim[2] - domain.xlim[1], domain.ylim[2] - domain.ylim[1])

function Base.show(io::IO, domain::Column)
    min = domain.zlim[1]
    max = domain.zlim[2]
    print("Domain set-up:\n\tSingle colume z-range:\t")
    printstyled(io, "[", color = 226)
    astring = @sprintf("%0.1f", min)
    bstring = @sprintf("%0.1f", max)
    printstyled(astring, ", ", bstring, color = 7)
    printstyled(io, "]", color = 226)
    @printf("\n\tvert elem:\t\t%d\n", domain.nelements)
end

function Base.show(io::IO, domain::Plane)
    minx = domain.xlim[1]
    maxx = domain.xlim[2]
    miny = domain.ylim[1]
    maxy = domain.ylim[2]
    print("Domain set-up:\n\tHorizontal plane:\t")
    printstyled(io, "[", color = 226)
    astring = @sprintf("%0.1f", minx)
    bstring = @sprintf("%0.1f", maxx)
    printstyled(astring, ", ", bstring, color = 7)
    domain.periodic[1] ? printstyled(io, ")", color = 226) :
    printstyled(io, "]", color = 226)
    printstyled(io, " × [", color = 226)
    astring = @sprintf("%0.1f", miny)
    bstring = @sprintf("%0.1f", maxy)
    printstyled(astring, ", ", bstring, color = 7)
    domain.periodic[2] ? printstyled(io, ")", color = 226) :
    printstyled(io, "]", color = 226)
    @printf(
        "\n\thorz elem:\t\t(%d, %d)",
        domain.nelements[1],
        domain.nelements[2]
    )
    @printf("\n\tpoly order:\t\t%d\n", domain.npolynomial)
end
