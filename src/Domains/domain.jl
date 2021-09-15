"""
    struct Column{FT} <: AbstractVerticalDomain
        
A column domain with `zlim` (Ordered Tuple) the domain extents,
and `nelements` (Integer) the number of cells in the domain. 
"""
struct Column{FT} <: AbstractVerticalDomain{FT}
    zlim::Tuple{FT, FT}
    nelements::Int32
end

"""
    Column([FT=Float64]; zlim, nelements)

Creates a column domain of type `FT`,
with extents zlim[1] < zlim[2] and `nelements` cells. 

Example:
Generate a Column{Float64} with extents (0,1) and 10 elements.

```julia-repl
julia> using ClimaAtmos.Domains
julia> z_domain = Column(Float64, 
                            zlim = (0,1), 
                            nelements = 10)
```
"""
function Column(FT::DataType = Float64; zlim, nelements)
    @assert zlim[1] < zlim[2]
    return Column{FT}(zlim, nelements)
end

"""
    Plane <: AbstractHorizontalDomain

A two-dimensional specialisation of an `AbstractHorizontalDomain`. 
An x-y plane with extents `xlim` (Ordered Tuple) in the x-direction,
extents `ylim` (Ordered Tuple) in the y-direction, `nelements` cells 
(Tuple of integers for x, y) directions, order of polynomial for 
spectral discretisation `npolynomial`, and description of periodicity in
x, y directions `periodic` (Tuple of Booleans). 

"""
struct Plane{FT} <: AbstractHorizontalDomain{FT}
    xlim::Tuple{FT, FT}
    ylim::Tuple{FT, FT}
    nelements::Tuple{Integer, Integer}
    npolynomial::Integer
    periodic::Tuple{Bool, Bool}
end

"""
    Plane([FT=Float64];
          xlim, 
          ylim, 
          nelements,
          npolynomial,
          periodic) 
    
Creates an xy plane bounded by keyword argument values of `xlim`, `ylim`, 
with `nelements` elements (which may differ in each direction) 
of polynomial order `npolynomial`, and periodicity specified by `periodic`
    
NOTE: This definition of `Plane` currently supports Spectral Discretizations 
only. Updates will contain support for spectral-element (`SpectralPlane`) and 
spectral-element + finite difference (`HybridPlane`) configurations.

Example: 
Generate a `Plane{Float64}` object with extents [0,π] × [0,π], contains 5 and 10 elements in the x and y directions respectively, 
with polynomial order 5, and is non-periodic in the y-direction
```julia-repl
julia> using ClimaAtmos.Domains
julia> xy_plane = Plane(Float64, xlim = (0,π), ylim = (0,π) , nelements=(5,10), npolynomial=5, periodic = (true,false))
```
"""
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
    PeriodicPlane([FT= Float64];
        xlim,
        ylim,
        nelements,
        npolynomial)
    
Creates an xy plane bounded by keyword argument values of `xlim`, `ylim`, 
with `nelements` elements (which may differ in each direction) 
of polynomial order `npolynomial`. Assumes domain is periodic in both x and 
y directions. Special case of `Plane`.
    
NOTE: This definition of `Plane` currently supports Spectral Discretizations 
only. Updates will contain support for spectral-element (`SpectralPlane`) and 
spectral-element + finite difference (`HybridPlane`) configurations.

Example: 
Generate a `Plane{Float64}` object with extents [0,π] × [0,π], contains 5 and 10 elements in the x and y directions respectively, 
with polynomial order 5, and is doubly-periodic
```julia-repl
julia> using ClimaAtmos.Domains
julia> xy_periodic_plane = PeriodicPlane(Float64, xlim = (0,π), ylim = (0,π) , nelements=(5,10), npolynomial=5)
# This is the same as 
julia> xy_periodic_plane_2 = Plane(Float64, xlim = (0,π), ylim = (0,π) , nelements=(5,10), npolynomial=5, periodic=(true,true))
```
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

"""
    struct HybridPlane{FT} <: AbstractHybridDomain 
"""
struct HybridPlane{FT} <: AbstractHybridDomain{FT}
    xlim::Tuple{FT, FT}
    zlim::Tuple{FT, FT}
    nelements::Tuple{Integer, Integer}
    npolynomial::Integer
end

"""
    HybridPlane([FT=Float64]; xlim, zlim, nelements, npolynomial)

Creates an 1D horizontal and 1D vertical hybrid domain with `xlim` the
horizontal domain extents, `zlim` the vertical domain extents,
`helement` the number of elements in the horizontal, `velement` the
number of cells in the vertical, and `npolynomial` the polynomial order
in the horizontal.
```
"""
function HybridPlane(FT::DataType = Float64; xlim, zlim, nelements, npolynomial)
    @assert xlim[1] < xlim[2]
    @assert zlim[1] < zlim[2]
    return HybridPlane{FT}(xlim, zlim, nelements, npolynomial)
end

Base.ndims(::Column) = 1

Base.ndims(::Plane) = 2

Base.ndims(::HybridPlane) = 2

Base.length(domain::Column) = domain.zlim[2] - domain.zlim[1]

Base.size(domain::Column) = length(domain)

Base.size(domain::Plane) =
    (domain.xlim[2] - domain.xlim[1], domain.ylim[2] - domain.ylim[1])

Base.size(domain::HybridPlane) =
    (domain.xlim[2] - domain.xlim[1], domain.zlim[2] - domain.zlim[1])

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

function Base.show(io::IO, domain::HybridPlane)
    minx = domain.xlim[1]
    maxx = domain.xlim[2]
    minz = domain.zlim[1]
    maxz = domain.zlim[2]
    print("Domain set-up:\n\tHorizontal and vertical hybrid plane:\t")
    printstyled(io, "[", color = 226)
    astring = @sprintf("%0.1f", minx)
    bstring = @sprintf("%0.1f", maxx)
    printstyled(astring, ", ", bstring, color = 7)
    printstyled(io, ")", color = 226)
    printstyled(io, " × [", color = 226)
    astring = @sprintf("%0.1f", minz)
    bstring = @sprintf("%0.1f", maxz)
    printstyled(astring, ", ", bstring, color = 7)
    printstyled(io, "]", color = 226)
    @printf(
        "\n\thorz and vert elem:\t\t(%d, %d)",
        domain.nelements[1],
        domain.nelements[2]
    )
    @printf("\n\tpoly order:\t\t%d\n", domain.npolynomial)
end
