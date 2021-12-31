struct Column{FT} <: AbstractVerticalDomain{FT}
    zlim::Tuple{FT, FT}
    nelements::Int
end

"""
    Column([FT = Float64]; zlim, nelements)

Construct a domain of type `FT` that represents a column along the z-axis with
limits `zlim` (where `zlim[1] < zlim[2]`) and `nelements` elements. This domain
is not periodic.

# Example
```jldoctest; setup = :(using ClimaAtmos.Domains)
julia> Column(zlim = (0, 1), nelements = 10)
Domain set-up:
\tz-column:\t[0.0, 1.0]
\t# of elements:\t10
```
"""
function Column(::Type{FT} = Float64; zlim, nelements) where {FT}
    @assert zlim[1] < zlim[2]
    return Column{FT}(zlim, nelements)
end

struct HybridPlane{FT} <: AbstractHybridDomain{FT}
    xlim::Tuple{FT, FT}
    zlim::Tuple{FT, FT}
    nelements::Tuple{Int, Int}
    npolynomial::Int
    xperiodic::Bool
end

"""
    HybridPlane([FT = Float64]; xlim, zlim, nelements, npolynomial, xperiodic = true)

Construct a domain of type `FT` that represents an xz-plane with limits `xlim`
and `zlim` (where `xlim[1] < xlim[2]` and `zlim[1] < zlim[2]`), `nelements`
elements of polynomial order `npolynomial`, and x-axis periodicity `xperiodic`.
`nelements` must be a tuple with two values, with the first value corresponding
to the x-axis and the second corresponding to the z-axis. This domain is not
periodic along the z-axis.

# Example
```jldoctest; setup = :(using ClimaAtmos.Domains)
julia> HybridPlane(
            xlim = (0, π),
            zlim = (0, 1),
            nelements = (5, 10),
            npolynomial = 5,
            xperiodic = true,
        )
Domain set-up:
\txz-plane:\t[0.0, 3.1) × [0.0, 1.0]
\t# of elements:\t(5, 10)
\tpoly order:\t5
```
"""
function HybridPlane(
    ::Type{FT} = Float64;
    xlim,
    zlim,
    nelements,
    npolynomial,
    xperiodic = true,
) where {FT}
    @assert xlim[1] < xlim[2]
    @assert zlim[1] < zlim[2]
    return HybridPlane{FT}(xlim, zlim, nelements, npolynomial, xperiodic)
end

function Base.show(io::IO, domain::Column)
    print(io, "Domain set-up:\n\tz-column:\t")
    printstyled(io, "[", color = 226)
    printstyled(io, @sprintf("%#.2g, %#.2g", domain.zlim...), color = 7)
    printstyled(io, "]", color = 226)
    print(io, "\n\t# of elements:\t", domain.nelements)
end

function Base.show(io::IO, domain::HybridPlane)
    print(io, "Domain set-up:\n\txz-plane:\t")
    printstyled(io, "[", color = 226)
    printstyled(io, @sprintf("%#.2g, %#.2g", domain.xlim...), color = 7)
    printstyled(io, domain.xperiodic ? ")" : "]", " × [", color = 226)
    printstyled(io, @sprintf("%#.2g, %#.2g", domain.zlim...), color = 7)
    printstyled(io, "]", color = 226)
    @printf(io, "\n\t# of elements:\t(%d, %d)", domain.nelements...)
    print(io, "\n\tpoly order:\t", domain.npolynomial)
end
