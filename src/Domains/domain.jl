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
    topography::Union{Function,Nothing}
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
            topography = nothing,
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
    topography = nothing
) where {FT}
    @assert xlim[1] < xlim[2]
    @assert zlim[1] < zlim[2]
    return HybridPlane{FT}(xlim, zlim, nelements, npolynomial, xperiodic, topography)
end

struct HybridBox{FT} <: AbstractHybridDomain{FT}
    xlim::Tuple{FT, FT}
    ylim::Tuple{FT, FT}
    zlim::Tuple{FT, FT}
    nelements::Tuple{Int, Int, Int}
    npolynomial::Int
    xperiodic::Bool
    yperiodic::Bool
end

"""
    HybridBox([FT = Float64]; xlim, ylim, zlim, nelements, npolynomial, xperiodic = true, yperiodic = true)

Construct a domain of type `FT` that represents an xz-plane with limits `xlim` `ylim`
and `zlim` (where `xlim[1] < xlim[2]`,`ylim[1] < ylim[2]`, and `zlim[1] < zlim[2]`), `nelements`
elements of polynomial order `npolynomial`, x-axis periodicity `xperiodic`, and y-axis periodicity `yperiodic`.
`nelements` must be a tuple with two values, with the first value corresponding
to the x-axis, the second corresponding to the y-axis, and the third corresponding to the z-axis. 
This domain is not periodic along the z-axis.

# Example
```jldoctest; setup = :(using ClimaAtmos.Domains)
julia> HybridBox(
            xlim = (0, π),
            ylim = (0, π),
            zlim = (0, 1),
            nelements = (5, 5, 10),
            npolynomial = 5,
            xperiodic = true,
            yperiodic = true,
            )
Domain set-up:
\txyz-box:\t[0.0, 3.1) × [0.0, 3.1) × [0.0, 1.0]
\t# of elements:\t(5, 5, 10)
\tpoly order:\t5
```
"""
function HybridBox(
    ::Type{FT} = Float64;
    xlim,
    ylim,
    zlim,
    nelements,
    npolynomial,
    xperiodic = true,
    yperiodic = true,
) where {FT}
    @assert xlim[1] < xlim[2]
    @assert ylim[1] < ylim[2]
    @assert zlim[1] < zlim[2]
    return HybridBox{FT}(
        xlim,
        ylim,
        zlim,
        nelements,
        npolynomial,
        xperiodic,
        yperiodic,
    )
end

struct SphericalShell{FT} <: AbstractHybridDomain{FT}
    radius::FT
    height::FT
    nelements::Tuple{Int, Int}
    npolynomial::Int
end

"""
    SphericalShell([FT = Float64]; radius, height, nelements, npolynomial)

Construct a domain of type `FT` that represents a spherical shell with radius `radius`, height `height`,
and `nelements` elements of polynomial order `npolynomial`.

# Example
```jldoctest; setup = :(using ClimaAtmos.Domains)
julia> SphericalShell(radius = 1, height = 1, nelements = (6, 10), npolynomial = 5)
Domain set-up:
\tsphere radius:\t1.0
\tsphere height:\t1.0
\t# of elements:\t(6, 10)
\tpoly order:\t5
```
"""
function SphericalShell(
    ::Type{FT} = Float64;
    radius,
    height,
    nelements,
    npolynomial,
) where {FT}
    @assert 0 < radius
    @assert 0 < height
    return SphericalShell{FT}(radius, height, nelements, npolynomial)
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

function Base.show(io::IO, domain::HybridBox)
    print(io, "Domain set-up:\n\txyz-box:\t")
    printstyled(io, "[", color = 226)
    printstyled(io, @sprintf("%#.2g, %#.2g", domain.xlim...), color = 7)
    printstyled(io, domain.xperiodic ? ")" : "]", " × [", color = 226)
    printstyled(io, @sprintf("%#.2g, %#.2g", domain.ylim...), color = 7)
    printstyled(io, domain.yperiodic ? ")" : "]", " × [", color = 226)
    printstyled(io, @sprintf("%#.2g, %#.2g", domain.zlim...), color = 7)
    printstyled(io, "]", color = 226)
    @printf(io, "\n\t# of elements:\t(%d, %d, %d)", domain.nelements...)
    print(io, "\n\tpoly order:\t", domain.npolynomial)
end

function Base.show(io::IO, domain::SphericalShell)
    print(io, "Domain set-up:\n\tsphere radius:\t")
    printstyled(io, @sprintf("%#.2g", domain.radius), color = 7)
    print(io, "\n\tsphere height:\t")
    printstyled(io, @sprintf("%#.2g", domain.height), color = 7)
    @printf(io, "\n\t# of elements:\t(%d, %d)", domain.nelements...)
    print(io, "\n\tpoly order:\t", domain.npolynomial)
end
