"""
    AtmosphericProfilesLibrary

Published reference profiles for single-column and LES cases (BOMEX, DYCOMS,
GABLS, Rico, TRMM-LBA, ...), in SI units.

Each profile is a function `case_variable(::Type{FT})` that returns a type-stable
callable wrapped in one of the `AbstractProfile` subtypes, which name its arguments:

  - `ZProfile`: `prof(z)`
  - `TimeProfile`: `prof(t)`
  - `TimeZProfile`: `prof(t, z)`
  - `ΠZProfile`: `prof(Π, z)`, where `Π` is the Exner function
  - `ΠTimeZProfile`: `prof(Π, t, z)`

A few functions such as `TRMM_LBA_z` return the raw data grid instead, and
`Larcform1_constants` returns the Larcform1 case constants.

The module depends only on interpolation packages, not on the rest of ClimaAtmos.

# Examples

```julia
import ClimaAtmos.AtmosphericProfilesLibrary as APL
θ = APL.Bomex_θ_liq_ice(Float64)
θ(500.0)
```
"""
module AtmosphericProfilesLibrary

import ClimaInterpolations.Interpolation1D as CI1D
import Interpolations as Intp
import LinearAlgebra
import StaticArrays: SVector

abstract type AbstractProfile end

struct TimeProfile{P} <: AbstractProfile
    prof::P
end
@inline (prof::TimeProfile)(t) = prof.prof(t)

struct TimeZProfile{P} <: AbstractProfile
    prof::P
end
@inline (prof::TimeZProfile)(t, z) = prof.prof(t, z)

struct ZProfile{P} <: AbstractProfile
    prof::P
end
@inline (prof::ZProfile)(z) = prof.prof(z)

struct ΠTimeZProfile{P} <: AbstractProfile
    prof::P
end
@inline (prof::ΠTimeZProfile)(Π, t, z) = prof.prof(Π, t, z)

struct ΠZProfile{P} <: AbstractProfile
    prof::P
end
@inline (prof::ΠZProfile)(Π, z) = prof.prof(Π, z)

struct LinearInterpolation{I}
    itp::I
end

@inline (li::LinearInterpolation)(z) = li.itp(convert(eltype(li.itp.xsource), z))

@inline function linear_interp(z, x)
    @assert length(z) == length(x)
    N = length(z)
    T = promote_type(eltype(z), eltype(x))
    xsource = SVector{N, T}(z)
    fsource = SVector{N, T}(x)
    itp = CI1D.Interpolate1D(
        xsource,
        fsource;
        interpolationorder = CI1D.Linear(),
        extrapolationorder = CI1D.Flat(),
    )
    return LinearInterpolation(itp)
end

# Data-based profiles
include("Soares.jl")
include("Nieuwstadt.jl")
include("Bomex.jl")
include("LifeCycleTan2018.jl")
include("Rico.jl")
include("TRMM_LBA.jl")
include("ARM_SGP.jl")
include("GATE_III.jl")
include("Dycoms_RF01.jl")
include("Dycoms_RF02.jl")
include("GABLS.jl")
include("SP.jl")
include("DryBubble.jl")
include("ISDAC.jl")

# Idealized profiles
include("Larcform1.jl")

end # module
