#####
##### Forcing-term specs for file-driven single-column forcing
#####
##### A file-driven forcing (`ExternalDrivenTVForcing`) is composed from a tuple
##### of these value objects, one per physical process. Each term declares its
##### data requirements (`required_column_variables`) here; its cache, per-step
##### refresh, and tendency contribution are defined in `external_forcing.jl`.
#####

"""
    AbstractForcingTerm

Supertype of the file-driven forcing terms composed into an
`ExternalDrivenTVForcing`. Concrete terms: [`HorizontalAdvection`](@ref),
[`VerticalFluctuation`](@ref), [`Nudging`](@ref), [`Subsidence`](@ref).
"""
abstract type AbstractForcingTerm end

"""
    HorizontalAdvection()

Large-scale horizontal-advection tendencies of temperature and total specific
humidity (canonical file variables `tntha`, `tnhusha`).
"""
struct HorizontalAdvection <: AbstractForcingTerm end

"""
    VerticalFluctuation()

Vertical eddy-fluctuation tendencies of temperature and total specific
humidity (canonical file variables `tntva`, `tnhusva`).
"""
struct VerticalFluctuation <: AbstractForcingTerm end

"""
    Subsidence()

Large-scale subsidence, from the vertical velocity `wa`.
"""
struct Subsidence <: AbstractForcingTerm end

"""
    DefaultTimescale()

Marker for [`Nudging`](@ref): resolve the inverse relaxation timescale from the
`CAP.gcmdriven_*` parameters (the Shen et al. 2022 profile) at cache build.
"""
struct DefaultTimescale end

const NUDGING_SCALAR_VARS = (:ta, :hus)
const NUDGING_MOMENTUM_VARS = (:ua, :va)
const NUDGING_VARS = (NUDGING_SCALAR_VARS..., NUDGING_MOMENTUM_VARS...)

"""
    Nudging(variables...; timescale = DefaultTimescale(), mask = nothing)

Relax the listed prognostic `variables` (a subset of `$(NUDGING_VARS)`) toward
the file's profiles. Compose multiple `Nudging` terms to give different groups
different timescales or masks (e.g. per-variable relaxation).

  - `timescale`: `DefaultTimescale()` (default parameters), a `Number` (constant
    relaxation timescale τ in seconds), or a function `z -> τ`.
  - `mask`: `nothing`, a `Number`, a function `z -> weight`, or a `Field`. It is
    multiplied into the inverse timescale and materialized once at cache build.
    Height-dependent masks (e.g. relax only above an inversion) use the function
    form.

`:ua` and `:va` share one horizontal-momentum vector, so they must appear in
the same term. A `DefaultTimescale` has no single value across a mixed
scalar/momentum group, so such a group must pass an explicit `timescale` or be
split into separate terms.
"""
struct Nudging{V <: Tuple, T, M} <: AbstractForcingTerm
    variables::V
    timescale::T
    mask::M
end

function Nudging(
    variables::Symbol...;
    timescale = DefaultTimescale(),
    mask = nothing,
)
    isempty(variables) &&
        error("`Nudging` requires at least one variable from $(NUDGING_VARS)")
    for v in variables
        v in NUDGING_VARS ||
            error("`Nudging` variable `$v` is not one of $(NUDGING_VARS)")
    end
    allunique(variables) ||
        error("`Nudging` variables must be unique; got $(variables)")
    # ua and va are one C12 momentum vector, nudged together or not at all.
    (:ua in variables) == (:va in variables) || error(
        "`Nudging` of horizontal momentum requires both `:ua` and `:va` in the \
         same term; got $(variables)",
    )
    if timescale isa DefaultTimescale
        has_scalar = any(in(NUDGING_SCALAR_VARS), variables)
        has_momentum = any(in(NUDGING_MOMENTUM_VARS), variables)
        has_scalar &&
            has_momentum &&
            error(
                "`Nudging` with `DefaultTimescale` cannot mix scalar (:ta/:hus) \
                 and momentum (:ua/:va) variables (they use different default \
                 profiles); got $(variables). Split into two `Nudging` terms or \
                 pass an explicit `timescale`.",
            )
    end
    return Nudging(variables, timescale, mask)
end

"""
    default_forcing_terms()

The forcing composition matching the historical default (horizontal advection,
vertical fluctuation, scalar and momentum nudging on the CAP timescales, and
subsidence).
"""
default_forcing_terms() = (
    HorizontalAdvection(),
    VerticalFluctuation(),
    Nudging(:ta, :hus),
    Nudging(:ua, :va),
    Subsidence(),
)

"""
    required_column_variables(term)

The canonical `(z, time)` file variables a forcing `term` needs. A composed
forcing requires the union over its terms; a file missing any is a loud error.
"""
required_column_variables(::HorizontalAdvection) = (:tntha, :tnhusha)
required_column_variables(::VerticalFluctuation) = (:tntva, :tnhusva)
required_column_variables(::Subsidence) = (:wa,)
required_column_variables(n::Nudging) = n.variables

"""
    validate_forcing_terms(terms::Tuple)

Check a forcing composition: at most one of each non-nudging term, and no
variable nudged by more than one `Nudging` term. Loud error otherwise.
"""
function validate_forcing_terms(terms::Tuple)
    for (T, name) in (
        (HorizontalAdvection, "HorizontalAdvection"),
        (VerticalFluctuation, "VerticalFluctuation"),
        (Subsidence, "Subsidence"),
    )
        count(t -> t isa T, terms) <= 1 ||
            error("a forcing composition may contain at most one $name term")
    end
    nudged = [v for t in terms if t isa Nudging for v in t.variables]
    allunique(nudged) ||
        error("each variable may be nudged by at most one `Nudging` term; got $nudged")
    return nothing
end
