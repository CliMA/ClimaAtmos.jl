"""
    DefaultMoninObukhov()

Callable that builds a [`MoninObukhov`](@ref) closure with roughness length
`z0 = 1e-5` m and no prescribed fluxes.

Calling `DefaultMoninObukhov()(params)` returns the closure at the float type
of `params`; the indirection lets the configuration name a flux scheme before
the parameter set exists.
"""
struct DefaultMoninObukhov end
function (::DefaultMoninObukhov)(params)
    FT = eltype(params)
    return MoninObukhov(; z0 = FT(1e-5))
end

"""
    DefaultExchangeCoefficients()

Callable that builds an [`ExchangeCoefficients`](@ref) closure with
`Cd = Ch = params.C_H`, the exchange coefficient from the parameter set.
"""
struct DefaultExchangeCoefficients end
(::DefaultExchangeCoefficients)(params) = ExchangeCoefficients(params.C_H)

"""
    FileHeatFluxes(data::ColumnDatasets.ColumnDataset, start_date; nan_to_zero = true)
    FileHeatFluxes(data::AbstractVector{<:ColumnDataset}, start_date; nan_to_zero = true)

Prescribed surface heat fluxes read from a column forcing file, callable as
`(t, FT) -> HeatFluxes`; with one dataset per column, the fluxes are vectors
with one entry per column.

Used as the `fluxes` field of a [`MoninObukhov`](@ref) scheme, where
`resolve_flux_scheme` calls it once per surface update. The `hfls`/`hfss`
series are read from `data` at construction and interpolated linearly in time,
held flat beyond the file's range; `t = 0` is `start_date`. The file's
convention (upward-positive latent and sensible heat fluxes) matches
[`HeatFluxes`](@ref). Construction errors if the file lacks `hfls`/`hfss`.

# Keyword Arguments

  - `nan_to_zero = true`: Whether masked or fill-value gaps (NaN) evaluate to
    zero flux rather than propagating NaN.

# Fields

  - `lhf_interp`, `shf_interp`: Time interpolants of the latent and sensible heat
    flux series [W/m²], in seconds since `start_date`; one per column for a
    vector of datasets.
  - `nan_to_zero`: As above [-].

# Examples

```julia
flux_scheme = MoninObukhov(; z0 = 0.05, ustar = 0.28,
    fluxes = FileHeatFluxes(data, "20140201"))
```
"""
struct FileHeatFluxes{L, S} <: Function
    lhf_interp::L
    shf_interp::S
    nan_to_zero::Bool
end

function FileHeatFluxes(
    data::ColumnDatasets.ColumnDataset,
    start_date;
    nan_to_zero = true,
)
    issubset((:hfls, :hfss), data.surface_vars) || error(
        "`FileHeatFluxes` for $(data.path) requires the `hfls` and `hfss` \
         surface variables",
    )
    read = ColumnDatasets.read_surface_series(
        data,
        (:hfls, :hfss),
        parse_date(start_date),
    )
    return FileHeatFluxes(
        _flat_time_interpolant(read.times, read.hfls),
        _flat_time_interpolant(read.times, read.hfss),
        nan_to_zero,
    )
end

# One dataset per column: one interpolant pair per column
function FileHeatFluxes(
    data::AbstractVector{<:ColumnDatasets.ColumnDataset},
    start_date;
    nan_to_zero = true,
)
    parts = map(d -> FileHeatFluxes(d, start_date; nan_to_zero), data)
    return FileHeatFluxes(
        getfield.(parts, :lhf_interp),
        getfield.(parts, :shf_interp),
        nan_to_zero,
    )
end

"""
    (f::FileHeatFluxes)(t, ::Type{FT})

Return the [`HeatFluxes`](@ref) interpolated to simulation time `t` [s], with
components converted to `FT` [W/m²] (vectors over the columns for one dataset
per column).
"""
function (f::FileHeatFluxes)(t, ::Type{FT}) where {FT}
    t_sec = Float64(t isa Number ? t : float(t))
    lhf = _at(f.lhf_interp, t_sec)
    shf = _at(f.shf_interp, t_sec)
    if f.nan_to_zero
        lhf = ifelse.(isnan.(lhf), 0.0, lhf)
        shf = ifelse.(isnan.(shf), 0.0, shf)
    end
    return HeatFluxes(; shf = FT.(shf), lhf = FT.(lhf))
end
_at(interp, t) = interp(t)
_at(interps::Vector, t) = map(interp -> interp(t), interps)

_flat_time_interpolant(times, data) = Interpolations.extrapolate(
    Interpolations.interpolate(
        (Float64.(times),),
        Float64.(data),
        Interpolations.Gridded(Interpolations.Linear()),
    ),
    Interpolations.Flat(),
)
