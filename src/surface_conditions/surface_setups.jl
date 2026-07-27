"""
    DefaultMoninObukhov()

Monin-Obukhov surface with a default roughness length
(see https://clima.github.io/SurfaceFluxes.jl/dev/SurfaceFluxes/#Monin-Obukhov-Similarity-Theory-(MOST)).
"""
struct DefaultMoninObukhov end
function (::DefaultMoninObukhov)(params)
    FT = eltype(params)
    return MoninObukhov(; z0 = FT(1e-5))
end

"""
    DefaultExchangeCoefficients()

Bulk surface, parameterized only by a default exchange coefficient.
"""
struct DefaultExchangeCoefficients end
(::DefaultExchangeCoefficients)(params) = ExchangeCoefficients(params.C_H)

"""
    FileHeatFluxes(data::ColumnDatasets.ColumnDataset, start_date; nan_to_zero = true)

A prescribed surface-heat-flux closure `(t, FT) -> HeatFluxes`, for use as the
`fluxes` of a [`MoninObukhov`](@ref) scheme. It reads the `hfls`/`hfss` series
from `data` at construction and interpolates linearly in time (flat beyond the
file range). Errors loudly at construction if the file lacks `hfls`/`hfss`.
With `nan_to_zero`, masked/fill-value gaps (NaN) evaluate to zero flux.

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

function (f::FileHeatFluxes)(t, ::Type{FT}) where {FT}
    t_sec = Float64(t isa Number ? t : float(t))
    lhf = f.lhf_interp(t_sec)
    shf = f.shf_interp(t_sec)
    if f.nan_to_zero
        lhf = isnan(lhf) ? 0.0 : lhf
        shf = isnan(shf) ? 0.0 : shf
    end
    return HeatFluxes(; shf = FT(shf), lhf = FT(lhf))
end

_flat_time_interpolant(times, data) = Interpolations.extrapolate(
    Interpolations.interpolate(
        (Float64.(times),),
        Float64.(data),
        Interpolations.Gridded(Interpolations.Linear()),
    ),
    Interpolations.Flat(),
)
