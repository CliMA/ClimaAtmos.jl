"""
Vertical bounds on the scored region.

The scoring window is restricted to cloud top plus a buffer. Above that the LES condensate is
invariably zero, so those levels contribute nothing but dimension, and the model's upper boundary
can produce spurious cloud there.

The bound is *derived* from the LES cloud top, so every case gets a principled value, with hand-set
per-flight numbers available as explicit overrides.
"""

using NCDatasets: NCDatasets as NC

"""Hand-set upper bounds [m] on the scored region, by flight number, for the `Obs`-forced cases."""
const OBS_Z_TOP = Base.ImmutableDict(
    1 => 3000.0,
    9 => 4000.0,
    10 => 3500.0,
    12 => 2000.0,
    13 => 2000.0,
)

"""Default headroom [m] added above the LES cloud top before rounding."""
const DEFAULT_Z_BUFFER = 1000.0

"""Rounding granularity [m] for a derived bound."""
const DEFAULT_Z_ROUND = 500.0

"""
    les_cloud_top(case; window, source, threshold)

Highest altitude [m] at which the LES carries cloud condensate during `window`.

Condensate is `QCL + QCI` (the reference's `clw` and `cli`), and a level counts as cloudy when it
exceeds `threshold` times the maximum over the window — a relative threshold so it behaves the same
for a liquid case and a much thinner ice case.
"""
function les_cloud_top(
    case::SocratesCase;
    window = score_window(case),
    source::Symbol = :processed,
    threshold::Real = 0.01,
)
    vars = les_outputvars(case; source, vars = ("clw", "cli"))
    z = vars["clw"].dims["z"]
    t = vars["clw"].dims["time"]
    keep = findall(ti -> first(window) <= ti <= last(window), t)
    isempty(keep) && error(
        "No LES times inside the scoring window $(window) s for $(case_name(case)); the LES \
         record spans $(extrema(t)) s.",
    )
    condensate = zeros(Float64, length(z))
    for name in ("clw", "cli")
        d = vars[name].data
        for k in eachindex(z)
            v = maximum(x -> isfinite(x) ? x : 0.0, view(d, k, keep))
            condensate[k] = max(condensate[k], v)
        end
    end
    peak = maximum(condensate)
    peak > 0 || error(
        "The LES reference for $(case_name(case)) has no cloud condensate in the scoring window, \
         so a cloud-top bound cannot be derived. Set an explicit override.",
    )
    cloudy = findall(>(threshold * peak), condensate)
    return z[last(cloudy)]
end

"""
    z_bounds(case; buffer, round_to, overrides, source, z_max)

The `(z_bottom, z_top)` bounds [m] of the scored region for `case`.

`z_top` is the explicit `overrides` entry when present, otherwise
`min(z_max, roundup(les_cloud_top + buffer, round_to))`.

`z_bottom` is 0: on the default (native) grid the model's levels *are* the LES levels, so every
scored level exists in both and nothing is interpolated. [`scored_levels`](@ref) selects the model
levels inside the bounds, which also keeps a coarsened or user-supplied grid strictly inside the
reference's own range.

Pass `overrides = Base.ImmutableDict{Int, Float64}()` to derive every case, or add entries to pin
particular ones.
"""
function z_bounds(
    case::SocratesCase;
    buffer::Real = DEFAULT_Z_BUFFER,
    round_to::Real = DEFAULT_Z_ROUND,
    overrides = case.forcing_type isa SSCF.ObsForcing ? OBS_Z_TOP :
                Base.ImmutableDict{Int, Float64}(),
    source::Symbol = :processed,
    z_max::Real = z_max_default(case),
)
    bottom = 0.0
    top = if haskey(overrides, case.flight_number)
        Float64(overrides[case.flight_number])
    else
        cloud_top = les_cloud_top(case; source)
        Float64(min(z_max, round_to * ceil((cloud_top + buffer) / round_to)))
    end
    bottom < top || error(
        "Scored region for $(case_name(case)) is empty: the LES starts at $bottom m but the upper \
         bound is $top m.",
    )
    return (bottom, top)
end

"""
    scored_levels(z_grid, bounds)

The entries of `z_grid` inside `bounds` — the model levels that both the model and the reference
cover, and therefore the levels a profile is scored on.

Resampling the reference onto exactly these levels means the interpolation is always interior, so no
reference value is ever extrapolated into a region the LES does not observe.
"""
function scored_levels(z_grid, bounds)
    lo, hi = bounds
    levels = filter(z -> lo <= z <= hi, collect(Float64, z_grid))
    isempty(levels) && error(
        "No model levels fall inside the scored region $(bounds) m; the grid spans \
         $(extrema(z_grid)) m.",
    )
    return levels
end
