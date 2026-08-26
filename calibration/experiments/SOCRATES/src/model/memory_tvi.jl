"""
An in-memory `(z, t)` column `TimeVaryingInput`.

ClimaUtilities' 2-D/3-D `TimeVaryingInput` requires a file-backed `DataHandler`
(`ext/TimeVaryingInputsExt.jl`), and its in-memory input fills its destination with a single
scalar (`ext/TimeVaryingInputs0DExt.jl`) — so neither can drive a column field from arrays already
in memory. This supplies that one missing piece: the data is pre-sampled onto the model's own
levels at build time, so evaluating is a linear blend of two adjacent columns and touches no files.
"""

using ClimaAtmos: ClimaAtmos as CA
using ClimaComms: ClimaComms

"""
    ColumnMemoryTimeVaryingInput(times, data, target_space; method)

A `TimeVaryingInput` over `data`, a `(n_levels, n_times)` array already on the target space's
levels, sampled at `times` (seconds since the simulation start date, sorted ascending).

`data` is moved to `target_space`'s device once here, so `evaluate!` does no host/device transfer
and no allocation.

`method` fixes the out-of-range policy: `LinearInterpolation()` (the default) errors outside
`times`, `LinearInterpolation(Flat())` holds the end values.
"""
struct ColumnMemoryTimeVaryingInput{T <: AbstractVector, D <: AbstractMatrix, M} <:
    CA.ClimaUtilities.TimeVaryingInputs.AbstractTimeVaryingInput
    """Sample times [s] since the simulation start date, ascending. Host-resident."""
    times::T
    """`(n_levels, n_times)`, on the target space's device."""
    data::D
    """Interpolation method; its `extrapolation_bc` sets the out-of-range policy."""
    method::M
end

function ColumnMemoryTimeVaryingInput(
    times::AbstractVector,
    data::AbstractMatrix,
    target_space;
    method = CA.ClimaUtilities.TimeVaryingInputs.LinearInterpolation(),
)
    issorted(times) || error("ColumnMemoryTimeVaryingInput times must be sorted ascending")
    length(times) >= 2 ||
        error("ColumnMemoryTimeVaryingInput needs at least 2 times, got $(length(times))")
    FT = CA.CC.Spaces.undertype(target_space)
    n_levels = _n_levels(target_space)
    size(data, 1) == n_levels || error(
        "ColumnMemoryTimeVaryingInput data has $(size(data, 1)) levels but the target space \
         has $n_levels. The data must already be sampled on the model's levels.",
    )
    size(data, 2) == length(times) || error(
        "ColumnMemoryTimeVaryingInput data has $(size(data, 2)) time samples but $(length(times)) \
         times were given",
    )
    array_type = ClimaComms.array_type(ClimaComms.device(target_space))
    t = collect(Float64, times)
    d = array_type(FT.(data))
    # Spelled with explicit type parameters: the field constructor has the same three-positional
    # signature as this one, so an unparameterized call would recurse into here instead.
    return ColumnMemoryTimeVaryingInput{typeof(t), typeof(d), typeof(method)}(t, d, method)
end

_n_levels(space) = size(parent(CA.CC.Fields.coordinate_field(space).z), 1)

Base.in(t::Number, itp::ColumnMemoryTimeVaryingInput) =
    first(itp.times) <= t <= last(itp.times)

"""Seconds since the simulation start date, for any time representation ClimaAtmos may pass."""
_seconds(t::Number) = Float64(t)
_seconds(t::CA.ClimaUtilities.TimeManager.ITime) = Float64(float(t))

"""
    evaluate!(dest, itp::ColumnMemoryTimeVaryingInput, t)

Fill the column field `dest` by linear interpolation of `itp` in time.

"""
function CA.ClimaUtilities.TimeVaryingInputs.evaluate!(
    dest,
    itp::ColumnMemoryTimeVaryingInput,
    t,
    args...;
    kwargs...,
)
    ts = _seconds(t)
    i, w = _bracket(itp.times, ts, CA.ClimaUtilities.TimeVaryingInputs.extrapolation_bc(itp.method))
    dv = parent(dest)
    d = itp.data
    if w == 0
        @inbounds @views dv[:, 1] .= d[:, i]
    else
        @inbounds @views dv[:, 1] .= (1 - w) .* d[:, i] .+ w .* d[:, i + 1]
    end
    return nothing
end

# Index `i` and weight `w` such that the value at `ts` is `(1-w)*data[:, i] + w*data[:, i+1]`.
# `w == 0` signals "use column `i` exactly", which also covers both clamped ends.
function _bracket(times, ts, bc)
    if ts <= first(times)
        ts < first(times) && _check_in_range(bc, ts, times)
        return (firstindex(times), 0.0)
    elseif ts >= last(times)
        ts > last(times) && _check_in_range(bc, ts, times)
        return (lastindex(times), 0.0)
    end
    i = searchsortedlast(times, ts)
    t0, t1 = times[i], times[i + 1]
    return (i, (ts - t0) / (t1 - t0))
end

_check_in_range(::CA.ClimaUtilities.TimeVaryingInputs.Flat, ts, times) = nothing
_check_in_range(bc, ts, times) = error(
    "ColumnMemoryTimeVaryingInput evaluated at t = $ts s, outside its data range \
     [$(first(times)), $(last(times))] s. Extend the forcing time axis, or build the input with \
     `method = TimeVaryingInputs.LinearInterpolation(TimeVaryingInputs.Flat())` to hold the end \
     values instead.",
)