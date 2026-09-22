# Utilities for comparing, field by field, the states of simulations on column spaces:
# a single column against one column of a multi-column run, or two columns of the same
# multi-column run. Modeled on ClimaLand's experiments/integrated/era5/comparison_utils.jl
# and ClimaAtmos's test/restart_utils.jl.
import ClimaCore
import ClimaCore: Fields, Geometry, Spaces
import NCDatasets
using Printf

rms(x) = isempty(x) ? 0.0 : sqrt(sum(abs2, x) / length(x))

"""
    mixed_error(x1, x2; abs_floor = 100 * eps(eltype(x1)))

Worst elementwise error of `x2` relative to `x1`: relative where `|x1| > abs_floor`,
absolute otherwise.
"""
function mixed_error(x1, x2; abs_floor = 100 * eps(eltype(x1)))
    diff = abs.(x1 .- x2)
    return maximum(ifelse.(abs.(x1) .> abs_floor, diff ./ abs.(x1), diff); init = 0.0)
end

"""
    FieldDiff

Worst mixed error, root-mean-square error, worst absolute error, root-mean-square
magnitude of the reference (first) field, and the number of non-finite entries in either
field (which are zeroed before the errors are computed). A non-finite entry only matches
a bitwise-identical one, so `NaN` against `0` gives infinite errors rather than passing
as agreement. A shape mismatch or an exception gives `NaN` errors.
"""
struct FieldDiff
    err::Float64
    rmse::Float64
    maxerr::Float64
    scale::Float64
    nonfinite::Int
end
FieldDiff(err, rmse, maxerr, scale) = FieldDiff(err, rmse, maxerr, scale, 0)
# Recorded when the two sides cannot be compared at all, so that it always fails.
const NOT_COMPARABLE = FieldDiff(NaN, NaN, NaN, NaN)
function FieldDiff(x1::AbstractVector, x2::AbstractVector)
    length(x1) == length(x2) || return NOT_COMPARABLE
    nonfinite = count(!isfinite, x1) + count(!isfinite, x2)
    y1, y2 = mask_nonfinite(x1), mask_nonfinite(x2)
    # `isequal` matches NaN with NaN; without this test, zeroing a non-finite entry would
    # make it compare equal to a finite zero on the other side.
    nonfinite_pair(i) = !(isfinite(x1[i]) && isfinite(x2[i]))
    if any(i -> nonfinite_pair(i) && !isequal(x1[i], x2[i]), eachindex(x1))
        return FieldDiff(Inf, Inf, Inf, rms(y1), nonfinite)
    end
    d = abs.(y1 .- y2)
    return FieldDiff(
        mixed_error(y1, y2),
        rms(d),
        maximum(d; init = 0.0),
        rms(y1),
        nonfinite,
    )
end

mask_nonfinite(x) = x .* isfinite.(x)

# Covariant components depend on the local basis, which differs between a Cartesian
# single column and a spherical multi-column grid; compare physical components instead.
physical(field) =
    eltype(field) <: Geometry.AxisVector ? Geometry.UVWVector.(field) : field

"""
    flatten(field, col)

Flatten `field` into a vector: the whole field for `col = nothing`, or its `col`th
column.
"""
flatten(field::Fields.Field, ::Nothing) = vec(Array(parent(physical(field))))
flatten(field::Fields.Field, col::Integer) =
    flatten(Fields.column(field, 1, 1, col), nothing)

is_leaf_eltype(T) = T <: Number || T <: Geometry.AxisVector
# Rank-2 tensors (e.g. the surface momentum flux) carry the horizontal metric in their
# covariant components; their effect is visible through the state, so they are skipped.
is_tensor_field(v) =
    v isa Fields.Field &&
    eltype(v) <: Geometry.AxisTensor &&
    !(eltype(v) <: Geometry.AxisVector)
const NOT_COMPARED = Union{
    AbstractArray,
    AbstractString,
    Symbol,
    Function,
    Type,
    Ref,
    Ptr,
    Nothing,
    Bool,
}
const MAX_DEPTH = 10

"""
    field_diffs(v1, v2; col1 = nothing, col2 = nothing, name = "", ignore = Set{Symbol}())

Recursively compare the `Field`s and numbers reachable from `v1` and `v2`, skipping the
property names in `ignore`, and return a `Dict` of `FieldDiff`s keyed by property path.
`col1` and `col2` select the column of each side (`nothing` for a whole single column).
Properties whose access throws are recorded with the exception type in the key, and a
property that holds different kinds of value on the two sides is recorded as a type
mismatch; both always fail, so nothing is dropped without a trace.
"""
field_diffs(v1, v2; col1 = nothing, col2 = nothing, name = "", ignore = Set{Symbol}()) =
    _field_diffs!(Dict{String, FieldDiff}(), v1, v2, name, col1, col2, ignore, 0)

# Which comparison path a value takes. Both sides must take the same path: a property
# that is, say, a `Field` in one run and `nothing` in the other is a difference, and
# dispatching on `v1` alone would silently drop it.
function _category(v)
    is_tensor_field(v) && return :tensor
    v isa Fields.Field && return :field
    v isa Number && return :number
    v isa Union{NamedTuple, Fields.FieldVector, Tuple} && return :composite
    v isa NOT_COMPARED && return :not_compared
    return isstructtype(typeof(v)) ? :struct : :opaque
end

function _field_diffs!(diffs, v1, v2, name, col1, col2, ignore, depth)
    if depth > MAX_DEPTH
        @warn "field_diffs: nesting deeper than $MAX_DEPTH at $name; not compared"
        return diffs
    end
    category = _category(v1)
    if category != _category(v2)
        diffs["$name ($(nameof(typeof(v1))) vs $(nameof(typeof(v2))))"] = NOT_COMPARABLE
    elseif category in (:tensor, :not_compared, :opaque)
        return diffs
    elseif category == :field
        if is_leaf_eltype(eltype(v1)) && is_leaf_eltype(eltype(v2))
            diffs[name] = FieldDiff(flatten(v1, col1), flatten(v2, col2))
        else
            _recurse!(diffs, v1, v2, name, col1, col2, ignore, depth)
        end
    elseif category == :number
        diffs[name] = FieldDiff([float(v1)], [float(v2)])
    else
        _recurse!(diffs, v1, v2, name, col1, col2, ignore, depth)
    end
    return diffs
end

function _recurse!(diffs, v1, v2, name, col1, col2, ignore, depth)
    props = filter(!in(ignore), collect(propertynames(v1)))
    if props != filter(!in(ignore), collect(propertynames(v2)))
        diffs[name] = NOT_COMPARABLE
        return diffs
    end
    for p in props
        try
            _field_diffs!(
                diffs, getproperty(v1, p), getproperty(v2, p),
                "$name.$p", col1, col2, ignore, depth + 1,
            )
        catch e
            diffs["$name.$p ($(nameof(typeof(e))))"] = NOT_COMPARABLE
        end
    end
    return diffs
end

"""
    diagnostic_variable(ds)

Name of the single diagnostic variable in a ClimaDiagnostics NetCDF dataset, i.e. the
variable that is neither a dimension, a bounds variable, nor a coordinate.
"""
function diagnostic_variable(ds)
    skip = ("date", "lat", "lon")
    names = filter(collect(keys(ds))) do k
        !(k in NCDatasets.dimnames(ds)) && !endswith(k, "_bnds") && !(k in skip)
    end
    return only(names)
end

"""
    diagnostic_diffs(dir_single, dir_multi; col = 1)

Compare every NetCDF diagnostic in `dir_single` (dimensions `(time[, z])`) with column
`col` of the same diagnostic in `dir_multi` (dimensions `(time, column[, z])`).
"""
function diagnostic_diffs(dir_single, dir_multi; col = 1)
    diffs = Dict{String, FieldDiff}()
    for file in filter(endswith(".nc"), readdir(dir_single))
        path_multi = joinpath(dir_multi, file)
        isfile(path_multi) || (diffs[file] = NOT_COMPARABLE; continue)
        NCDatasets.NCDataset(joinpath(dir_single, file)) do ds1
            NCDatasets.NCDataset(path_multi) do dsn
                name = diagnostic_variable(ds1)
                a1 = Array(ds1[name])
                an = Array(dsn[name])
                ak = ndims(an) == 3 ? an[:, col, :] : an[:, col]
                diffs[file] = FieldDiff(vec(Float64.(a1)), vec(Float64.(ak)))
            end
        end
    end
    return diffs
end

"""
    passes(d::FieldDiff, rtol, atol)

Whether the worst mixed error is within `rtol`, or the worst absolute error is within
`atol` (for fields that are essentially zero). `atol` is applied to the worst absolute
error rather than the RMS error, so that an isolated large difference cannot be averaged
away by the rest of the field.
"""
passes(d::FieldDiff, rtol, atol) = d.err <= rtol || d.maxerr <= atol

"""
    report_diffs(diffs; label = "", rtol = 0.0, atol = 0.0)

Print the entries of `diffs` that exceed both tolerances and return their number. An
empty `diffs` counts as one failure: comparing nothing is no evidence of agreement.
"""
function report_diffs(diffs; label = "", rtol = 0.0, atol = 0.0)
    bad = filter(((_, d),) -> !passes(d, rtol, atol), diffs)
    println(
        "Field comparison [$label]: $(length(diffs)) entries, ",
        "$(length(bad)) exceed rtol=$rtol atol=$atol",
    )
    for (path, d) in sort(collect(bad); by = first)
        @printf(
            "  %-58s err=%-9.3g rmse=%-9.3g maxerr=%-9.3g scale=%-9.3g nonfinite=%d\n",
            path, d.err, d.rmse, d.maxerr, d.scale, d.nonfinite,
        )
    end
    isempty(diffs) && println("  nothing was compared")
    return isempty(diffs) ? 1 : length(bad)
end

# Cache entries holding bookkeeping, solver handles, file readers, or uninitialized scratch.
const CACHE_IGNORE = Set([
    :params, :atmos, :dt, :output_dir, :job_id, :start_date, :ghost_buffer,
    :rrtmgp_solver, :sfc_setup, :steady_state_velocity, :scratch, :data_handler,
])
