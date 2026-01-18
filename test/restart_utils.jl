redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))
import ClimaAtmos as CA
import ClimaAtmos.RRTMGPInterface as RRTMGPI
import ClimaCore
import ClimaCore: DataLayouts, Fields, Geometry, Meshes
import ClimaCore.Fields: Field, FieldVector, field_values
import ClimaCore.DataLayouts: AbstractData
import ClimaCore.Geometry: AxisTensor
import ClimaCore.Spaces: AbstractSpace
import ClimaComms
import ClimaParams
import ClimaTimeSteppers as CTS
import ClimaUtilities.OutputPathGenerator: maybe_wait_filesystem
pkgversion(ClimaComms) >= v"0.6" && ClimaComms.@import_required_backends
import Logging
import NCDatasets
import YAML
using Test, Dates

import Random
Random.seed!(1234)

const device = ClimaComms.device()
const comms_ctx = ClimaComms.context(device)
ClimaComms.init(comms_ctx)
const secs = 1

MANYTESTS = false
if length(ARGS) > 0
    if ARGS[1] == "--manytests"
        # Check if the first argument is "--manytests" (if provided), if yes, check
        # the second argument for true/false. If the second argument is not provided
        # assume true.
        second_argument = lowercase(get(ARGS, 2, "true"))
        second_argument == "true" && (MANYTESTS = true)
    else
        error("Argument $(ARGS[1]) not recognized")
    end
end
MANYTESTS && @info "Running multiple tests"

# Technical note:
#
# Test.jl really wants to give you a stacktrace for failing tests. This seems to be
# hardcoded in the package and not easy to change without defining a whole new
# AbstractTestSet. We don't want stacktraces, we just want to know which fields are
# different.
#
# For this reason, we don't use Test but just print to screen the differences.

"""
    _error(arr1::AbstractArray, arr2::AbstractArray; ABS_TOL = 100eps(eltype(arr1)))

We compute the error in this way:
- when the absolute value is larger than ABS_TOL, we use the absolute error
- in the other cases, we compare the relative errors
"""
function _error(
    arr1::AbstractArray,
    arr2::AbstractArray;
    ABS_TOL = 100eps(eltype(arr1)),
)
    # There are some parameters, e.g. Obukhov length, for which Inf
    # is a reasonable value (implying a stability parameter in the neutral boundary layer
    # regime, for instance). We account for such instances with the `isfinite` function.
    arr1 = Array(arr1) .* isfinite.(Array(arr1))
    arr2 = Array(arr2) .* isfinite.(Array(arr2))
    diff = abs.(arr1 .- arr2)
    denominator = abs.(arr1)
    error = ifelse.(denominator .> ABS_TOL, diff ./ denominator, diff)
    return error
end


"""
    compare(v1, v2; name = "", ignore = Set([:rc]))

Return whether `v1` and `v2` are the same (up to floating point errors).

`compare` walks through all the properties in `v1` and `v2` until it finds
that there are no more properties. At that point, `compare` tries to match the
resulting objects. When such objects are arrays with floating point, `compare`
defines a notion of `error` that is the following: when the absolute value is
less than `100eps(eltype)`, `error = absolute_error`, otherwise it is relative
error. The `error` is then compared against a tolerance.

Keyword arguments
=================

- `name` is used to collect the name of the property while we go recursively
  over all the properties. You can pass a base name.
- `ignore` is a collection of `Symbol`s that identify properties that are
  ignored when walking through the tree. This is useful for properties that
  are known to be different (e.g., `output_dir`).

`:rc` is some CUDA/CuArray internal object that we don't care about
"""
function compare(
    v1::T,
    v2::T;
    name = "",
    ignore = Set([:rc]),
) where {T <: Union{FieldVector, CA.AtmosCache, AbstractSpace}}
    pass = true
    return _compare(pass, v1, v2; name, ignore)
end

function _compare(pass, v1::T, v2::T; name, ignore) where {T}
    properties = filter(x -> !(x in ignore), propertynames(v1))
    if isempty(properties)
        pass &= _compare(v1, v2; name, ignore)
    else
        # Recursive case
        for p in properties
            pass &= _compare(
                pass,
                getproperty(v1, p),
                getproperty(v2, p);
                name = "$(name).$(p)",
                ignore,
            )
        end
    end
    return pass
end

function _compare(v1::T, v2::T; name, ignore) where {T}
    return print_maybe(v1 == v2, "$name differs")
end

function _compare(
    v1::T,
    v2::T;
    name,
    ignore,
) where {T <: Union{AbstractString, Symbol}}
    # What we can safely print without filling STDOUT
    return print_maybe(v1 == v2, "$name differs: $v1 vs $v2")
end

function _compare(v1::T, v2::T; name, ignore) where {T <: Number}
    # We check with triple equal so that we also catch NaNs being equal
    return print_maybe(v1 === v2, "$name differs: $v1 vs $v2")
end

# We ignore NCDatasets. They contain a lot of state-ful information
function _compare(
    pass,
    v1::T,
    v2::T;
    name,
    ignore,
) where {T <: NCDatasets.NCDataset}
    return pass
end

function _compare(
    v1::T,
    v2::T;
    name,
    ignore,
) where {T <: Field{<:AbstractData{<:Real}}}
    return _compare(parent(v1), parent(v2); name, ignore)
end

function _compare(pass, v1::T, v2::T; name, ignore) where {T <: AbstractData}
    return pass && _compare(parent(v1), parent(v2); name, ignore)
end

# Handle views
function _compare(
    pass,
    v1::SubArray{FT},
    v2::SubArray{FT};
    name,
    ignore,
) where {FT <: AbstractFloat}
    return pass && _compare(collect(v1), collect(v2); name, ignore)
end

function _compare(
    v1::AbstractArray{FT},
    v2::AbstractArray{FT};
    name,
    ignore,
) where {FT <: AbstractFloat}
    error = maximum(_error(v1, v2))
    return print_maybe(error <= 100eps(eltype(v1)), "$name error: $error")
end

function _compare(pass, v1::T1, v2::T2; name, ignore) where {T1, T2}
    error("v1 and v2 have different types")
end

function print_maybe(exp, what)
    exp || println(what)
    return exp
end

# Disable all the @info statements that are produced when creating a simulation
Logging.disable_logging(Logging.Info)
