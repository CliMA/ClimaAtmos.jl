import ForwardDiff
import ClimaComms: @threaded
import LinearAlgebra: I, Adjoint, diagind
import ClimaCore.InputOutput: HDF5, HDF5Writer

using ClimaCore.MatrixFields
import ClimaCore.MatrixFields: @name

"""
    JacobianAlgorithm

A description of how to compute the matrix ``∂R/∂Y``, where ``R(Y)`` denotes the
residual of an implicit step with the state ``Y``. Concrete implementations of
this abstract type should define 4 methods:
 - `jacobian_cache(alg::JacobianAlgorithm, Y, atmos)`
 - `update_jacobian!(alg::JacobianAlgorithm, cache, Y, p, dtγ, t)`
 - `invert_jacobian!(alg::JacobianAlgorithm, cache, ΔY, R)`
 - `save_jacobian!(alg::JacobianAlgorithm, cache, Y, p, dtγ, t)`
An additional method can also be defined to enable debugging of the Jacobian:
 - `update_and_check_jacobian!(alg::JacobianAlgorithm, cache, Y, p, dtγ, t)`

See [Implicit Solver](@ref) for additional background information.
"""
abstract type JacobianAlgorithm end

"""
    Jacobian(alg, Y, atmos, output_dir)

Wrapper for a `JacobianAlgorithm` and its cache, which it uses to update and
invert the Jacobian. The `output_dir` is the directory used for saving plots.
"""
struct Jacobian{A <: JacobianAlgorithm, C}
    alg::A
    cache::C
end
function Jacobian(alg, Y, atmos, output_dir)
    FT = eltype(Y)
    DA = ClimaComms.array_type(Y)
    Yₜ = similar(Y)

    n_columns = Fields.ncolumns(Y.c)
    n_εs = length(first_column(Y))
    column_vectors = DA{FT}(undef, n_columns, n_εs)
    column_matrix = DA{FT}(undef, n_εs, n_εs)
    column_vector = DA{FT}(undef, n_εs)

    is_cpu = DA <: Array
    cpu_column_matrix = is_cpu ? column_matrix : Array(column_matrix)
    cpu_column_vector = is_cpu ? column_vector : Array(column_vector)

    plot_cache = (;
        Yₜ,
        column_vectors,
        column_matrix,
        column_vector,
        cpu_column_matrix,
        cpu_column_vector,
        output_dir,
    )
    krylov_cache = (; ΔY_krylov = similar(Y), R_krylov = similar(Y))
    cache = (; jacobian_cache(alg, Y, atmos)..., plot_cache..., krylov_cache...)
    return Jacobian(alg, cache)
end

# ClimaTimeSteppers.jl calls zero(jac_prototype) to initialize the Jacobian, but
# we don't need to allocate a second Jacobian for this (in particular, the exact
# Jacobian can be very expensive to allocate).
Base.zero(jacobian::Jacobian) = jacobian

# Convert dtγ (which may be an ITime) to the same floating point type as Y.
safe_float(dtγ, Y) = eltype(Y)(float(dtγ))

# These are either called by ClimaTimeSteppers.jl before each linear solve, or
# by a callback once every dt_update_auto_jacobian.
NVTX.@annotate update_jacobian!(jacobian, Y, p, dtγ, t) =
    update_jacobian!(jacobian.alg, jacobian.cache, Y, p, safe_float(dtγ, Y), t)
NVTX.@annotate update_and_check_jacobian!(jacobian, Y, p, dtγ, t) =
    update_and_check_jacobian!(
        jacobian.alg,
        jacobian.cache,
        Y,
        p,
        safe_float(dtγ, Y),
        t,
    )

# This is called by ClimaTimeSteppers.jl before each linear solve.
NVTX.@annotate LinearAlgebra.ldiv!(
    ΔY::Fields.FieldVector,
    jacobian::Jacobian,
    R::Fields.FieldVector,
) = invert_jacobian!(jacobian.alg, jacobian.cache, ΔY, R)

# This is called by Krylov.jl from inside ClimaTimeSteppers.jl. See
# https://github.com/JuliaSmoothOptimizers/Krylov.jl/issues/605 for a related
# issue that requires the same workaround.
function LinearAlgebra.ldiv!(
    ΔY::AbstractVector,
    jacobian::Jacobian,
    R::AbstractVector,
)
    (; ΔY_krylov, R_krylov) = jacobian.cache
    R_krylov .= R
    LinearAlgebra.ldiv!(ΔY_krylov, jacobian, R_krylov)
    ΔY .= ΔY_krylov
end

# This is called by a callback once every dt_save_jacobian.
NVTX.@annotate function save_jacobian!(jacobian, Y, p, dtγ, t)
    (; Yₜ, column_vectors, column_vector) = jacobian.cache

    # TODO: Add support for MPI reductions, instead of only saving from root.
    ClimaComms.iamroot(ClimaComms.context(Y.c)) || return

    implicit_tendency!(Yₜ, Y, p, t)
    column_vectors_to_field_vector(column_vectors, Yₜ) .= Yₜ
    sum!(abs, reshape(column_vector, 1, :), column_vectors)
    column_vector ./= Fields.ncolumns(Y.c)

    save_jacobian!(jacobian.alg, jacobian.cache, Y, safe_float(dtγ, Y), t)
end

contains_any_fields(::Union{Fields.Field, Fields.FieldVector}) = true
contains_any_fields(x::T) where {T} =
    fieldcount(T) == 0 ? false : unrolled_any(StaticOneTo(fieldcount(T))) do i
        contains_any_fields(getfield(x, i))
    end

first_column(x::MatrixFields.FieldNameDict) =
    MatrixFields.FieldNameDict(x.keys, first_column(x.entries))
first_column(x::Union{Fields.Field, Fields.FieldVector}) =
    Fields.column(x, 1, 1, 1)
first_column(x::Union{Tuple, NamedTuple}) = unrolled_map(first_column, x)
first_column(x::T) where {T} =
    fieldcount(T) == 0 || !contains_any_fields(x) ? x :
    T.name.wrapper(
        ntuple(i -> first_column(getfield(x, i)), Val(fieldcount(T)))...,
    )

function first_column_str(Y)
    coord_field =
        Fields.coordinate_field(Fields.level(Fields.column(Y.c, 1, 1, 1), 1))
    coord =
        ClimaComms.allowscalar(getindex, ClimaComms.device(Y.c), coord_field)
    round_value(value) = round(value; sigdigits = 3)
    return if coord isa Geometry.XZPoint
        "x = $(round_value(coord.x)) Meters"
    elseif coord isa Geometry.XYZPoint
        "x = $(round_value(coord.x)) Meters, y = $(round_value(coord.y)) Meters"
    elseif coord isa Geometry.LatLongZPoint
        "lat = $(round_value(coord.lat))°, long = $(round_value(coord.long))°"
    else
        error("Unrecognized coordinate type $(typeof(coord))")
    end
end

function save_cached_column_matrix_and_vector!(cache, file_name, title, t)
    (; Yₜ, column_matrix, column_vector, output_dir) = cache
    (; cpu_column_matrix, cpu_column_vector) = cache
    if !(column_matrix isa Array)
        copyto!(cpu_column_matrix, column_matrix)
        copyto!(cpu_column_vector, column_vector)
    end
    file_name = joinpath(output_dir, file_name * ".hdf5")
    HDF5Writer(file_name, ClimaComms.context(Yₜ.c); overwrite = false) do writer
        key = string(float(t))
        haskey(writer.file, key) && return
        group = HDF5.create_group(writer.file, key)
        group["title"] = "$title, t = $(time_and_units_str(float(t)))"
        group["∂Yₜ_∂Y"] = cpu_column_matrix
        group["Yₜ"] = cpu_column_vector
    end
end
