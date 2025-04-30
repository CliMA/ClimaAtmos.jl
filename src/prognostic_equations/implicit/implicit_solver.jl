import ForwardDiff
import ClimaCore.MatrixFields: @name
import ClimaCore.InputOutput: HDF5, HDF5Writer

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

# Background

When we use an implicit or split implicit-explicit (IMEX) timestepping scheme,
we end up with a nonlinear equation of the form ``R(Y) = 0``, where
```math
    R(Y) = Y_{imp}(Y) - Y = \\hat{Y} + Δt * T_{imp}(Y) - Y.
```
In this expression, ``Y_{imp}(Y)`` denotes the state at some time ``t + Δt``.
This can be expressed as the sum of ``\\hat{Y}``, the contribution from the
state at time ``t`` (and possibly also at earlier times, depending on the order
of the timestepping scheme), and ``Δt * T_{imp}(Y)``, the contribution from the
implicit tendency ``T_{imp}`` between times ``t`` and ``t + Δt``. The new state
at the end of each implicit step in the timestepping scheme is the value of
``Y`` that solves this equation, i.e., the value of ``Y`` that is consistent
with the state ``Y_{imp}(Y)`` predicted by the implicit step.

Note: When we use a higher-order timestepping scheme, the full step ``Δt`` is
divided into several sub-steps or "stages", where the duration of stage ``i`` is
``Δt * γ_i`` for some constant ``γ_i`` between 0 and 1.

In order to solve this equation using Newton's method, we must specify the
derivative ``∂R/∂Y``. Since ``\\hat{Y}`` does not depend on ``Y`` (it is only a
function of the state at or before time ``t``), this derivative is
```math
    R'(Y) = Δt * T_{imp}'(Y) - I.
```
In addition, we must specify how to divide ``R(Y)`` by this derivative, i.e.,
how to solve the linear equation
```math
    R'(Y) * ΔY = R(Y).
```

Note: This equation comes from assuming that there is some ``ΔY`` such that
``R(Y - ΔY) = 0`` and making the first-order approximation
```math
    R(Y - ΔY) \\approx R(Y) - R'(Y) * ΔY.
```

After initializing ``Y`` to ``Y[0] = \\hat{Y}``, Newton's method executes the
following steps:
- Compute the derivative ``R'(Y[0])``.
- Compute the implicit tendency ``T_{imp}(Y[0])`` and use it to get ``R(Y[0])``.
- Solve the linear equation ``R'(Y[0]) * ΔY[0] = R(Y[0])`` for ``ΔY[0]``.
- Update ``Y`` to ``Y[1] = Y[0] - ΔY[0]``.
If the number of Newton iterations is limited to 1, this new value of ``Y`` is
taken to be the solution of the implicit equation. Otherwise, this sequence of
steps is repeated, i.e., ``ΔY[1]`` is computed and used to update ``Y`` to
``Y[2] = Y[1] - ΔY[1]``, then ``ΔY[2]`` is computed and used to update ``Y`` to
``Y[3] = Y[2] - ΔY[2]``, and so on. The iterative process is terminated either
when the residual ``R(Y)`` is sufficiently close to 0 (according to the
convergence condition passed to Newton's method), or when the maximum number of
iterations is reached.
"""
abstract type JacobianAlgorithm end

"""
    ImplicitEquationJacobian(alg, Y, atmos, output_dir)

Wrapper for a `JacobianAlgorithm` and its cache, which it uses to update and
invert the Jacobian. The `output_dir` is the directory used for saving plots.
"""
struct ImplicitEquationJacobian{A <: JacobianAlgorithm, C}
    alg::A
    cache::C
end
function ImplicitEquationJacobian(alg, Y, atmos, output_dir)
    FT = eltype(Y)
    DA = ClimaComms.array_type(Y)
    Yₜ = similar(Y)

    Y_columns = column_iterator(Y)
    n_columns = length(Y_columns)
    n_εs = length(first(Y_columns))
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
    return ImplicitEquationJacobian(alg, cache)
end

# ClimaTimeSteppers.jl calls zero(jac_prototype) to initialize the Jacobian, but
# we don't need to allocate a second Jacobian for this (in particular, the exact
# Jacobian can be very expensive to allocate).
Base.zero(jacobian::ImplicitEquationJacobian) = jacobian

# These are either called by ClimaTimeSteppers.jl before each linear solve, or
# by a callback once every dt_update_exact_jacobian.
NVTX.@annotate update_jacobian!(jacobian, Y, p, dtγ, t) =
    update_jacobian!(jacobian.alg, jacobian.cache, Y, p, eltype(Y)(dtγ), t)
NVTX.@annotate update_and_check_jacobian!(jacobian, Y, p, dtγ, t) =
    update_and_check_jacobian!(
        jacobian.alg,
        jacobian.cache,
        Y,
        p,
        eltype(Y)(dtγ),
        t,
    )

# This is called by ClimaTimeSteppers.jl before each linear solve.
NVTX.@annotate LinearAlgebra.ldiv!(
    ΔY::Fields.FieldVector,
    jacobian::ImplicitEquationJacobian,
    R::Fields.FieldVector,
) = invert_jacobian!(jacobian.alg, jacobian.cache, ΔY, R)

# This is called by Krylov.jl from inside ClimaTimeSteppers.jl. See
# https://github.com/JuliaSmoothOptimizers/Krylov.jl/issues/605 for a related
# issue that requires the same workaround.
function LinearAlgebra.ldiv!(
    ΔY::AbstractVector,
    jacobian::ImplicitEquationJacobian,
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
    n_columns = length(column_iterator(Y))
    column_vector ./= n_columns

    save_jacobian!(jacobian.alg, jacobian.cache, Y, eltype(Y)(dtγ), t)
end

# Helper functions used to implement save_jacobian!.
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
        group = HDF5.create_group(writer.file, key)
        group["title"] = "$title, t = $(time_and_units_str(float(t)))"
        group["∂Yₜ_∂Y"] = cpu_column_matrix
        group["Yₜ"] = cpu_column_vector
    end
end
