import ForwardDiff
import LinearAlgebra: diagind

import ClimaComms
import ClimaCore.DataLayouts: MArray

"""
    AutoDenseJacobian()

A `JacobianAlgorithm` that computes the `Jacobian` using forward-mode automatic
differentiation and inverts it using LU factorization, without making any
assumptions about sparsity structure.

To automatically compute the derivative of `implicit_tendency!` with respect to
`Y`, we first create copies of `Y`, `p.precomputed`, and `p.scratch` in which
every floating-point number is replaced by a `ForwardDiff.Dual` number. A dual
number can be expressed as ``x + y₁ε₁ + y₂ε₂ + ... + yₙεₙ``, where ``x`` and
``yᵢ`` are floating-point numbers and ``εᵢ`` is a hyperreal number that
satisfies ``εᵢεⱼ = 0``. When each ``εᵢ`` is initialized to 1 on the ``i``-th
value in each column of `Y`, and every other dual number component is
initialized to 0, calling `implicit_tendency!` on this dual version of `Y`
gives us a dense representation of the Jacobian matrix, with the ``j``-th
component of the ``i``-th dual number in each column representing the derivative
of the ``i``-th value in `Yₜ` with respect to the ``j``-th value in `Y`.

When the number of values in each column is very large, computing the entire
dense matrix in a single evaluation of `implicit_tendency!` can be too expensive
to compile and run. So, we split the dual number components into batches with a
fixed maximum size, and we call `implicit_tendency!` once for each batch.
"""
struct AutoDenseJacobian <: JacobianAlgorithm end

exact_jacobian_batch_size() = 32 # number of derivatives computed simultaneously

function jacobian_cache(::AutoDenseJacobian, Y, atmos)
    FT = eltype(Y)
    DA = ClimaComms.array_type(Y)

    FT_dual =
        ForwardDiff.Dual{AutoDenseJacobian, FT, exact_jacobian_batch_size()}
    Y_dual = replace_parent_type(Y, FT_dual)
    Yₜ_dual = similar(Y_dual)
    precomputed_dual =
        replace_parent_type(implicit_precomputed_quantities(Y, atmos), FT_dual)
    scratch_dual = replace_parent_type(temporary_quantities(Y, atmos), FT_dual)

    N = length(Fields.column(Y, 1, 1, 1))
    n_columns = Fields.ncolumns(Y.c)
    column_matrices = DA{FT}(undef, N, N, n_columns)
    column_lu_factors = copy(column_matrices)
    column_lu_solve_vectors = DA{FT}(undef, N, n_columns)

    # LinearAlgebra.I does not support broadcasting, so we need a workaround.
    I_column_matrix = DA{FT}(undef, N, N)
    I_column_matrix .= 0
    I_column_matrix[diagind(I_column_matrix)] .= 1
    I_matrix = reshape(I_column_matrix, N, N, 1)

    return (;
        Y_dual,
        Yₜ_dual,
        precomputed_dual,
        scratch_dual,
        column_matrices,
        column_lu_factors,
        column_lu_solve_vectors,
        I_matrix,
        N_val = Val(N), # Save N as a statically inferrable parameter.
    )
end

function update_jacobian_skip_factorizing!(
    ::AutoDenseJacobian,
    cache,
    Y,
    p,
    dtγ,
    t,
)
    (; Y_dual, Yₜ_dual, precomputed_dual, scratch_dual, column_matrices) = cache
    batch_size = exact_jacobian_batch_size()

    p_dual_args = ntuple(Val(fieldcount(typeof(p)))) do cache_field_index
        cache_field_name = fieldname(typeof(p), cache_field_index)
        if cache_field_name == :precomputed
            (; p.precomputed..., precomputed_dual...)
        elseif cache_field_name == :scratch
            scratch_dual
        else
            getfield(p, cache_field_index)
        end
    end
    p_dual = AtmosCache(p_dual_args...)

    Y_dual_scalar_levels = scalar_level_iterator(Y_dual)
    level_batches =
        Iterators.partition(enumerate(Y_dual_scalar_levels), batch_size)

    Y_dual .= Y
    for level_batch in level_batches
        for (partial_index, (_, Y_dual_scalar_level)) in enumerate(level_batch)
            partials = ntuple(i -> i == partial_index ? 1 : 0, Val(batch_size))
            parent(Y_dual_scalar_level) .+=
                ForwardDiff.Dual{AutoDenseJacobian}(0, partials...)
        end # Add a unique ε to Y_dual for each combination of scalar and level.
        set_implicit_precomputed_quantities!(Y_dual, p_dual, t) # Compute ∂p/∂Y.
        implicit_tendency!(Yₜ_dual, Y_dual, p_dual, t) # Compute ∂Yₜ/∂Y.
        for (partial_index, (ε_index, _)) in enumerate(level_batch)
            ε_column_vectors = view(column_matrices, :, ε_index, :)
            column_vectors_to_field_vector(ε_column_vectors, Y) .=
                getindex.(ForwardDiff.partials.(Yₜ_dual), partial_index)
        end # Copy the new values of ∂Yₜ/∂Y into column_matrices.
        Y_dual .= ForwardDiff.value.(Y_dual) # Drop this batch's εs from Y_dual.
    end
end

# TODO: Fuse update_jacobian! and invert_jacobian! so that the full dense matrix
# does not need to be stored in global memory.

function update_jacobian!(::AutoDenseJacobian, cache, Y, p, dtγ, t)
    (; column_matrices, column_lu_factors, I_matrix, N_val) = cache
    device = ClimaComms.device(Y.c)
    update_jacobian_skip_factorizing!(AutoDenseJacobian(), cache, Y, p, dtγ, t)
    column_lu_factors .= dtγ .* column_matrices .- I_matrix
    lu_factorize_columns!(device, column_lu_factors, N_val)
end

function invert_jacobian!(::AutoDenseJacobian, cache, ΔY, R)
    (; column_lu_solve_vectors, column_lu_factors, N_val) = cache
    device = ClimaComms.device(ΔY.c)
    column_vectors_to_field_vector(column_lu_solve_vectors, R) .= R
    lu_solve_columns!(device, column_lu_solve_vectors, column_lu_factors, N_val)
    ΔY .= column_vectors_to_field_vector(column_lu_solve_vectors, ΔY)
end

# Set the derivative of `sqrt(x)` to `iszero(x) ? zero(x) : inv(2 * sqrt(x))` in
# order to properly handle derivatives of `x * sqrt(x)`. Without this change,
# the derivative of `x * sqrt(x)` is `NaN` when `x` is zero. This method
# specializes on the tag `AutoDenseJacobian` because not specializing on any tag
# overwrites the generic method for `Dual` in `ForwardDiff` and breaks
# precompilation, while specializing on the default tag `Nothing` causes the
# type piracy Aqua test to fail.
@inline function Base.sqrt(d::ForwardDiff.Dual{AutoDenseJacobian})
    tag = Val{AutoDenseJacobian}()
    x = ForwardDiff.value(d)
    partials = ForwardDiff.partials(d)
    val = sqrt(x)
    deriv = iszero(x) ? zero(x) : inv(2 * val)
    return ForwardDiff.dual_definition_retval(tag, val, deriv, partials)
end

function lu_factorize_columns!(device, column_matrices, ::Val{N}) where {N}
    n_columns = size(column_matrices, 3)
    @assert size(column_matrices, 1) == size(column_matrices, 2) == N
    ClimaComms.@threaded device for column_index in 1:n_columns
        # Copy each column into local memory to minimize global memory reads.
        FT = eltype(column_matrices)
        column_matrix = MArray{Tuple{N, N}, FT}(undef)
        @inbounds for j in 1:N, i in 1:N # Copy entries in column-major order.
            column_matrix[i, j] = column_matrices[i, j, column_index]
        end

        # Overwrite the local column matrix with its LU factorization.
        @inbounds for k in 1:N
            isnan(column_matrix[k, k]) && error("LU error: NaN on diagonal")
            iszero(column_matrix[k, k]) && error("LU error: 0 on diagonal")
            inverse_of_diagonal_value = inv(column_matrix[k, k])
            for i in (k + 1):N
                column_matrix[i, k] *= inverse_of_diagonal_value
            end
            for j in (k + 1):N, i in (k + 1):N
                column_matrix[i, j] -= column_matrix[i, k] * column_matrix[k, j]
            end
        end

        # Copy the new column matrix back into global memory.
        @inbounds for j in 1:N, i in 1:N
            column_matrices[i, j, column_index] = column_matrix[i, j]
        end
    end
end

function lu_solve_columns!(
    device,
    column_vectors,
    column_matrices,
    ::Val{N},
) where {N}
    n_columns = size(column_matrices, 3)
    @assert size(column_vectors, 1) == N
    @assert size(column_vectors, 2) == n_columns
    @assert size(column_matrices, 1) == size(column_matrices, 2) == N
    ClimaComms.@threaded device for column_index in 1:n_columns
        # Copy each column into local memory to minimize global memory reads.
        FT = eltype(column_matrices)
        column_vector = MArray{Tuple{N}, FT}(undef)
        column_matrix = MArray{Tuple{N, N}, FT}(undef)
        @inbounds for i in 1:N
            column_vector[i] = column_vectors[i, column_index]
        end
        @inbounds for j in 1:N, i in 1:N # Copy entries in column-major order.
            column_matrix[i, j] = column_matrices[i, j, column_index]
        end

        # Overwrite the local column vector with the linear system's solution.
        @inbounds for i in 2:N
            l_row_i_dot_product = zero(FT)
            for j in 1:(i - 1)
                l_row_i_dot_product += column_matrix[i, j] * column_vector[j]
            end
            column_vector[i] -= l_row_i_dot_product
        end
        @inbounds column_vector[N] /= column_matrix[N, N]
        @inbounds for i in (N - 1):-1:1
            u_row_i_dot_product = zero(FT)
            for j in (i + 1):N
                u_row_i_dot_product += column_matrix[i, j] * column_vector[j]
            end
            column_vector[i] -= u_row_i_dot_product
            column_vector[i] /= column_matrix[i, i]
        end

        # Copy the new column vector back into global memory.
        @inbounds for i in 1:N
            column_vectors[i, column_index] = column_vector[i]
        end
    end
end
