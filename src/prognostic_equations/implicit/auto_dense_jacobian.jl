import ForwardDiff
import LinearAlgebra: diagind

import ClimaComms
import ClimaCore.DataLayouts: MArray

"""
    AutoDenseJacobian([max_simultaneous_derivatives])

A [`JacobianAlgorithm`](@ref) that computes the Jacobian using forward-mode
automatic differentiation, without making any assumptions about sparsity
structure. After the dense matrix for each spatial column is updated,
[`parallel_lu_factorize!`](@ref) computes its LU factorization in parallel
across all columns. The linear solver is also run in parallel with
[`parallel_lu_solve!`](@ref).

To automatically compute the derivative of `implicit_tendency!` with respect to
`Y`, we first create copies of `Y`, `p.precomputed`, and `p.scratch` in which
every floating-point number is replaced by a
[dual number](https://juliadiff.org/ForwardDiff.jl/stable/dev/how_it_works/)
from `ForwardDiff.jl`. A dual number can be expressed as
``Xᴰ = X + ε₁x₁ + ε₂x₂ + ... + εₙxₙ``, where ``X`` and ``xᵢ`` are floating-point
numbers, and where ``εᵢ`` is a hyperreal number that satisfies ``εᵢεⱼ = 0``. If
the ``i``-th value in dual column state ``Yᴰ`` is set to ``Yᴰᵢ = Yᵢ + 1εᵢ``,
where ``Yᵢ`` is the ``i``-th value in the column state ``Y``, then evaluating
the implicit tendency of the dual column state generates a dense representation
of the Jacobian matrix ``∂T/∂Y``. Specifically, the ``i``-th value in the dual
column tendency ``Tᴰ = T(Yᴰ)`` is ``Tᴰᵢ = Tᵢ + (∂Tᵢ/∂Y₁)ε₁ + ... + (∂Tᵢ/∂Yₙ)εₙ``,
where ``Tᵢ`` is the ``i``-th value in the column tendency ``T(Y)``, and where
``n`` is the number of values in ``Y``. In other words, the entry in the
``i``-th row and ``j``-th column of the matrix ``∂T/∂Y`` is the coefficient of
``εⱼ`` in ``Tᴰᵢ``. The size of the dense matrix scales as ``O(n^2)``, leading to
very large memory requirements at higher vertical resolutions.

When the number of values in each column is very large, computing the entire
dense matrix in a single evaluation of `implicit_tendency!` can be too expensive
to compile and run. So, the dual number components are split into batches with a
maximum size of `max_simultaneous_derivatives`, and we call `implicit_tendency!`
once for each batch. That is, if the batch size is ``s``, then the first batch
evaluates the coefficients of ``ε₁`` through ``εₛ``, the second evaluates the
coefficients of ``εₛ₊₁`` through ``ε₂ₛ``, and so on until ``εₙ``. The default
batch size is 32.
"""
struct AutoDenseJacobian{S} <: JacobianAlgorithm end
AutoDenseJacobian(max_simultaneous_derivatives = 32) =
    AutoDenseJacobian{max_simultaneous_derivatives}()

# The number of derivatives computed simultaneously by AutoDenseJacobian.
max_simultaneous_derivatives(::AutoDenseJacobian{S}) where {S} = S

function jacobian_cache(alg::AutoDenseJacobian, Y, atmos)
    FT = eltype(Y)
    DA = ClimaComms.array_type(Y)

    FT_dual = ForwardDiff.Dual{Jacobian, FT, max_simultaneous_derivatives(alg)}
    Y_dual = replace_parent_type(Y, FT_dual)
    Yₜ_dual = similar(Y_dual)
    precomputed_dual =
        replace_parent_type(implicit_precomputed_quantities(Y, atmos), FT_dual)
    scratch_dual = replace_parent_type(temporary_quantities(Y, atmos), FT_dual)

    N = length(Fields.column(Y, 1, 1, 1))
    n_columns = Fields.ncolumns(Y.c)
    column_matrices = DA{FT}(undef, N, N, n_columns)
    column_lu_factors = copy(column_matrices)
    column_lu_vectors = DA{FT}(undef, N, n_columns)

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
        column_lu_vectors,
        I_matrix,
        N_val = Val(N), # Save N as a statically inferrable parameter.
    )
end

function update_column_matrices!(alg::AutoDenseJacobian, cache, Y, p, dtγ, t)
    (; Y_dual, Yₜ_dual, precomputed_dual, scratch_dual, column_matrices) = cache
    device = ClimaComms.device(Y.c)
    column_indices = column_index_iterator(Y)
    scalar_names = scalar_field_names(Y)
    scalar_level_indices = scalar_level_index_pairs(Y)
    batch_size = max_simultaneous_derivatives(alg)
    batch_size_val = Val(batch_size)

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

    batches = Iterators.partition(scalar_level_indices, batch_size)
    for batch_scalar_level_indices in ClimaComms.threadable(device, batches)
        Y_dual .= Y

        # Add a unique ε to Y for each scalar level index in this batch. With
        # Y_col and Yᴰ_col denoting the columns of Y and Y_dual at column_index,
        # set Yᴰ_col to Y_col + I[:, batch_scalar_level_indices] * εs, where I
        # is the identity matrix for Y_col (i.e., the value of ∂Y_col/∂Y_col),
        # εs is a vector of batch_size dual number components, and
        # batch_scalar_level_indices are the batch's indices into Y_col.
        ClimaComms.@threaded device begin
            # On multithreaded devices, assign one thread to each combination of
            # spatial column index and scalar level index in this batch.
            for column_index in column_indices,
                (ε_index, (_, (scalar_index, level_index))) in
                enumerate(batch_scalar_level_indices)

                Y_partials = ntuple(i -> i == ε_index ? 1 : 0, batch_size_val)
                Y_dual_increment = ForwardDiff.Dual{Jacobian}(0, Y_partials...)
                unrolled_applyat(scalar_index, scalar_names) do name
                    field = MatrixFields.get_field(Y_dual, name)
                    @inbounds point(field, level_index, column_index...)[] +=
                        Y_dual_increment
                end
            end
        end

        # Compute this batch's portions of ∂p/∂Y and ∂Yₜ/∂Y.
        set_implicit_precomputed_quantities!(Y_dual, p_dual, t)
        implicit_tendency!(Yₜ_dual, Y_dual, p_dual, t)

        # Copy this batch's portion of ∂Yₜ/∂Y into column_matrices. With Yₜ_col
        # and Yₜᴰ_col denoting the columns of Yₜ and Yₜ_dual at column_index, and
        # with col_matrix denoting the matrix at the corresponding matrix_index
        # in column_matrices, copy the coefficients of the εs in Yₜᴰ_col into
        # col_matrix, where the previous steps have set Yₜᴰ_col to
        # Yₜ_col + (∂Yₜ_col/∂Y_col)[:, batch_scalar_level_indices] * εs.
        # Specifically, set col_matrix[scalar_level_index1, scalar_level_index2]
        # to ∂Yₜ_col[scalar_level_index1]/∂Y_col[scalar_level_index2], obtaining
        # this derivative from the coefficient of εs[ε_index] in
        # Yₜᴰ_col[scalar_level_index1], where ε_index is the index of
        # scalar_level_index2 in batch_scalar_level_indices. After all batches
        # have been processed, col_matrix is the full Jacobian ∂Yₜ_col/∂Y_col.
        ClimaComms.@threaded device begin
            # On multithreaded devices, assign one thread to each combination of
            # spatial column index and scalar level index.
            for (matrix_index, column_index) in enumerate(column_indices),
                (scalar_level_index1, (scalar_index1, level_index1)) in
                scalar_level_indices

                Yₜ_dual_value =
                    unrolled_applyat(scalar_index1, scalar_names) do name
                        field = MatrixFields.get_field(Yₜ_dual, name)
                        @inbounds point(field, level_index1, column_index...)[]
                    end
                Yₜ_partials = ForwardDiff.partials(Yₜ_dual_value)
                for (ε_index, (scalar_level_index2, _)) in
                    enumerate(batch_scalar_level_indices)
                    cartesian_index =
                        (scalar_level_index1, scalar_level_index2, matrix_index)
                    @inbounds column_matrices[cartesian_index...] =
                        Yₜ_partials[ε_index]
                end
            end
        end
    end
end

function update_jacobian!(alg::AutoDenseJacobian, cache, Y, p, dtγ, t)
    (; column_matrices, column_lu_factors, I_matrix, N_val) = cache
    device = ClimaComms.device(Y.c)

    # Set column_matrices to ∂Yₜ/∂Y.
    update_column_matrices!(alg, cache, Y, p, dtγ, t)

    # Set column_lu_factors to ∂R/∂Y = dtγ * ∂Yₜ/∂Y - I, where R is the residual
    # of the implicit equation.
    column_lu_factors .= dtγ .* column_matrices .- I_matrix

    # Replace each matrix in column_lu_factors with triangular L and U matrices.
    parallel_lu_factorize!(device, column_lu_factors, N_val)
end

function invert_jacobian!(::AutoDenseJacobian, cache, ΔY, R)
    (; column_lu_vectors, column_lu_factors, N_val) = cache
    device = ClimaComms.device(ΔY.c)
    column_indices = column_index_iterator(ΔY)
    scalar_names = scalar_field_names(ΔY)
    scalar_level_indices = scalar_level_index_pairs(ΔY)

    # Copy all scalar values from R into column_lu_vectors.
    ClimaComms.@threaded device begin
        # On multithreaded devices, assign one thread to each index into R.
        for (vector_index, column_index) in enumerate(column_indices),
            (scalar_level_index, (scalar_index, level_index)) in
            scalar_level_indices

            value = unrolled_applyat(scalar_index, scalar_names) do name
                field = MatrixFields.get_field(R, name)
                @inbounds point(field, level_index, column_index...)[]
            end
            @inbounds column_lu_vectors[scalar_level_index, vector_index] =
                value
        end
    end

    # Solve L * U * ΔY = R for ΔY in each column.
    parallel_lu_solve!(device, column_lu_vectors, column_lu_factors, N_val)

    # Copy all scalar values from column_lu_vectors into ΔY.
    ClimaComms.@threaded device begin
        # On multithreaded devices, assign one thread to each index into ΔY.
        for (vector_index, column_index) in enumerate(column_indices),
            (scalar_level_index, (scalar_index, level_index)) in
            scalar_level_indices

            @inbounds value =
                column_lu_vectors[scalar_level_index, vector_index]
            unrolled_applyat(scalar_index, scalar_names) do name
                field = MatrixFields.get_field(ΔY, name)
                @inbounds point(field, level_index, column_index...)[] = value
            end
        end
    end
end

"""
    parallel_lu_factorize!(device, matrices, ::Val{N})

Runs a parallel LU factorization algorithm on the specified `device`. If each
slice `matrices[1:N, 1:N, i]` represents a matrix ``Mᵢ``, this function
overwrites it with the lower triangular matrix ``Lᵢ`` and the upper triangular
matrix ``Uᵢ``, where ``Mᵢ = Lᵢ * Uᵢ``. The value of `N` must be wrapped in a
`Val` to ensure that it is statically inferrable, which allows the LU
factorization to avoid dynamic local memory allocations.

The runtime of this algorithm scales as ``O(N^3)``.
"""
function parallel_lu_factorize!(device, matrices, ::Val{N}) where {N}
    n_matrices = size(matrices, 3)
    @assert size(matrices, 1) == size(matrices, 2) == N
    ClimaComms.@threaded device for matrix_index in 1:n_matrices
        # Copy each column into local memory to minimize global memory reads.
        FT = eltype(matrices)
        matrix = MArray{Tuple{N, N}, FT}(undef)
        @inbounds for j in 1:N, i in 1:N # Copy entries in column-major order.
            matrix[i, j] = matrices[i, j, matrix_index]
        end

        # Overwrite the local column matrix with its LU factorization.
        @inbounds for k in 1:N
            isnan(matrix[k, k]) && error("LU error: NaN on diagonal")
            iszero(matrix[k, k]) && error("LU error: 0 on diagonal")
            inverse_of_diagonal_value = inv(matrix[k, k])
            for i in (k + 1):N
                matrix[i, k] *= inverse_of_diagonal_value
            end
            for j in (k + 1):N, i in (k + 1):N
                matrix[i, j] -= matrix[i, k] * matrix[k, j]
            end
        end

        # Copy the new column matrix back into global memory.
        @inbounds for j in 1:N, i in 1:N
            matrices[i, j, matrix_index] = matrix[i, j]
        end
    end
end

"""
    parallel_lu_solve!(device, vectors, matrices, ::Val{N})

Runs a parallel LU solver algorithm on the specified `device`. If each slice
`vectors[1:N, i]` represents a vector ``vᵢ``, and if each slice
`matrices[1:N, 1:N, i]` represents a matrix ``Lᵢ * Uᵢ`` that was factorized by
[`parallel_lu_factorize!`](@ref), this function overwrites the slice
`vectors[1:N, i]` with ``(Lᵢ * Uᵢ)⁻¹ * vᵢ``. The value of `N` must be wrapped in
a `Val` to ensure that it is statically inferrable, which allows the LU solver
to avoid dynamic local memory allocations.

The runtime of this algorithm scales as ``O(N^2)``.
"""
function parallel_lu_solve!(device, vectors, matrices, ::Val{N}) where {N}
    n_matrices = size(matrices, 3)
    @assert size(vectors, 1) == N
    @assert size(vectors, 2) == n_matrices
    @assert size(matrices, 1) == size(matrices, 2) == N
    ClimaComms.@threaded device for vector_and_matrix_index in 1:n_matrices
        # Copy each column into local memory to minimize global memory reads.
        FT = eltype(matrices)
        vector = MArray{Tuple{N}, FT}(undef)
        matrix = MArray{Tuple{N, N}, FT}(undef)
        @inbounds for i in 1:N
            vector[i] = vectors[i, vector_and_matrix_index]
        end
        @inbounds for j in 1:N, i in 1:N # Copy entries in column-major order.
            matrix[i, j] = matrices[i, j, vector_and_matrix_index]
        end

        # Overwrite the local column vector with the linear system's solution.
        @inbounds for i in 2:N
            l_row_i_dot_product = zero(FT)
            for j in 1:(i - 1)
                l_row_i_dot_product += matrix[i, j] * vector[j]
            end
            vector[i] -= l_row_i_dot_product
        end
        @inbounds vector[N] /= matrix[N, N]
        @inbounds for i in (N - 1):-1:1
            u_row_i_dot_product = zero(FT)
            for j in (i + 1):N
                u_row_i_dot_product += matrix[i, j] * vector[j]
            end
            vector[i] -= u_row_i_dot_product
            vector[i] /= matrix[i, i]
        end

        # Copy the new column vector back into global memory.
        @inbounds for i in 1:N
            vectors[i, vector_and_matrix_index] = vector[i]
        end
    end
end
