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
to compile and run. So, the dual number components are split into partitions
with a maximum size of `max_simultaneous_derivatives`, and we call
`implicit_tendency!` once for each partition. That is, if the partition size is
``s``, then the first partition evaluates the coefficients of ``ε₁`` through
``εₛ``, the second evaluates the coefficients of ``εₛ₊₁`` through ``ε₂ₛ``, and so
on until ``εₙ``. The default partition size is 32.
"""
struct AutoDenseJacobian{S} <: JacobianAlgorithm end
AutoDenseJacobian(max_simultaneous_derivatives = 32) =
    AutoDenseJacobian{max_simultaneous_derivatives}()

# The number of derivatives computed simultaneously by AutoDenseJacobian.
max_simultaneous_derivatives(::AutoDenseJacobian{S}) where {S} = S

function jacobian_cache(alg::AutoDenseJacobian, Y, atmos)
    FT = eltype(Y)
    DA = ClimaComms.array_type(Y)

    precomputed = implicit_precomputed_quantities(Y, atmos)
    scratch = implicit_temporary_quantities(Y, atmos)

    FT_dual = ForwardDiff.Dual{Jacobian, FT, max_simultaneous_derivatives(alg)}
    precomputed_dual = replace_parent_eltype(precomputed, FT_dual)
    scratch_dual = replace_parent_eltype(scratch, FT_dual)
    Y_dual = replace_parent_eltype(Y, FT_dual)
    Yₜ_dual = similar(Y_dual)

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
        precomputed_dual,
        scratch_dual,
        Y_dual,
        Yₜ_dual,
        column_matrices,
        column_lu_factors,
        column_lu_vectors,
        I_matrix,
        N_val = Val(N), # Save N as a statically inferrable parameter.
    )
end

function update_column_matrices!(alg::AutoDenseJacobian, cache, Y, p, t)
    (; precomputed_dual, scratch_dual, Y_dual, Yₜ_dual, column_matrices) = cache
    device = ClimaComms.device(Y.c)
    column_indices = column_index_iterator(Y)
    scalar_names = scalar_field_names(Y)
    field_vector_indices = field_vector_index_iterator(Y)
    p_dual = append_to_atmos_cache(p, precomputed_dual, scratch_dual)

    jacobian_index_to_Y_index_map_partitions = Iterators.partition(
        enumerate(field_vector_indices),
        max_simultaneous_derivatives(alg),
    )
    for jacobian_index_to_Y_index_map_partition in
        ClimaComms.threadable(device, jacobian_index_to_Y_index_map_partitions)

        # Add a unique ε to each value in Y that is part of this partition. With
        # Y_col and Yᴰ_col denoting the columns of Y and Y_dual at column_index,
        # set Yᴰ_col to Y_col + I[:, jacobian_column_indices] * εs, where I is
        # the identity matrix for Y_col (i.e., the value of ∂Y_col/∂Y_col), εs
        # is a vector of max_simultaneous_derivatives(alg) dual number
        # components, and jacobian_column_indices is equal to
        # first.(jacobian_index_to_Y_index_map_partition).
        Y_dual .= Y
        ClimaComms.@threaded device begin
            # On multithreaded devices, use one thread for each dual number.
            for column_index in column_indices,
                (diagonal_entry_ε_index, (_, (scalar_index, level_index))) in
                enumerate(jacobian_index_to_Y_index_map_partition)

                n_εs_val = Val(max_simultaneous_derivatives(alg))
                ε_coefficients = ntuple(==(diagonal_entry_ε_index), n_εs_val)
                unrolled_applyat(scalar_index, scalar_names) do name
                    field = MatrixFields.get_field(Y_dual, name)
                    @inbounds point(field, level_index, column_index...)[] +=
                        ForwardDiff.Dual{Jacobian}(0, ε_coefficients)
                end
            end
        end

        # Compute this partition of ∂p/∂Y and ∂Yₜ/∂Y.
        set_implicit_precomputed_quantities!(Y_dual, p_dual, t)
        implicit_tendency!(Yₜ_dual, Y_dual, p_dual, t)

        # Copy this partition of ∂Yₜ/∂Y into column_matrices. With Yₜ_col and
        # Yₜᴰ_col denoting the columns of Yₜ and Yₜ_dual at column_index, and
        # with col_matrix denoting the matrix at the corresponding matrix_index
        # in column_matrices, copy the coefficients of the εs in Yₜᴰ_col into
        # col_matrix, where the previous steps have set Yₜᴰ_col to
        # Yₜ_col + (∂Yₜ_col/∂Y_col)[:, jacobian_column_indices] * εs. In
        # other words, set col_matrix[jacobian_row_index, jacobian_column_index]
        # to ∂Yₜ_col[jacobian_row_index]/∂Y_col[jacobian_column_index],
        # obtaining this derivative from the coefficient of
        # εs[jacobian_column_ε_index] in Yₜᴰ_col[jacobian_row_index], where
        # jacobian_column_ε_index is the index of jacobian_column_index in
        # jacobian_column_indices.
        ClimaComms.@threaded device begin
            # On multithreaded devices, use one thread for each dual number.
            for (matrix_index, column_index) in enumerate(column_indices),
                (jacobian_row_index, (scalar_index, level_index)) in
                enumerate(field_vector_indices)

                dual_number =
                    unrolled_applyat(scalar_index, scalar_names) do name
                        field = MatrixFields.get_field(Yₜ_dual, name)
                        @inbounds point(field, level_index, column_index...)[]
                    end
                ε_coefficients = ForwardDiff.partials(dual_number)
                for (jacobian_column_ε_index, (jacobian_column_index, _)) in
                    enumerate(jacobian_index_to_Y_index_map_partition)
                    cartesian_index = (
                        jacobian_row_index,
                        jacobian_column_index,
                        matrix_index,
                    )
                    @inbounds column_matrices[cartesian_index...] =
                        ε_coefficients[jacobian_column_ε_index]
                end
            end
        end
    end
end

function update_jacobian!(alg::AutoDenseJacobian, cache, Y, p, dtγ, t)
    (; column_matrices, column_lu_factors, I_matrix, N_val) = cache
    device = ClimaComms.device(Y.c)

    # Set column_matrices to ∂Yₜ/∂Y.
    update_column_matrices!(alg, cache, Y, p, t)

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
    vector_index_to_field_vector_index_map =
        enumerate(field_vector_index_iterator(ΔY))

    # Copy all scalar values from R into column_lu_vectors.
    ClimaComms.@threaded device begin
        # On multithreaded devices, use one thread for each number.
        for (vector_index, column_index) in enumerate(column_indices),
            (scalar_level_index, (scalar_index, level_index)) in
            vector_index_to_field_vector_index_map

            number = unrolled_applyat(scalar_index, scalar_names) do name
                field = MatrixFields.get_field(R, name)
                @inbounds point(field, level_index, column_index...)[]
            end
            @inbounds column_lu_vectors[scalar_level_index, vector_index] =
                number
        end
    end

    # Solve L * U * ΔY = R for ΔY in each column.
    parallel_lu_solve!(device, column_lu_vectors, column_lu_factors, N_val)

    # Copy all scalar values from column_lu_vectors into ΔY.
    ClimaComms.@threaded device begin
        # On multithreaded devices, use one thread for each number.
        for (vector_index, column_index) in enumerate(column_indices),
            (scalar_level_index, (scalar_index, level_index)) in
            vector_index_to_field_vector_index_map

            @inbounds number =
                column_lu_vectors[scalar_level_index, vector_index]
            unrolled_applyat(scalar_index, scalar_names) do name
                field = MatrixFields.get_field(ΔY, name)
                @inbounds point(field, level_index, column_index...)[] = number
            end
        end
    end
end

function first_column_block_arrays(alg::AutoDenseJacobian, Y, p, dtγ, t)
    scalar_names = scalar_field_names(Y)
    field_vector_indices = field_vector_index_iterator(Y)
    column_Y = first_column_view(Y)
    column_p = first_column_view(p)
    column_cache = jacobian_cache(alg, column_Y, p.atmos)

    update_column_matrices!(alg, column_cache, column_Y, column_p, t)
    column_∂R_∂Y = dtγ .* column_cache.column_matrices .- column_cache.I_matrix

    block_arrays = Dict()
    for block_key in Iterators.product(scalar_names, scalar_names)
        block_jacobian_row_index_to_Yₜ_index_map =
            Iterators.filter(enumerate(field_vector_indices)) do index_pair
                (_, (scalar_index, _)) = index_pair
                scalar_names[scalar_index] == block_key[1]
            end
        block_jacobian_column_index_to_Y_index_map =
            Iterators.filter(enumerate(field_vector_indices)) do index_pair
                (_, (scalar_index, _)) = index_pair
                scalar_names[scalar_index] == block_key[2]
            end
        block_view_indices = (
            map(first, block_jacobian_row_index_to_Yₜ_index_map),
            map(first, block_jacobian_column_index_to_Y_index_map),
        )
        block_arrays[block_key] =
            Array(view(column_∂R_∂Y, block_view_indices..., 1))
    end
    return block_arrays
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

On GPU, an in-place version is used (no per-thread MArray) to stay within
CUDA per-thread local memory limits when N is large (e.g. N > 128).
"""
function parallel_lu_factorize!(device, matrices, n_val::Val{N}) where {N}
    if device isa ClimaComms.CUDADevice
        _parallel_lu_factorize_gpu!(device, matrices, n_val)
    else
        _parallel_lu_factorize_cpu!(device, matrices, n_val)
    end
end

function _parallel_lu_factorize_cpu!(device, matrices, ::Val{N}) where {N}
    n_matrices = size(matrices, 3)
    @assert size(matrices, 1) == size(matrices, 2) == N
    ClimaComms.@threaded device for matrix_index in 1:n_matrices
        FT = eltype(matrices)
        matrix = MArray{Tuple{N, N}, FT}(undef)
        @inbounds for j in 1:N, i in 1:N
            matrix[i, j] = matrices[i, j, matrix_index]
        end
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
        @inbounds for j in 1:N, i in 1:N
            matrices[i, j, matrix_index] = matrix[i, j]
        end
    end
end

function _parallel_lu_factorize_gpu!(device, matrices, ::Val{N}) where {N}
    n_matrices = size(matrices, 3)
    @assert size(matrices, 1) == size(matrices, 2) == N
    # In-place LU (Doolittle): no per-thread N×N copy, stays under CUDA local memory.
    ClimaComms.@threaded device for matrix_index in 1:n_matrices
        @inbounds for k in 1:N
            pivot = matrices[k, k, matrix_index]
            isnan(pivot) && error("LU error: NaN on diagonal")
            iszero(pivot) && error("LU error: 0 on diagonal")
            inv_pivot = inv(pivot)
            for i in (k + 1):N
                matrices[i, k, matrix_index] *= inv_pivot
            end
            for j in (k + 1):N, i in (k + 1):N
                matrices[i, j, matrix_index] -= matrices[i, k, matrix_index] * matrices[k, j, matrix_index]
            end
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

On GPU, an in-place version is used so each thread only touches global memory
(no per-thread N×N matrix copy), staying within CUDA local memory limits.
"""
function parallel_lu_solve!(device, vectors, matrices, n_val::Val{N}) where {N}
    if device isa ClimaComms.CUDADevice
        _parallel_lu_solve_gpu!(device, vectors, matrices, n_val)
    else
        _parallel_lu_solve_cpu!(device, vectors, matrices, n_val)
    end
end

function _parallel_lu_solve_cpu!(device, vectors, matrices, ::Val{N}) where {N}
    n_matrices = size(matrices, 3)
    @assert size(vectors, 1) == N
    @assert size(vectors, 2) == n_matrices
    @assert size(matrices, 1) == size(matrices, 2) == N
    ClimaComms.@threaded device for vector_and_matrix_index in 1:n_matrices
        FT = eltype(matrices)
        vector = MArray{Tuple{N}, FT}(undef)
        matrix = MArray{Tuple{N, N}, FT}(undef)
        @inbounds for i in 1:N
            vector[i] = vectors[i, vector_and_matrix_index]
        end
        @inbounds for j in 1:N, i in 1:N
            matrix[i, j] = matrices[i, j, vector_and_matrix_index]
        end
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
        @inbounds for i in 1:N
            vectors[i, vector_and_matrix_index] = vector[i]
        end
    end
end

function _parallel_lu_solve_gpu!(device, vectors, matrices, ::Val{N}) where {N}
    n_matrices = size(matrices, 3)
    @assert size(vectors, 1) == N
    @assert size(vectors, 2) == n_matrices
    @assert size(matrices, 1) == size(matrices, 2) == N
    # In-place solve: L y = b (forward), then U x = y (back). No per-thread N×N copy.
    ClimaComms.@threaded device for idx in 1:n_matrices
        @inbounds for i in 2:N
            s = zero(eltype(vectors))
            for j in 1:(i - 1)
                s += matrices[i, j, idx] * vectors[j, idx]
            end
            vectors[i, idx] -= s
        end
        @inbounds vectors[N, idx] /= matrices[N, N, idx]
        @inbounds for i in (N - 1):-1:1
            s = zero(eltype(vectors))
            for j in (i + 1):N
                s += matrices[i, j, idx] * vectors[j, idx]
            end
            vectors[i, idx] = (vectors[i, idx] - s) / matrices[i, i, idx]
        end
    end
end
