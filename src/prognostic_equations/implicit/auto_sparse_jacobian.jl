import SparseMatrixColorings

"""
    AutoSparseJacobian(sparse_alg)

TODO
"""
struct AutoSparseJacobian{A} <: JacobianAlgorithm
    sparse_alg::A
end

#=
Cache:
1. Allocate ∂T/∂Y and ∂R/∂Y, get solution arrays
2. Solve coloring problem, get the map

Update:
1. Initialize Y^D
2. Compute p^D and T^D
3. Set ∂T/∂Y (hard!)
4. Rescale to get ∂R/∂Y

Solve:
1. Run sparse solver
=#

function jacobian_cache(alg::AutoSparseJacobian, Y, atmos)
    (; matrix) = jacobian_cache(alg.sparse_alg, Y, atmos) # Allocate ∂R/∂Y.
    tendency_matrix = matrix .+ one(matrix) # Allocate ∂T/∂Y.

    device = ClimaComms.device(Y.c)
    DA = ClimaComms.array_type(Y)
    
    scalar_names = scalar_field_names(Y) # contains f
    scalar_level_indices = scalar_level_index_pairs(Y) # contains n => (f, v)
    sparse_matrix = MatrixFields.scalar_fieldmatrix(tendency_matrix, Y)

    N = length(scalar_level_indices)
    sparsity_pattern = Array{Bool}(undef, N, N)

    sparsity_pattern .= false

    for ((row_name, column_name), sparse_matrix_block) in sparse_matrix
        sparse_matrix_block isa Fields.Field || continue
        
        n_rows, _, lower_band, upper_band =
            MatrixFields.band_matrix_info(sparse_matrix_block)

        scalar_level_indices_for_row =
            filter(scalar_level_indices) do (_, (scalar_index, _))
                row_name == scalar_names[scalar_index]
            end
        scalar_level_indices_for_column =
            filter(scalar_level_indices) do (_, (scalar_index, _))
                column_name == scalar_names[scalar_index]
            end

        dense_row_indices = map(first, scalar_level_indices_for_row)
        dense_column_indices = map(first, scalar_level_indices_for_column)
        block_sparsity_pattern =
            view(sparsity_pattern, dense_row_indices, dense_column_indices)

        for level_index in 1:n_rows, band_index in lower_band:upper_band
            # If the band matrix index corresponds to a valid index into the
            # dense array, mark that index in the sparsity pattern.
            level_index_min = band_index < 0 ? 1 - band_index : 1
            level_index_max =
                band_index < n_cols - n_rows ? n_rows : n_cols - band_index
            if level_index_min <= level_index <= level_index_max
                block_sparsity_pattern[level_index, level_index + band_index] =
                    true
            end
        end
    end
    
    result = SparseMatrixColorings.coloring(
        SparseMatrixColorings.sparse(sparsity_pattern),
        SparseMatrixColorings.ColoringProblem(),
        SparseMatrixColorings.GreedyColoringAlgorithm(),
    )

    # TODO
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

invert_jacobian!(alg::AutoSparseJacobian, cache, ΔY, R) =
    invert_jacobian!(alg.sparse_alg, cache, ΔY, R)
