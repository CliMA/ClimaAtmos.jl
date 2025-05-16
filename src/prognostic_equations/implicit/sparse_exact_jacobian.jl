"""
    SparseExactJacobian(approx_jacobian_algorithm)

A `JacobianAlgorithm` that computes the `ExactJacobian`, then copies its values
into the non-zero bands of the `ApproxJacobian` and uses the resulting sparse
matrix to solve the implicit equation.
"""
struct SparseExactJacobian{A <: ApproxJacobian} <: JacobianAlgorithm
    approx_jacobian_algorithm::A
end

function jacobian_cache(alg::SparseExactJacobian, Y, atmos)
    exact_cache = jacobian_cache(ExactJacobian(), Y, atmos)
    approx_cache = jacobian_cache(alg.approx_jacobian_algorithm, Y, atmos)
    return (; exact_cache..., approx_cache...)
end

function update_jacobian!(::SparseExactJacobian, cache, Y, p, dtγ, t)
    (; column_matrices, I_matrix, matrix) = cache
    rescaled_column_matrices = cache.column_lu_factors
    update_jacobian_skip_factorizing!(ExactJacobian(), cache, Y, p, dtγ, t)
    rescaled_column_matrices .= dtγ .* column_matrices .- I_matrix
    dense_matrix_to_field_matrix!(matrix, rescaled_column_matrices, Y)
end

invert_jacobian!(::SparseExactJacobian, cache, ΔY, R) =
    LinearAlgebra.ldiv!(ΔY, cache.matrix, R)

save_jacobian!(::SparseExactJacobian, cache, Y, dtγ, t) =
    save_jacobian!(alg.approx_jacobian_algorithm, cache, Y, dtγ, t)

function dense_matrix_to_field_matrix!(field_matrix, dense_matrix, Y)
    sparse_matrix = MatrixFields.scalar_fieldmatrix(field_matrix.matrix, Y)
    field_names = scalar_field_names(Y)
    index_ranges = scalar_field_index_ranges(Y)
    device = ClimaComms.device(Y.c)

    for ((row_name, col_name), sparse_matrix_block) in sparse_matrix
        sparse_matrix_block isa LinearAlgebra.UniformScaling && continue

        # TODO: This may not be the best way to represent tensors as scalars.
        is_child_name_of_row = Base.Fix2(MatrixFields.is_child_name, row_name)
        is_child_name_of_col = Base.Fix2(MatrixFields.is_child_name, col_name)
        row_indices = index_ranges[findfirst(is_child_name_of_row, field_names)]
        col_indices = index_ranges[findfirst(is_child_name_of_col, field_names)]

        horz_space = Spaces.horizontal_space(axes(sparse_matrix_block))
        Ni = Quadratures.degrees_of_freedom(Spaces.quadrature_style(horz_space))
        sparse_matrix_block_size =
            DataLayouts.farray_size(Fields.field_values(sparse_matrix_block))
        cartesian_indices =
            CartesianIndices(map(Base.OneTo, sparse_matrix_block_size))

        @threaded device for cartesian_index in cartesian_indices
            layout_type = Fields.field_values(sparse_matrix_block)
            if layout_type isa DataLayouts.VIJFH
                (v, i, j, f, h) = Tuple(cartesian_index)
                sparse_column_index = (i, j, h)
                dense_column_index = (h - 1) * Ni * Ni + (j - 1) * Ni + i
            elseif layout_type isa DataLayouts.VIFH
                (v, i, f, h) = Tuple(cartesian_index)
                sparse_column_index = (i, h)
                dense_column_index = (h - 1) * Ni + i
            end
            sparse_column_block =
                Fields.column(sparse_matrix_block, sparse_column_index...)
            dense_column_block =
                view(dense_matrix, dense_column_index, row_indices, col_indices)

            # Convert the sparse matrix diagonal index f to a dense matrix
            # diagonal index, and determine whether the sparse matrix index
            # (v, f) is a valid index into the dense matrix.
            # Note: The dense matrix index (1, 1) corresponds to the diagonal
            # index 0, and the dense matrix index (n_rows, n_cols) corresponds
            # to the diagonal index n_cols - n_rows.
            n_rows, n_cols, matrix_ld, _ =
                MatrixFields.band_matrix_info(sparse_column_block)
            matrix_d = matrix_ld + f - 1
            first_row = matrix_d < 0 ? 1 - matrix_d : 1
            last_row = matrix_d < n_cols - n_rows ? n_rows : n_cols - matrix_d

            if first_row <= v <= last_row
                parent(sparse_column_block)[v, f] =
                    dense_column_block[v, v + matrix_d]
            end
        end
    end
end
