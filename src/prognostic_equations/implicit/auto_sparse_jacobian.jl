"""
    AutoSparseJacobian(sparse_jacobian_algorithm)

A `JacobianAlgorithm` that computes an `AutoDenseJacobian`, copies its values
into the non-zero bands of a `ManualSparseJacobian`, and uses the resulting
sparse matrix to solve the implicit equation.
"""
struct AutoSparseJacobian{A <: ManualSparseJacobian} <: JacobianAlgorithm
    sparse_jacobian_algorithm::A
end

function jacobian_cache(alg::AutoSparseJacobian, Y, atmos)
    dense_cache = jacobian_cache(AutoDenseJacobian(), Y, atmos)
    sparse_cache = jacobian_cache(alg.sparse_jacobian_algorithm, Y, atmos)
    return (; dense_cache..., sparse_cache...)
end

function update_jacobian!(::AutoSparseJacobian, cache, Y, p, dtγ, t)
    (; column_matrices, I_matrix, matrix) = cache
    rescaled_column_matrices = cache.column_lu_factors
    update_jacobian_skip_factorizing!(AutoDenseJacobian(), cache, Y, p, dtγ, t)
    rescaled_column_matrices .= dtγ .* column_matrices .- I_matrix
    dense_matrix_to_field_matrix!(matrix, rescaled_column_matrices, Y)
end

invert_jacobian!(::AutoSparseJacobian, cache, ΔY, R) =
    LinearAlgebra.ldiv!(ΔY, cache.matrix, R)

save_jacobian!(::AutoSparseJacobian, cache, Y, dtγ, t) =
    save_jacobian!(alg.sparse_jacobian_algorithm, cache, Y, dtγ, t)

function dense_matrix_to_field_matrix!(field_matrix, dense_matrix, Y)
    device = ClimaComms.device(Y.c)
    field_names = scalar_field_names(Y)
    index_ranges = scalar_field_index_ranges(Y)
    sparse_matrix = MatrixFields.scalar_fieldmatrix(field_matrix.matrix, Y)

    dense_row_ranges = unrolled_map(keys(sparse_matrix).values) do (row_name, _)
        is_child_name = Base.Fix2(MatrixFields.is_child_name, row_name)
        index_ranges[unrolled_findfirst(is_child_name, field_names)]
    end
    dense_col_ranges = unrolled_map(keys(sparse_matrix).values) do (_, col_name)
        is_child_name = Base.Fix2(MatrixFields.is_child_name, col_name)
        index_ranges[unrolled_findfirst(is_child_name, field_names)]
    end

    unrolled_foreach(
        dense_row_ranges,
        dense_col_ranges,
        values(sparse_matrix),
    ) do dense_rows, dense_cols, sparse_matrix_block
        sparse_matrix_block isa Fields.Field || return

        n_rows, n_cols, lower_band, _ =
            MatrixFields.band_matrix_info(sparse_matrix_block)
        horz_space = Spaces.horizontal_space(axes(sparse_matrix_block))
        Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(horz_space))
        sparse_matrix_block_size =
            DataLayouts.farray_size(Fields.field_values(sparse_matrix_block))
        cartesian_indices =
            CartesianIndices(map(Base.OneTo, sparse_matrix_block_size))

        @threaded device for cartesian_index in cartesian_indices
            layout_type = typeof(Fields.field_values(sparse_matrix_block))
            v, f, sparse_horz_index, dense_horz_index =
                if layout_type <: DataLayouts.VIJFH
                    (v, i, j, f, h) = Tuple(cartesian_index)
                    v, f, (i, j, h), (h - 1) * Nq * Nq + (j - 1) * Nq + i
                elseif layout_type <: DataLayouts.VIFH
                    (v, i, f, h) = Tuple(cartesian_index)
                    v, f, (i, h), (h - 1) * Nq + i
                elseif layout_type <: DataLayouts.VF
                    (v, f) = Tuple(cartesian_index)
                    v, f, (1, 1, 1), 1
                else
                    error("Unsupported DataLayout type: $layout_type")
                end
            sparse_column_array =
                parent(Fields.column(sparse_matrix_block, sparse_horz_index...))
            dense_column_array =
                view(dense_matrix, dense_horz_index, dense_rows, dense_cols)

            # Convert the field index f to a dense matrix band index. The dense
            # array index (row, col) corresponds to the band index col - row.
            band = lower_band + f - 1

            # If the sparse array index (v, f) corresponds to a valid index into
            # the dense array, set the value at that index.
            v_min = band < 0 ? 1 - band : 1
            v_max = band < n_cols - n_rows ? n_rows : n_cols - band
            if v_min <= v <= v_max
                sparse_column_array[v, f] = dense_column_array[v, v + band]
            end
        end
    end
end
