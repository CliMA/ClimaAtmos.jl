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
        sparse_column_blocks = column_iterator(sparse_matrix_block)

        # TODO: This may not be the best way to represent tensors as scalars.
        is_child_name_of_row = Base.Fix2(MatrixFields.is_child_name, row_name)
        is_child_name_of_col = Base.Fix2(MatrixFields.is_child_name, col_name)
        row_indices = index_ranges[findfirst(is_child_name_of_row, field_names)]
        col_indices = index_ranges[findfirst(is_child_name_of_col, field_names)]
        dense_column_blocks =
            Iterators.map(axes(dense_matrix, 1)) do column_index
                view(dense_matrix, column_index, row_indices, col_indices)
            end

        itr = zip(sparse_column_blocks, dense_column_blocks)
        @threaded device for (sparse_column_block, dense_column_block) in itr
            banded_matrix_transpose =
                MatrixFields.column_field2array_view(sparse_column_block)'
            band_indices =
                (-banded_matrix_transpose.u):(banded_matrix_transpose.l)
            @inbounds for band_index in band_indices
                transposed_band = MatrixFields.band(-band_index)
                sparse_column_block_view =
                    dataview(view(banded_matrix_transpose, transposed_band))
                dense_column_block_indices =
                    diagind(dense_column_block, band_index)
                dense_column_block_view =
                    view(dense_column_block, dense_column_block_indices)
                for level_index in axes(sparse_column_block_view, 1)
                    sparse_column_block_view[level_index] =
                        dense_column_block_view[level_index]
                end
            end
        end
    end
end
