import SparseMatrixColorings

"""
    AutoSparseJacobian(sparse_alg)

TODO
"""
struct AutoSparseJacobian{A} <: JacobianAlgorithm
    sparse_alg::A
    pentadiagonal_padding::Bool
end

function jacobian_cache(alg::AutoSparseJacobian, Y, atmos)
    FT = eltype(Y)
    DA = ClimaComms.array_type(Y)
    device = ClimaComms.device(Y.c)

    column_indices = column_index_iterator(Y) # iterator of (i, j, h)
    field_vector_indices = field_vector_index_iterator(Y) # iterator of (f, v)
    scalar_names = scalar_field_names(Y) # iterator of names corresponding to f

    # TODO: Add FieldNameTree(Y) to the matrix in FieldMatrixWithSolver. The
    # tree is needed to evaluate scalar_tendency_matrix[autodiff_matrix_keys].
    # (; matrix) = jacobian_cache(alg.sparse_alg, Y, atmos) # sparse matrix ∂R/∂Y
    matrix_without_tree = jacobian_cache(alg.sparse_alg, Y, atmos).matrix
    tree = MatrixFields.FieldNameTree(Y)
    matrix = MatrixFields.FieldMatrixWithSolver(
        MatrixFields.replace_name_tree(matrix_without_tree.matrix, tree),
        matrix_without_tree.solver,
    )

    # TODO: Use rescaling instead of padding to avoid dual number errors.
    if alg.pentadiagonal_padding
        # Replace all tridiagonal blocks with pentadiagonal blocks, and replace
        # bidiagonal pressure gradient blocks with quaddiagonal blocks.
        # TODO: Add methods of map for FieldNameDict and FieldMatrixWithSolver.
        padded_matrix_blocks =
            unrolled_map(pairs(matrix)) do ((block_row_name, _), matrix_block)
                if matrix_block isa Fields.Field
                    row_type = eltype(matrix_block)
                    T = eltype(row_type)
                    new_row_type =
                        if row_type == MatrixFields.TridiagonalMatrixRow{T}
                            MatrixFields.PentadiagonalMatrixRow{T}
                        elseif (
                            row_type == MatrixFields.BidiagonalMatrixRow{T} &&
                            block_row_name == @name(f.u₃)
                        )
                            MatrixFields.QuaddiagonalMatrixRow{T}
                        else
                            row_type
                        end
                    new_row_type == row_type ? matrix_block :
                    similar(matrix_block, new_row_type)
                else
                    matrix_block
                end
            end
        matrix = MatrixFields.FieldMatrixWithSolver(
            MatrixFields.FieldNameDict(keys(matrix), padded_matrix_blocks),
            Y,
            matrix.solver.alg,
        )
    end

    # sparse matrix ∂Yₜ/∂Y and a view of the scalar components of its blocks
    tendency_matrix = matrix .+ one(matrix)
    scalar_tendency_matrix = MatrixFields.scalar_field_matrix(tendency_matrix)

    # Find scalar keys that correspond to the scalar components of the blocks.
    # When we approximate a tensor derivative as a scalar multiple of the
    # identity tensor, we compute the scalar quantity using the top-left
    # component of the tensor. For example, the derivative of Yₜ.c.uₕ with
    # respect to Y.c.uₕ is computed as the derivative of
    # Yₜ.c.uₕ.components.data.:1 with respect to Y.c.uₕ.components.data.:1.
    scalar_block_keys =
        map(keys(scalar_tendency_matrix)) do (block_row_name, block_column_name)
            scalar_block_row_name =
                block_row_name in scalar_names ? block_row_name :
                unrolled_argfirst(scalar_names) do scalar_name
                    MatrixFields.is_child_name(scalar_name, block_row_name)
                end
            scalar_block_column_name =
                block_column_name in scalar_names ? block_column_name :
                unrolled_argfirst(scalar_names) do scalar_name
                    MatrixFields.is_child_name(scalar_name, block_column_name)
                end
            (scalar_block_row_name, scalar_block_column_name)
        end

    # keys of non-constant scalar blocks, which are represented by matrix fields
    non_constant_scalar_block_keys =
        unrolled_filter(scalar_block_keys) do scalar_block_key
            scalar_tendency_matrix[scalar_block_key] isa Fields.Field
        end

    # view of scalar_tendency_matrix whose blocks will be updated using autodiff
    autodiff_matrix_keys =
        MatrixFields.FieldMatrixKeys(non_constant_scalar_block_keys)
    autodiff_matrix = scalar_tendency_matrix[autodiff_matrix_keys]

    # Construct a mask for nonzero entries in autodiff_matrix as a dense array.
    N = length(Fields.column(Y, 1, 1, 1))
    jacobian_axis_index_to_field_vector_index_map =
        collect(enumerate(field_vector_indices))
    sparsity_mask = Array{Bool}(undef, N, N)
    sparsity_mask .= false
    for (scalar_block_key, matrix_field) in autodiff_matrix
        # Use the block key to get a view of the sparsity mask for this block.
        (scalar_block_row_name, scalar_block_column_name) = scalar_block_key
        block_row_index_to_Yₜ_index_map =
            filter(jacobian_axis_index_to_field_vector_index_map) do index_pair
                (_, (scalar_index, _)) = index_pair
                scalar_names[scalar_index] == scalar_block_row_name
            end
        block_row_indices = map(first, block_row_index_to_Yₜ_index_map)
        block_column_index_to_Y_index_map =
            filter(jacobian_axis_index_to_field_vector_index_map) do index_pair
                (_, (scalar_index, _)) = index_pair
                scalar_names[scalar_index] == scalar_block_column_name
            end
        block_column_indices = map(first, block_column_index_to_Y_index_map)
        block_sparsity_mask =
            view(sparsity_mask, block_row_indices, block_column_indices)

        # Update the mask entries for the band matrix field in this block.
        (n_band_matrix_rows, n_band_matrix_columns, lower_band, upper_band) =
            MatrixFields.band_matrix_info(matrix_field)
        for band in lower_band:upper_band
            level_index_min = band < 0 ? 1 - band : 1
            level_index_max =
                band < n_band_matrix_columns - n_band_matrix_rows ?
                n_band_matrix_rows : n_band_matrix_columns - band
            for level_index in level_index_min:level_index_max
                @inbounds block_sparsity_mask[level_index, level_index + band] =
                    true
            end
        end
    end

    # TODO: Find an optimal graph coloring.
    coloring = SparseMatrixColorings.coloring(
        SparseMatrixColorings.sparse(sparsity_mask),
        SparseMatrixColorings.ColoringProblem(),
        SparseMatrixColorings.GreedyColoringAlgorithm(),
    )

    # number of components in each dual number (and a Val for GPU compatibility)
    n_εs = SparseMatrixColorings.ncolors(coloring)
    n_εs_val = Val(n_εs)

    FT_dual = ForwardDiff.Dual{Jacobian, FT, n_εs}
    Y_dual = replace_parent_type(Y, FT_dual)
    Y_dual_εs = similar(Y_dual) # TODO: Allow this to be a column FieldVector.
    Yₜ_dual = similar(Y_dual)
    precomputed_dual =
        replace_parent_type(implicit_precomputed_quantities(Y, atmos), FT_dual)
    scratch_dual = replace_parent_type(temporary_quantities(Y, atmos), FT_dual)

    # iterator of dual number components for each of the values in Y that
    # require autodiff, and 0 for values that do not require it
    all_ε_indices = SparseMatrixColorings.column_colors(coloring)
    for Y_value_index_and_sparsity_mask in enumerate(eachcol(sparsity_mask))
        (Y_value_index, Y_value_sparsity_mask) = Y_value_index_and_sparsity_mask
        if !any(Y_value_sparsity_mask)
            all_ε_indices[Y_value_index] = 0
        end
    end

    # iterator of pairs ((f, v), ε_index), where ε_index specifies the component
    # of the dual number in row (f, v) of Y_dual that corresponds to the
    # diagonal entry in the same row of the matrix ∂Y/∂Y (or 0 if the
    # corresponding value in Y does not require autodiff)
    Y_index_to_ε_index_map = collect(zip(field_vector_indices, all_ε_indices))

    # Set Y_dual_εs to be a FieldVector of the same type as Y_dual, whose dual
    # number components correspond to the N × N identity matrix ∂Y/∂Y.
    # Specifically, each column of Y_dual_εs is a vector of N dual numbers,
    # where each dual number is stored as a combination of a value and n_εs
    # partial derivatives. The partial derivative components can be interpreted
    # as a sparse N × n_εs representation of ∂Y/∂Y.
    ClimaComms.@threaded device begin
        # On multithreaded devices, assign one thread to each value in Y_dual.
        for column_index in column_indices,
            ((scalar_index, level_index), ε_index) in DA(Y_index_to_ε_index_map)

            Y_dual_εs_value =
                ForwardDiff.Dual{Jacobian}(0, ntuple(==(ε_index), n_εs_val))
            unrolled_applyat(scalar_index, scalar_names) do name
                field = MatrixFields.get_field(Y_dual_εs, name)
                @inbounds point(field, level_index, column_index...)[] =
                    Y_dual_εs_value
            end
        end
    end

    # dual number components needed to represent a band matrix row in any block
    max_εs_in_block = maximum(values(autodiff_matrix)) do matrix_field
        (_, _, lower_band, upper_band) =
            MatrixFields.band_matrix_info(matrix_field)
        upper_band - lower_band + 1
    end

    # iterator of pairs ((b, v), (f, ε_indices)), where (b, v) is the index of a
    # band matrix row in autodiff_matrix, (f, v) is the index of a dual
    # number in Yₜ_dual, and ε_indices is the order in which the entries of the
    # band matrix row appear in the components of the dual number (zero-padded
    # to have a constant size and converted to a DA for GPU compatibility)
    band_matrix_row_index_to_ε_indices_maps =
        map(enumerate(pairs(autodiff_matrix))) do block_index_and_pair
            (block_index, (scalar_block_key, matrix_field)) =
                block_index_and_pair
            (scalar_block_row_name, scalar_block_column_name) = scalar_block_key
            (
                n_band_matrix_rows,
                n_band_matrix_columns,
                lower_band,
                upper_band,
            ) = MatrixFields.band_matrix_info(matrix_field)

            block_Yₜ_indices =
                Iterators.filter(field_vector_indices) do (scalar_index, _)
                    scalar_names[scalar_index] == scalar_block_row_name
                end
            block_Y_index_to_ε_index_map =
                filter(Y_index_to_ε_index_map) do ((scalar_index, _), _)
                    scalar_names[scalar_index] == scalar_block_column_name
                end
            block_ε_indices = last.(block_Y_index_to_ε_index_map)

            map(block_Yₜ_indices) do (scalar_index, level_index)
                ε_indices = ntuple(max_εs_in_block) do band_index
                    band = lower_band + band_index - 1
                    level_index_min = band < 0 ? 1 - band : 1
                    level_index_max =
                        band < n_band_matrix_columns - n_band_matrix_rows ?
                        n_band_matrix_rows : n_band_matrix_columns - band
                    is_ε_in_band =
                        band <= upper_band &&
                        level_index_min <= level_index <= level_index_max
                    is_ε_in_band ? block_ε_indices[level_index + band] : 0
                end
                ((block_index, level_index), (scalar_index, ε_indices))
            end
        end
    band_matrix_row_index_to_ε_indices_map =
        DA(collect(Iterators.flatten(band_matrix_row_index_to_ε_indices_maps)))

    return (;
        matrix,
        tendency_matrix,
        autodiff_matrix,
        Y_dual,
        Y_dual_εs,
        Yₜ_dual,
        precomputed_dual,
        scratch_dual,
        band_matrix_row_index_to_ε_indices_map,
    )
end

function update_jacobian!(alg::AutoSparseJacobian, cache, Y, p, dtγ, t)
    (; matrix, tendency_matrix, autodiff_matrix) = cache
    (; Y_dual, Y_dual_εs, Yₜ_dual, precomputed_dual, scratch_dual) = cache
    (; band_matrix_row_index_to_ε_indices_map) = cache

    device = ClimaComms.device(Y.c)

    column_indices = column_index_iterator(Y)
    scalar_names = scalar_field_names(Y)

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

    Y_dual .= Y .+ Y_dual_εs

    # Compute sparse approximations of ∂p/∂Y and ∂Yₜ/∂Y.
    set_implicit_precomputed_quantities!(Y_dual, p_dual, t)
    implicit_tendency!(Yₜ_dual, Y_dual, p_dual, t)

    # Move the entries of ∂Yₜ/∂Y from Yₜ_dual into autodiff_matrix.
    ClimaComms.@threaded device begin
        # On multithreaded devices, assign one thread to each band matrix row
        # in autodiff_matrix.
        for column_index in column_indices,
            index_pair in band_matrix_row_index_to_ε_indices_map

            ((block_index, level_index), (scalar_index, ε_indices)) = index_pair
            matrix_fields = values(autodiff_matrix)

            Yₜ_dual_value = unrolled_applyat(scalar_index, scalar_names) do name
                field = MatrixFields.get_field(Yₜ_dual, name)
                @inbounds point(field, level_index, column_index...)[]
            end
            Yₜ_partials = ForwardDiff.partials(Yₜ_dual_value)
            unrolled_applyat(block_index, matrix_fields) do matrix_field
                (_, _, lower_band, upper_band) =
                    MatrixFields.band_matrix_info(matrix_field)
                band_matrix_row_entries =
                    ntuple(Val(upper_band - lower_band + 1)) do band_index
                        FT = eltype(Yₜ_partials)
                        ε_index = ε_indices[band_index]
                        ε_index == 0 ? zero(FT) : Yₜ_partials[ε_index]
                    end
                @inbounds point(matrix_field, level_index, column_index...)[] =
                    eltype(matrix_field)(band_matrix_row_entries)
            end
        end
    end

    # Update the matrix for ∂R/∂Y using the new values of ∂Yₜ/∂Y.
    matrix .= dtγ .* tendency_matrix .- one(matrix)
end

invert_jacobian!(alg::AutoSparseJacobian, cache, ΔY, R) =
    invert_jacobian!(alg.sparse_alg, cache, ΔY, R)
