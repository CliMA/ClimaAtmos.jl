import SparseMatrixColorings

"""
    AutoSparseJacobian(sparse_alg, max_padding_bands_per_block)

TODO: Document this monstrosity of an algorithm.
"""
struct AutoSparseJacobian{A <: JacobianAlgorithm} <: JacobianAlgorithm
    sparse_alg::A
    max_padding_bands_per_block::Int
end

function jacobian_cache(alg::AutoSparseJacobian, Y, atmos)
    (; sparse_alg, max_padding_bands_per_block) = alg

    FT = eltype(Y)
    DA = ClimaComms.array_type(Y)
    device = ClimaComms.device(Y.c)

    column_indices = column_index_iterator(Y) # iterator of (i, j, h)
    field_vector_indices = field_vector_index_iterator(Y) # iterator of (f, v)
    scalar_names = scalar_field_names(Y) # iterator of names corresponding to f

    # TODO: Add FieldNameTree(Y) to the matrix in FieldMatrixWithSolver. The
    # tree is needed to evaluate scalar_tendency_matrix[autodiff_matrix_keys].
    # (; matrix) = jacobian_cache(sparse_alg, Y, atmos) # sparse matrix ∂R/∂Y
    matrix_without_tree = jacobian_cache(sparse_alg, Y, atmos).matrix
    tree = MatrixFields.FieldNameTree(Y)
    matrix = MatrixFields.FieldMatrixWithSolver(
        MatrixFields.replace_name_tree(matrix_without_tree.matrix, tree),
        matrix_without_tree.solver,
    )

    # Allocate ∂Yₜ/∂Y and create a view of the scalar components of its blocks.
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

    # Find keys of non-constant blocks, which are represented by matrix fields.
    non_constant_scalar_block_keys =
        unrolled_filter(scalar_block_keys) do scalar_block_key
            scalar_tendency_matrix[scalar_block_key] isa Fields.Field
        end

    # Create a new view of ∂Yₜ/∂Y that has scalar keys and non-constant blocks.
    autodiff_matrix_keys =
        MatrixFields.FieldMatrixKeys(non_constant_scalar_block_keys)
    autodiff_matrix = scalar_tendency_matrix[autodiff_matrix_keys]

    # Construct a mask for nonzero entries in autodiff_matrix as a dense array,
    # and a similar mask with additional padding bands in each block.
    # TODO: Improve performance by rescaling blocks instead of adding new bands.
    N = length(Fields.column(Y, 1, 1, 1))
    jacobian_axis_index_to_field_vector_index_map =
        collect(enumerate(field_vector_indices))
    sparsity_mask = Array{Bool}(undef, N, N)
    sparsity_mask .= false
    padded_sparsity_mask = copy(sparsity_mask)
    for (block_row_index, block_row_name) in enumerate(scalar_names),
        (block_column_index, block_column_name) in enumerate(scalar_names)

        # Get a view of this block's sparsity masks with its row/column indices.
        jacobian_row_index_to_Yₜ_index_map =
            filter(jacobian_axis_index_to_field_vector_index_map) do index_pair
                (_, (scalar_index, _)) = index_pair
                scalar_index == block_row_index
            end
        jacobian_column_index_to_Y_index_map =
            filter(jacobian_axis_index_to_field_vector_index_map) do index_pair
                (_, (scalar_index, _)) = index_pair
                scalar_index == block_column_index
            end
        block_mask_indices = (
            map(first, jacobian_row_index_to_Yₜ_index_map),
            map(first, jacobian_column_index_to_Y_index_map),
        )
        block_sparsity_mask = view(sparsity_mask, block_mask_indices...)
        padded_block_sparsity_mask =
            view(padded_sparsity_mask, block_mask_indices...)

        # Identify the nonzero bands in the sparse matrix and the padding bands.
        (n_band_matrix_rows, n_band_matrix_columns) = size(block_sparsity_mask)
        if (block_row_name, block_column_name) in keys(autodiff_matrix)
            matrix_field = autodiff_matrix[block_row_name, block_column_name]
            (_, _, lower_band, upper_band) =
                MatrixFields.band_matrix_info(matrix_field)
        elseif n_band_matrix_rows != n_band_matrix_columns
            (lower_band, upper_band) =
                n_band_matrix_rows < n_band_matrix_columns ? (1, 0) : (0, -1)
        else
            (lower_band, upper_band) = (1 / 2, -1 / 2)
        end
        n_padding_bands_per_side = max_padding_bands_per_block / 2
        lower_padding_band = ceil(Int, lower_band - n_padding_bands_per_side)
        upper_padding_band = floor(Int, upper_band + n_padding_bands_per_side)

        # Update the mask entries for the bands in this block.
        for band in lower_padding_band:upper_padding_band
            is_not_padding_band = band in lower_band:upper_band
            level_index_min = band < 0 ? 1 - band : 1
            level_index_max =
                band < n_band_matrix_columns - n_band_matrix_rows ?
                n_band_matrix_rows : n_band_matrix_columns - band
            for level_index in level_index_min:level_index_max
                block_mask_index = (level_index, level_index + band)
                if is_not_padding_band
                    block_sparsity_mask[block_mask_index...] = true
                end
                padded_block_sparsity_mask[block_mask_index...] = true
            end
        end
    end

    # TODO: Find a graph coloring that is guaranteed to be optimal.
    coloring = SparseMatrixColorings.coloring(
        SparseMatrixColorings.sparse(padded_sparsity_mask),
        SparseMatrixColorings.ColoringProblem(),
        SparseMatrixColorings.GreedyColoringAlgorithm(),
    )

    # type of each dual number (add the tag "Jacobian" for specialized dispatch)
    n_εs = SparseMatrixColorings.ncolors(coloring)
    FT_dual = ForwardDiff.Dual{Jacobian, FT, n_εs}

    # FieldVectors and cached fields with dual numbers instead of real numbers
    Y_dual = replace_parent_type(Y, FT_dual)
    Y_dual_εs = similar(Fields.column(Y_dual, 1, 1, 1))
    Yₜ_dual = similar(Y_dual)
    precomputed_dual =
        replace_parent_type(implicit_precomputed_quantities(Y, atmos), FT_dual)
    scratch_dual = replace_parent_type(temporary_quantities(Y, atmos), FT_dual)

    # iterator of dual number components for each of the values in Y that
    # require autodiff, and 0 for values that do not require it (by not
    # initializing dual number components when they are unneeded, we avoid some
    # of the errors introduced by our sparsity approximation)
    all_ε_indices = SparseMatrixColorings.column_colors(coloring)
    for Y_index_and_sparsity_mask in enumerate(eachcol(sparsity_mask))
        (Y_index, jacobian_column_sparsity_mask) = Y_index_and_sparsity_mask
        if !any(jacobian_column_sparsity_mask)
            all_ε_indices[Y_index] = 0
        end
    end

    # iterator of pairs ((f, v), ε_index), where ε_index specifies the component
    # of the dual number in row (f, v) of Y_dual that corresponds to the
    # diagonal entry in the same row of the matrix ∂Y/∂Y (or 0 if the
    # corresponding value in Y does not require autodiff)
    Y_index_to_ε_index_map = zip(field_vector_indices, all_ε_indices)

    # Set Y_dual_εs to be a FieldVector of the same type as Y_dual, whose dual
    # number components correspond to the N × N identity matrix ∂Y/∂Y.
    # Specifically, each column of Y_dual_εs is a vector of N dual numbers,
    # where each dual number is stored as a combination of a value and n_εs
    # partial derivatives. The partial derivative components can be interpreted
    # as a sparse N × n_εs representation of ∂Y/∂Y.
    for ((scalar_index, level_index), ε_index) in Y_index_to_ε_index_map
        dual_field =
            MatrixFields.get_field(Y_dual_εs, scalar_names[scalar_index])
        point(dual_field, level_index, 1, 1, 1)[] =
            ForwardDiff.Dual{Jacobian}(0, ntuple(==(ε_index), n_εs))
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
    # band matrix row appear in the components of the dual number, with an index
    # of 0 denoting entries that are not in the sparse matrix; ε_indices is
    # padded to have a constant size and the entire iterator is converted to a
    # DA for GPU compatibility
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
                Iterators.filter(Y_index_to_ε_index_map) do index_pair
                    ((scalar_index, _), _) = index_pair
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

function update_jacobian!(::AutoSparseJacobian, cache, Y, p, dtγ, t)
    (; matrix, tendency_matrix, autodiff_matrix) = cache
    (; Y_dual, Y_dual_εs, Yₜ_dual, precomputed_dual, scratch_dual) = cache
    (; band_matrix_row_index_to_ε_indices_map) = cache

    device = ClimaComms.device(Y.c)
    column_indices = column_index_iterator(Y)
    scalar_names = scalar_field_names(Y)
    p_dual = replace_precomputed_and_scratch(p, precomputed_dual, scratch_dual)

    # Set the dual number components in Y_dual to represent the identity matrix.
    # TODO: Update is_diagonal_bc in ClimaCore to fix FieldVector broadcasting.
    Y_dual.c .= Y.c .+ Y_dual_εs.c
    Y_dual.f .= Y.f .+ Y_dual_εs.f

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
                dual_field = MatrixFields.get_field(Yₜ_dual, name)
                @inbounds point(dual_field, level_index, column_index...)[]
            end
            Yₜ_partials = ForwardDiff.partials(Yₜ_dual_value)
            unrolled_applyat(block_index, matrix_fields) do matrix_field
                (_, _, lower_band, upper_band) =
                    MatrixFields.band_matrix_info(matrix_field)
                band_matrix_row_entries =
                    ntuple(Val(upper_band - lower_band + 1)) do band_index
                        FT = eltype(Yₜ_partials)
                        ε_index = ε_indices[band_index]
                        ε_index > 0 ? Yₜ_partials[ε_index] : FT(0)
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
