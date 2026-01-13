import SparseMatrixColorings

"""
    AutoSparseJacobian(sparse_jacobian_alg, [padding_bands_per_block])

A [`JacobianAlgorithm`](@ref) that computes the Jacobian using forward-mode
automatic differentiation, assuming that the Jacobian's sparsity structure is
given by `sparse_jacobian_alg`.

Only entries that are exptected to be nonzero according to the sparsity
structure are updated, but any other entries that are nonzero can introduce
errors to the updated entries. This issue can be avoided by adding padding bands
to blocks that are likely to introduce errors. In cases where the default
padding bands are insufficient, `padding_bands_per_block` can be specified to
add a fixed number of padding bands to every block.

For more information about this algorithm, see [Implicit Solver](@ref).
"""
struct AutoSparseJacobian{A <: SparseJacobian, P} <: SparseJacobian
    sparse_jacobian_alg::A
    padding_bands_per_block::P
end
AutoSparseJacobian(sparse_jacobian_alg) =
    AutoSparseJacobian(sparse_jacobian_alg, nothing)

function jacobian_cache(alg::AutoSparseJacobian, Y, atmos; verbose = true)
    (; sparse_jacobian_alg, padding_bands_per_block) = alg

    FT = eltype(Y)
    DA = ClimaComms.array_type(Y)
    device = ClimaComms.device(Y.c)

    column_indices = column_index_iterator(Y) # iterator of (i, j, h)
    field_vector_indices = field_vector_index_iterator(Y) # iterator of (f, v)
    scalar_names = scalar_field_names(Y) # iterator of names corresponding to f

    precomputed = implicit_precomputed_quantities(Y, atmos)
    scratch = implicit_temporary_quantities(Y, atmos)

    # Allocate ∂R/∂Y and its corresponding linear solver.
    # TODO: Add FieldNameTree(Y) to the matrix in FieldMatrixWithSolver. The
    # tree is needed to evaluate scalar_tendency_matrix[autodiff_matrix_keys].
    # (; matrix) = jacobian_cache(sparse_jacobian_alg, Y, atmos)
    matrix_without_tree = jacobian_cache(sparse_jacobian_alg, Y, atmos).matrix
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
    # TODO: Improve performance by only adding bands where they are necessary,
    # or by rescaling blocks instead of adding new bands.
    N = length(Fields.column(Y, 1, 1, 1))
    sparsity_mask = Array{Bool}(undef, N, N)
    sparsity_mask .= false
    padded_sparsity_mask = copy(sparsity_mask)
    for block_key in Iterators.product(scalar_names, scalar_names)
        (block_row_name, block_column_name) = block_key

        # Get a view of this block's sparsity masks with its row/column indices.
        block_jacobian_row_index_to_Yₜ_index_map =
            Iterators.filter(enumerate(field_vector_indices)) do index_pair
                (_, (scalar_index, _)) = index_pair
                scalar_names[scalar_index] == block_row_name
            end
        block_jacobian_column_index_to_Y_index_map =
            Iterators.filter(enumerate(field_vector_indices)) do index_pair
                (_, (scalar_index, _)) = index_pair
                scalar_names[scalar_index] == block_column_name
            end
        block_view_indices = (
            map(first, block_jacobian_row_index_to_Yₜ_index_map),
            map(first, block_jacobian_column_index_to_Y_index_map),
        )
        block_sparsity_mask = view(sparsity_mask, block_view_indices...)
        padded_block_sparsity_mask =
            view(padded_sparsity_mask, block_view_indices...)

        # Compute the lower and upper band indices of this block, with empty
        # blocks corresponding to index ranges whose length is -1 (centered
        # around 0 for square blocks and around ±1/2 for non-square blocks).
        (n_rows_in_block, n_columns_in_block) = size(block_sparsity_mask)
        if block_key in keys(autodiff_matrix)
            (_, _, lower_band, upper_band) =
                MatrixFields.band_matrix_info(autodiff_matrix[block_key])
        else
            (lower_band, upper_band) =
                n_rows_in_block == n_columns_in_block ? (1 / 2, -1 / 2) :
                (n_rows_in_block < n_columns_in_block ? (1, 0) : (0, -1))
        end

        # Symmetrically expand the range of band indices, with the number of
        # new bands either limited by padding_bands_per_block, or hardcoded
        # for each block when padding_bands_per_block is not specified.
        mass_names = (@name(c.ρ), @name(c.sgsʲs.:(1).ρa))
        uₕ_component_names =
            (@name(c.uₕ.components.data.:(1)), @name(c.uₕ.components.data.:(2)))
        condensate_names =
            (@name(c.ρq_liq), @name(c.ρq_ice), @name(c.ρq_rai), @name(c.ρq_sno))
        max_padding_bands = if !isnothing(padding_bands_per_block)
            padding_bands_per_block
        elseif (
            (
                block_row_name in uₕ_component_names &&
                block_column_name in (mass_names..., @name(c.ρtke)) ||
                block_row_name == @name(c.ρe_tot) &&
                block_column_name in (
                    mass_names...,
                    condensate_names...,
                    @name(c.ρtke),
                    @name(c.sgsʲs.:(1).q_tot),
                ) ||
                block_row_name == @name(c.sgsʲs.:(1).ρa) &&
                block_column_name == @name(c.ρq_tot) ||
                block_row_name == @name(f.sgsʲs.:(1).u₃.components.data.:(1)) &&
                block_column_name in uₕ_component_names
            ) &&
            !(block_key in keys(autodiff_matrix)) &&
            (block_row_name, block_row_name) in keys(autodiff_matrix)
        )
            # Missing off-diagonal blocks whose entries typically have
            # magnitudes that are larger than (or similar to) diagonal blocks in
            # the same rows:
            # - ‖∂ᶜuᵢₜ/∂ᶜρ‖, ‖∂ᶜuᵢₜ/∂ᶜρaʲ‖ ≳ ‖∂ᶜuᵢₜ/∂ᶜuᵢ‖, and ‖∂ᶜuᵢₜ/∂ᶜρtke‖
            #   where uᵢ is either u₁ or u₂, as long as ‖δᶜρ‖, ‖δᶜρaʲ‖ and
            #   ‖δᶜρtke‖ are relatively smaller than ‖δᶜuᵢ‖.
            # - ‖∂ᶜρe_totₜ/∂ᶜρ‖, ‖∂ᶜρe_totₜ/∂ᶜχ‖, and ‖∂ᶜρe_totₜ/∂ᶜρχ‖ ≳
            #   ‖∂ᶜρe_totₜ/∂ᶜρe_tot‖ when χ is any scalar of order unity, as
            #   long as ‖δᶜρ‖ and ‖δᶜχ‖ are relatively smaller than ‖δᶜρe_tot‖.
            # - ‖∂ᶜρaʲₜ/∂ᶜρq_tot‖ ≳ ‖∂ᶜρaʲₜ/∂ᶜρaʲ‖, as long as ‖δᶜρq_tot‖ is
            #   relatively smaller than ‖δᶜρaʲ‖.
            # - ‖∂ᶠu₃ʲₜ/∂ᶜu₁‖ and ‖∂ᶠu₃ʲₜ/∂ᶜu₁‖ ≳ ‖∂ᶠu₃ʲₜ/∂ᶠu₃ʲ‖, as long as
            #   ‖δᶜu₁‖ and ‖δᶜu₂‖ are relatively smaller than ‖δᶠu₃ʲ‖.
            # Diagonal blocks are critical for conservation and stability, so
            # these potential errors from off-diagonal blocks should be avoided.
            3
        elseif (
            block_row_name == @name(c.sgsʲs.:(1).ρa) &&
            block_column_name == @name(c.ρ) &&
            !(block_key in keys(autodiff_matrix)) &&
            (block_row_name, @name(c.sgsʲs.:(1).mse)) in keys(autodiff_matrix)
        )
            # ‖∂ᶜρaʲₜ/∂ᶜρ‖ ≳ ‖∂ᶜρaʲₜ/∂ᶜmseʲ‖, as long as ‖δᶜρ‖ is relatively
            # smaller than ‖δᶜmseʲ‖. The ∂ᶜρaʲₜ/∂ᶜmseʲ block is important for
            # stability in some simulations with turbulence, so this potential
            # error from the ∂ᶜρaʲₜ/∂ᶜρ block should be avoided.
            3
        elseif (
            block_row_name == @name(f.u₃.components.data.:(1)) &&
            block_column_name in condensate_names &&
            !(block_key in keys(autodiff_matrix)) &&
            (block_row_name, uₕ_component_names[1]) in keys(autodiff_matrix)
        )
            # ‖∂ᶜu₃ₜ/∂ᶜρχ‖ ≳ ‖∂ᶜu₃ₜ/∂ᶜuₕ‖ when χ is any specific humidity, as
            # long as ‖δᶜρχ‖ is relatively smaller than ‖δᶜuₕ‖. The ∂ᶜu₃ₜ/∂ᶜuₕ
            # block is important for stability in some simulations with
            # topography, so this potential error from ∂ᶜu₃ₜ/∂ᶜρχ blocks should
            # be avoided.
            2
        else
            0
        end
        padded_lower_band = ceil(Int, lower_band - max_padding_bands / 2)
        padded_upper_band = floor(Int, upper_band + max_padding_bands / 2)

        if verbose
            n_padding_bands =
                length(padded_lower_band:padded_upper_band) -
                length(lower_band:upper_band)
            n_padding_bands > 0 &&
                @info "Adding $n_padding_bands padding bands for $block_key"
        end

        # Update the sparsity mask entries corresponding to bands in this block.
        for band in padded_lower_band:padded_upper_band
            is_not_padding_band = band in lower_band:upper_band
            level_index_min = band < 0 ? 1 - band : 1
            level_index_max =
                band < n_columns_in_block - n_rows_in_block ? n_rows_in_block :
                n_columns_in_block - band
            for level_index in level_index_min:level_index_max
                block_mask_index = (level_index, level_index + band)
                block_sparsity_mask[block_mask_index...] = is_not_padding_band
                padded_block_sparsity_mask[block_mask_index...] = true
            end
        end
    end

    # Find a coloring that minimizes the number of required colors.
    all_coloring_orders = SparseMatrixColorings.all_orders()
    jacobian_column_colorings = map(all_coloring_orders) do coloring_order
        SparseMatrixColorings.coloring(
            SparseMatrixColorings.sparse(padded_sparsity_mask),
            SparseMatrixColorings.ColoringProblem(),
            SparseMatrixColorings.GreedyColoringAlgorithm(coloring_order),
        )
    end
    best_order_index =
        findmin(SparseMatrixColorings.ncolors, jacobian_column_colorings)[2]
    best_jacobian_column_coloring = jacobian_column_colorings[best_order_index]
    n_colors = SparseMatrixColorings.ncolors(best_jacobian_column_coloring)

    # When running on GPU devices, divide n_colors into partitions that are each
    # guaranteed to fit in the memory that is currently free (adding a factor of
    # 2 to account for potential future garbage collection).
    n_partitions = if device isa ClimaComms.AbstractCPUDevice
        1
    else
        free_memory = ClimaComms.free_memory(device)
        max_memory = 2 * free_memory
        memory_for_I_matrix = n_colors * parent_memory(Y)
        memory_per_ε =
            (parent_memory(precomputed) + parent_memory(scratch)) +
            2 * parent_memory(Y)
        # Find the smallest possible integer n_partitions and some other integer
        # n_εs such that n_partitions * n_εs >= n_colors and
        # (n_εs + 1) * memory_per_ε + memory_for_I_matrix <= max_memory, where
        # (n_εs + 1) * memory_per_ε is the memory required to store
        # precomputed_dual, scratch_dual, Y_dual, and Yₜ_dual, and where
        # memory_for_I_matrix is an approximation of the memory required to
        # store I_matrix_partitions. The actual memory_for_I_matrix is given by
        # (n_colors + n_partitions) * parent_memory(Y), but we can ignore the
        # value of n_partitions if it is negligible compared to n_colors. When
        # max_memory is too small to fit any εs, try using one ε per partition.
        # TODO: Replace the fields in I_matrix_partitions with column fields,
        # making memory_for_I_matrix negligible compared to max_memory.
        n_εs_max = (max_memory - memory_for_I_matrix) ÷ memory_per_ε - 1
        cld(n_colors, max(n_εs_max, 1))
    end
    n_εs = cld(n_colors, n_partitions)

    if verbose
        @info "Using coloring order $(all_coloring_orders[best_order_index])"
        if n_partitions == 1
            @info "Updating Jacobian using $n_εs ε components per dual number"
        else
            @info "Updating Jacobian using $n_partitions partitions of \
                   $n_colors colors, with $n_εs ε components per dual number"
        end
    end

    # FieldVectors and cached fields with dual numbers instead of real numbers,
    # with dual numbers using the tag "Jacobian" for specialized dispatch
    # TODO: Refactor FieldVector broadcasting so that performance does not
    # deteriorate if we only store one column of each partition_εs.
    FT_dual = ForwardDiff.Dual{Jacobian, FT, n_εs}
    precomputed_dual = replace_parent_eltype(precomputed, FT_dual)
    scratch_dual = replace_parent_eltype(scratch, FT_dual)
    Y_dual = replace_parent_eltype(Y, FT_dual)
    Yₜ_dual = similar(Y_dual)
    I_matrix_partitions = ntuple(_ -> similar(Y_dual), n_partitions)

    # iterator of colors for each of the values in Y that require autodiff, and
    # 0 for values that do not require it (by not initializing dual number
    # components when they are unneeded, we might be able to avoid some of the
    # errors introduced by our sparsity approximation)
    jacobian_column_colors =
        SparseMatrixColorings.column_colors(best_jacobian_column_coloring)
    for Y_index_and_sparsity_mask in enumerate(eachcol(sparsity_mask))
        (Y_index, jacobian_column_sparsity_mask) = Y_index_and_sparsity_mask
        if !any(jacobian_column_sparsity_mask)
            jacobian_column_colors[Y_index] = 0
        end
    end

    # iterator of pairs ((f, v), c), where the color c identifies a component
    # of the dual number in row (f, v) of Y_dual that corresponds to the
    # diagonal entry in the same row of the matrix ∂Y/∂Y (or 0 if the
    # corresponding value in Y does not require autodiff)
    Y_index_to_diagonal_color_map =
        zip(field_vector_indices, jacobian_column_colors)

    # Set the dual numbers in each FieldVector partition_εs so that the ε
    # components correspond to partitions of the N × N identity matrix ∂Y/∂Y.
    # Specifically, every column of partition_εs is a vector of N dual numbers,
    # each of which is stored as a combination of a value and n_εs partial
    # derivatives. The ε components can be interpreted as representing N × n_εs
    # slices of a sparse N × n_colors representation of ∂Y/∂Y. Convert n_εs to
    # a Val and Y_index_to_diagonal_color_map to a DA for GPU compatibility, and
    # drop spatial information from every Field to ensure that this kernel stays
    # below the GPU parameter memory limit.
    n_εs_val = Val(n_εs)
    I_matrix_partitions_data = unrolled_map(I_matrix_partitions) do partition_εs
        unrolled_map(Fields.field_values, Fields._values(partition_εs))
    end
    ClimaComms.@threaded device begin
        # On multithreaded devices, use one thread for each dual number.
        for (partition_index, partition_εs_data) in
            enumerate(I_matrix_partitions_data),
            column_index in column_indices,
            index_pair in DA(collect(Y_index_to_diagonal_color_map))

            ((scalar_index, level_index), diagonal_entry_color) = index_pair
            ε_offset = (partition_index - 1) * n_εs
            diagonal_entry_ε_index =
                ε_offset < diagonal_entry_color <= ε_offset + n_εs ?
                diagonal_entry_color - ε_offset : 0
            ε_coefficients = ntuple(==(diagonal_entry_ε_index), n_εs_val)
            unrolled_applyat(scalar_index, scalar_names) do name
                field = MatrixFields.get_field(partition_εs_data, name)
                @inbounds point(field, level_index, column_index...)[] =
                    ForwardDiff.Dual{Jacobian}(0, ε_coefficients)
            end
        end
    end

    # number of colors needed to represent a band matrix row in any block
    colors_per_band_matrix_row =
        maximum(values(autodiff_matrix)) do matrix_field
            (_, _, lower_band, upper_band) =
                MatrixFields.band_matrix_info(matrix_field)
            upper_band - lower_band + 1
        end

    # iterator of pairs ((b, v), (f, cs)), where (b, v) is the index of a band
    # matrix row in autodiff_matrix, (f, v) is the index of a dual number in
    # Yₜ_dual, and cs is a tuple that contains the colors of the band matrix row
    # entries, which is padded to have a constant size for GPU compatibility
    # TODO: Use an iterator of pairs ((f, v), ((b1, cs1), (b2, cs2), ...)), so
    # that each pair corresponds to a dual number instead of a band matrix row.
    band_matrix_row_index_to_colors_map = Iterators.flatmap(
        enumerate(pairs(autodiff_matrix)),
    ) do (block_index, (block_key, matrix_field))
        (block_row_name, block_column_name) = block_key
        (n_rows_in_block, n_columns_in_block, lower_band, upper_band) =
            MatrixFields.band_matrix_info(matrix_field)

        block_Yₜ_indices =
            Iterators.filter(field_vector_indices) do (scalar_index, _)
                scalar_names[scalar_index] == block_row_name
            end
        block_Y_index_to_color_map =
            Iterators.filter(Y_index_to_diagonal_color_map) do index_pair
                ((scalar_index, _), _) = index_pair
                scalar_names[scalar_index] == block_column_name
            end
        block_colors = last.(block_Y_index_to_color_map)

        map(block_Yₜ_indices) do (scalar_index, level_index)
            entry_colors = ntuple(colors_per_band_matrix_row) do band_index
                band = lower_band + band_index - 1
                level_index_min = band < 0 ? 1 - band : 1
                level_index_max =
                    band < n_columns_in_block - n_rows_in_block ?
                    n_rows_in_block : n_columns_in_block - band
                is_color_at_index =
                    band <= upper_band &&
                    level_index_min <= level_index <= level_index_max
                is_color_at_index ? block_colors[level_index + band] : 0
            end
            ((block_index, level_index), (scalar_index, entry_colors))
        end
    end

    # Convert the lazy iterator to a DA for GPU compatibility.
    band_matrix_row_index_to_colors_map =
        DA(collect(band_matrix_row_index_to_colors_map))

    return (;
        matrix,
        tendency_matrix,
        autodiff_matrix,
        precomputed_dual,
        scratch_dual,
        Y_dual,
        Yₜ_dual,
        I_matrix_partitions,
        band_matrix_row_index_to_colors_map,
    )
end

function update_jacobian!(::AutoSparseJacobian, cache, Y, p, dtγ, t)
    (; matrix, tendency_matrix, autodiff_matrix) = cache
    (; precomputed_dual, scratch_dual, Y_dual, Yₜ_dual) = cache
    (; I_matrix_partitions, band_matrix_row_index_to_colors_map) = cache

    device = ClimaComms.device(Y.c)
    column_indices = column_index_iterator(Y)
    scalar_names = scalar_field_names(Y)
    p_dual = append_to_atmos_cache(p, precomputed_dual, scratch_dual)

    for (partition_index, partition_εs) in enumerate(I_matrix_partitions)
        # Set the εs in Y_dual to represent a partition of the identity matrix.
        Y_dual .= Y .+ partition_εs

        # Compute ∂p/∂Y * I_matrix_partition and ∂Yₜ/∂Y * I_matrix_partition.
        set_implicit_precomputed_quantities!(Y_dual, p_dual, t)
        implicit_tendency!(Yₜ_dual, Y_dual, p_dual, t)

        # Move the entries of ∂Yₜ/∂Y * I_matrix_partition from Yₜ_dual into the
        # blocks of autodiff_matrix. Drop spatial information from every Field
        # to ensure that this kernel stays below the GPU parameter memory limit.
        Yₜ_dual_data =
            unrolled_map(Fields.field_values, Fields._values(Yₜ_dual))
        matrix_fields_data =
            unrolled_map(Fields.field_values, values(autodiff_matrix))
        ClimaComms.@threaded device begin
            # On multithreaded devices, use one thread for each band matrix row.
            # TODO: Modify the map and use one thread for each dual number.
            for column_index in column_indices,
                index_pair in band_matrix_row_index_to_colors_map

                ((block_index, level_index), (scalar_index, entry_colors)) =
                    index_pair
                dual_number =
                    unrolled_applyat(scalar_index, scalar_names) do name
                        data = MatrixFields.get_field(Yₜ_dual_data, name)
                        @inbounds point(data, level_index, column_index...)[]
                    end
                ε_coefficients = ForwardDiff.partials(dual_number)
                n_εs = length(ε_coefficients)
                ε_offset = (partition_index - 1) * n_εs
                unrolled_applyat(block_index, matrix_fields_data) do block_data
                    @inbounds entries_data =
                        point(block_data, level_index, column_index...).entries
                    entries_data[] =
                        map(entry_colors, entries_data[]) do entry_color, entry
                            # If the entry has a color in the current partition,
                            # set the entry to the ε coefficient for that color.
                            # Otherwise, keep the value from the block's data.
                            ε_offset < entry_color <= ε_offset + n_εs ?
                            (@inbounds ε_coefficients[entry_color - ε_offset]) :
                            entry
                        end # TODO: Why does unrolled_map break GPU compilation?
                end
            end
        end
    end
    # TODO: Figure out why this is currently 2--3 orders of magnitude more
    # expensive than any other kernel we are launching on GPUs.

    # Update the matrix for ∂R/∂Y using the new values of ∂Yₜ/∂Y.
    matrix .= dtγ .* tendency_matrix .- one(matrix)
end

invert_jacobian!(alg::AutoSparseJacobian, cache, ΔY, R) =
    invert_jacobian!(alg.sparse_jacobian_alg, cache, ΔY, R)
