"""
    print_jacobian_summary(integrator)

Print a collection of tables that summarize the sparsity patterns and typical
block magnitudes of different Jacobian algorithms, comparing all available
sparse approximations against the dense matrix.
"""
function print_jacobian_summary(integrator)
    Y = integrator.u
    t = integrator.t
    (; p) = integrator
    timestepper_alg = integrator.alg
    tableau_coefficients =
        timestepper_alg isa CA.CTS.RosenbrockAlgorithm ?
        timestepper_alg.tableau.Γ : timestepper_alg.tableau.a_imp
    jacobian =
        timestepper_alg isa CA.CTS.RosenbrockAlgorithm ? integrator.cache.W :
        integrator.cache.newtons_method_cache.j

    FT = eltype(Y)
    γs = filter(!iszero, CA.LinearAlgebra.diag(tableau_coefficients))
    dtγ = p.dt * FT(γs[end])
    scalar_names = CA.scalar_field_names(Y)
    block_keys = Iterators.product(scalar_names, scalar_names)

    dense_jacobian_alg = CA.AutoDenseJacobian()
    all_sparse_jacobian_algs =
        jacobian.alg isa CA.AutoSparseJacobian ?
        (; manual = jacobian.alg.sparse_jacobian_alg, auto = jacobian.alg) :
        (; manual = jacobian.alg)
    dense_blocks =
        CA.first_column_block_arrays(dense_jacobian_alg, Y, p, dtγ, t)
    all_sparse_blocks = map(all_sparse_jacobian_algs) do jacobian_alg
        CA.first_column_block_arrays(jacobian_alg, Y, p, dtγ, t)
    end
    block_rescalings = CA.first_column_rescaling_arrays(Y, p, t)

    highlighters = (
        Highlighter((d, i, j) -> d[i, j] < 1e-12; foreground = :dark_gray),
        Highlighter((d, i, j) -> 1e-12 <= d[i, j] < 1e-6; foreground = :blue),
        Highlighter((d, i, j) -> 1e-6 <= d[i, j] < 1e-3; foreground = :cyan),
        Highlighter((d, i, j) -> 1e-3 <= d[i, j] < 1e-1; foreground = :green),
        Highlighter((d, i, j) -> 1e-1 <= d[i, j] < 1; foreground = :yellow),
        Highlighter((d, i, j) -> d[i, j] == 1; foreground = :light_red),
        Highlighter((d, i, j) -> d[i, j] > 1; foreground = :light_magenta),
    )
    row_labels = map(collect(scalar_names)) do name
        replace(
            string(name),
            "@name" => "",
            "components.data." => "",
            ['(', ')', ':'] => "",
            'ₕ' => "_h",
            '₃' => "_3",
            '⁰' => "^0",
            'ʲ' => "^j",
        )
    end
    table_kwargs = (;
        columns_width = 5,
        crop = :none,
        formatters = ft_printf("%1.0e"),
        highlighters,
        row_labels,
        show_header = false,
        tf = tf_matrix,
        vlines = [1],
    )
    dense_table_kwargs = (; table_kwargs..., highlighters = highlighters[1])
    bandwidth_table_kwargs = (; dense_table_kwargs..., formatters = nothing)

    bandwidth(block) =
        count((1 - size(block, 1)):(size(block, 2) - 1)) do band_index
            any(!iszero, diag(block, band_index))
        end

    dense_bandwidth_values = map(block_keys) do block_key
        bandwidth(dense_blocks[block_key])
    end
    sparse_bandwidth_values = map(block_keys) do block_key
        sparse_blocks = first(all_sparse_blocks)
        haskey(sparse_blocks, block_key) ?
        (
            sparse_blocks[block_key] isa UniformScaling ? 1 :
            bandwidth(sparse_blocks[block_key])
        ) : 0
    end
    missing_bandwidth_values =
        max.(dense_bandwidth_values .- sparse_bandwidth_values, 0)
    @info "dense, number of nonzero bands per block:"
    pretty_table(dense_bandwidth_values; bandwidth_table_kwargs...)
    @info "sparse, number of nonzero bands per block:"
    pretty_table(sparse_bandwidth_values; bandwidth_table_kwargs...)
    @info "dense - sparse, number of missing nonzero bands per block:"
    pretty_table(missing_bandwidth_values; bandwidth_table_kwargs...)
    println("<$('='^70)>\n")

    rms(block) = sqrt(mean(abs2.(block)))

    dense_rms_values = map(block_keys) do block_key
        rms(dense_blocks[block_key])
    end
    normalized_dense_rms_values = map(block_keys) do block_key
        rms(dense_blocks[block_key] .* block_rescalings[block_key])
    end
    @info "unnormalized dense, RMS per block:"
    pretty_table(dense_rms_values; dense_table_kwargs...)
    @info "normalized dense, RMS per block [s^-1]:"
    pretty_table(normalized_dense_rms_values; dense_table_kwargs...)
    println("<$('='^70)>\n")

    if jacobian.alg isa CA.AutoSparseJacobian
        normalized_sparse_difference_rms_values = map(block_keys) do block_key
            (; manual, auto) = all_sparse_blocks
            rescaling = block_rescalings[block_key]
            haskey(manual, block_key) &&
                !(manual[block_key] isa UniformScaling) ?
            rms((manual[block_key] - auto[block_key]) .* rescaling) : FT(0)
        end
        @info "normalized manual sparse - auto sparse, RMS per block [s^-1]:"
        pretty_table(normalized_sparse_difference_rms_values; table_kwargs...)
    end
    for (sparse_name, sparse_blocks) in pairs(all_sparse_blocks)
        normalized_sparse_error_rms_values = map(block_keys) do block_key
            sparse_error =
                haskey(sparse_blocks, block_key) ?
                dense_blocks[block_key] - sparse_blocks[block_key] :
                dense_blocks[block_key]
            rescaling = block_rescalings[block_key]
            rms(sparse_error .* rescaling)
        end
        @info "normalized dense - $sparse_name sparse, RMS per block [s^-1]:"
        pretty_table(normalized_sparse_error_rms_values; table_kwargs...)
    end
    println("<$('='^70)>\n")

    if jacobian.alg isa CA.AutoSparseJacobian
        sparse_difference_relative_rms_values = map(block_keys) do block_key
            (; manual, auto) = all_sparse_blocks
            rescaling = block_rescalings[block_key]
            sparse_difference_rms_value =
                haskey(manual, block_key) &&
                !(manual[block_key] isa UniformScaling) ?
                rms((manual[block_key] - auto[block_key])) : FT(0)
            sparse_difference_rms_value == 0 ? FT(0) :
            sparse_difference_rms_value / rms(dense_blocks[block_key])
        end
        @info "manual sparse - auto sparse, relative RMS per block [unitless]:"
        pretty_table(sparse_difference_relative_rms_values; table_kwargs...)
    end
    for (sparse_name, sparse_blocks) in pairs(all_sparse_blocks)
        sparse_error_relative_rms_values = map(block_keys) do block_key
            sparse_error_rms_value =
                haskey(sparse_blocks, block_key) ?
                rms(dense_blocks[block_key] - sparse_blocks[block_key]) :
                rms(dense_blocks[block_key])
            sparse_error_rms_value == 0 ? FT(0) :
            sparse_error_rms_value / rms(dense_blocks[block_key])
        end
        @info "dense - $sparse_name sparse, relative RMS per block [unitless]:"
        pretty_table(sparse_error_relative_rms_values; table_kwargs...)
    end
end
