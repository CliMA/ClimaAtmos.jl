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

    highlighters = [
        TextHighlighter((d, i, j) -> d[i, j] < 1e-12, crayon"dark_gray"),
        TextHighlighter((d, i, j) -> 1e-12 <= d[i, j] < 1e-6, crayon"blue"),
        TextHighlighter((d, i, j) -> 1e-6 <= d[i, j] < 1e-3, crayon"cyan"),
        TextHighlighter((d, i, j) -> 1e-3 <= d[i, j] < 1e-1, crayon"green"),
        TextHighlighter((d, i, j) -> 1e-1 <= d[i, j] < 1, crayon"yellow"),
        TextHighlighter((d, i, j) -> d[i, j] == 1, crayon"light_red"),
        TextHighlighter((d, i, j) -> d[i, j] > 1, crayon"light_magenta"),
    ]
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
        fixed_data_column_widths = 5,
        formatters = [fmt__printf("%1.0e")],
        highlighters = highlighters,
        row_labels = row_labels,
        show_column_labels = false,
        table_format = PrettyTables.text_table_format__matrix,
    )
    dense_table_kwargs = (; table_kwargs..., highlighters = [highlighters[1]])
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

    # Aliasing risk audit for seed scaling: normalize the dense blocks by the
    # ratios of the seed scales (instead of the state-dependent tendency
    # ratios used above), and list the blocks outside of the sparsity
    # structure whose seed-scale-normalized magnitudes are not negligible
    # compared to the significance threshold 1/dt. These are the blocks whose
    # entries can alias into same-colored columns of the sparsity structure;
    # they measure what the padding band comments in auto_sparse_jacobian.jl
    # can only estimate, and they indicate where the seed scales in
    # default_jacobian_seed_scale need to be adjusted.
    seed_scaling =
        jacobian.alg isa CA.AutoSparseJacobian ? jacobian.alg.seed_scaling :
        nothing
    if !isnothing(seed_scaling)
        uₕ_scale = CA.uₕ_seed_scale(Y)
        seed_scales = Dict(
            name => CA.seed_scale(FT, name, seed_scaling, uₕ_scale) for
            name in scalar_names
        )
        seed_normalized_rms(block_key) =
            rms(dense_blocks[block_key]) * seed_scales[block_key[2]] /
            seed_scales[block_key[1]]
        seed_normalized_dense_rms_values = map(block_keys) do block_key
            seed_normalized_rms(block_key)
        end
        @info "seed-scale-normalized dense, RMS per block [s^-1]:"
        pretty_table(seed_normalized_dense_rms_values; dense_table_kwargs...)
        manual_blocks = all_sparse_blocks.manual
        aliasing_risk_threshold = 1e-3 / FT(float(p.dt))
        aliasing_risk_keys = filter(collect(block_keys)) do block_key
            is_in_structure =
                haskey(manual_blocks, block_key) &&
                !(manual_blocks[block_key] isa UniformScaling)
            !is_in_structure &&
                seed_normalized_rms(block_key) > aliasing_risk_threshold
        end
        if isempty(aliasing_risk_keys)
            @info "All blocks outside of the sparsity structure have \
                   negligible seed-scale-normalized magnitudes"
        else
            @info "Blocks outside of the sparsity structure whose \
                   seed-scale-normalized magnitudes exceed 1e-3 / dt (these \
                   can alias into same-colored columns; adjust their seed \
                   scales or add padding bands):"
            for block_key in aliasing_risk_keys
                (block_row_name, block_column_name) = block_key
                value = seed_normalized_rms(block_key)
                @info "    ∂Yₜ($block_row_name)/∂Y($block_column_name): \
                       $value s^-1"
            end
        end
        println("<$('='^70)>\n")
    end

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
