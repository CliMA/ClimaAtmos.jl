function print_jacobian_summary(integrator)
    Y = integrator.u
    t = integrator.t
    (; p, dt) = integrator
    timestepper_alg = integrator.alg
    tableau_coefficients =
        timestepper_alg isa CA.CTS.RosenbrockAlgorithm ?
        timestepper_alg.tableau.Γ : timestepper_alg.tableau.a_imp
    jacobian =
        timestepper_alg isa CA.CTS.RosenbrockAlgorithm ? integrator.cache.W :
        integrator.cache.newtons_method_cache.j

    FT = eltype(Y)
    γs = filter(!iszero, CA.LinearAlgebra.diag(tableau_coefficients))
    dtγ = FT(float(dt) * γs[end])
    scalar_names = CA.scalar_field_names(Y)
    block_keys = Iterators.product(scalar_names, scalar_names)

    exact_jacobian_alg = CA.AutoDenseJacobian()
    all_approx_jacobian_algs =
        jacobian.alg isa CA.AutoSparseJacobian ?
        (; manual = jacobian.alg.sparse_jacobian_alg, auto = jacobian.alg) :
        (; manual = jacobian.alg)
    exact_blocks =
        CA.first_column_block_arrays(exact_jacobian_alg, Y, p, dtγ, t)
    all_approx_blocks = map(all_approx_jacobian_algs) do jacobian_alg
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
    table_kwargs = (;
        columns_width = 5,
        crop = :none,
        formatters = ft_printf("%1.0e"),
        highlighters,
        row_labels = collect(scalar_names),
        show_header = false,
        tf = tf_matrix,
        vlines = [1],
    )
    exact_table_kwargs = (; table_kwargs..., highlighters = highlighters[1])
    bandwidth_table_kwargs = (; exact_table_kwargs..., formatters = nothing)

    rms(block) = sqrt(mean(abs2.(block)))
    bandwidth(block) =
        count((1 - size(block, 1)):(size(block, 2) - 1)) do band_index
            any(!iszero, diag(block, band_index))
        end

    exact_bandwidth_values = map(block_keys) do block_key
        bandwidth(exact_blocks[block_key])
    end
    approx_bandwidth_error_values = map(block_keys) do block_key
        approx_blocks = first(all_approx_blocks)
        approx_bandwidth =
            haskey(approx_blocks, block_key) ?
            (
                approx_blocks[block_key] isa UniformScaling ? 1 :
                bandwidth(approx_blocks[block_key])
            ) : 0
        max(bandwidth(exact_blocks[block_key]) - approx_bandwidth, 0)
    end
    @info "exact, number of bands per block:"
    pretty_table(exact_bandwidth_values; bandwidth_table_kwargs...)
    @info "exact - approx, number of missing bands per block:"
    pretty_table(approx_bandwidth_error_values; bandwidth_table_kwargs...)

    exact_rms_values = map(block_keys) do block_key
        rms(exact_blocks[block_key])
    end
    normalized_exact_rms_values = map(block_keys) do block_key
        rms(exact_blocks[block_key] .* block_rescalings[block_key])
    end
    @info "unnormalized exact, RMS per block:"
    pretty_table(exact_rms_values; exact_table_kwargs...)
    @info "normalized exact, RMS per block [s^-1]:"
    pretty_table(normalized_exact_rms_values; exact_table_kwargs...)

    println("<$('='^70)>\n")
    if jacobian.alg isa CA.AutoSparseJacobian
        normalized_approx_difference_rms_values = map(block_keys) do block_key
            (; manual, auto) = all_approx_blocks
            rescaling = block_rescalings[block_key]
            haskey(manual, block_key) &&
                !(manual[block_key] isa UniformScaling) ?
            rms((manual[block_key] - auto[block_key]) .* rescaling) : FT(0)
        end
        @info "normalized manual approx - auto approx, RMS per block [s^-1]:"
        pretty_table(normalized_approx_difference_rms_values; table_kwargs...)
    end
    for (approx_name, approx_blocks) in pairs(all_approx_blocks)
        normalized_approx_error_rms_values = map(block_keys) do block_key
            approx_error =
                haskey(approx_blocks, block_key) ?
                exact_blocks[block_key] - approx_blocks[block_key] :
                exact_blocks[block_key]
            rescaling = block_rescalings[block_key]
            rms(approx_error .* rescaling)
        end
        @info "normalized exact - $approx_name approx, RMS per block [s^-1]:"
        pretty_table(normalized_approx_error_rms_values; table_kwargs...)
    end
    println("<$('='^70)>\n")
    if jacobian.alg isa CA.AutoSparseJacobian
        approx_difference_relative_rms_values = map(block_keys) do block_key
            (; manual, auto) = all_approx_blocks
            rescaling = block_rescalings[block_key]
            approx_difference_rms_value =
                haskey(manual, block_key) &&
                !(manual[block_key] isa UniformScaling) ?
                rms((manual[block_key] - auto[block_key])) : FT(0)
            approx_difference_rms_value == 0 ? FT(0) :
            approx_difference_rms_value / rms(exact_blocks[block_key])
        end
        @info "manual approx - auto approx, relative RMS per block [unitless]:"
        pretty_table(approx_difference_relative_rms_values; table_kwargs...)
    end
    for (approx_name, approx_blocks) in pairs(all_approx_blocks)
        approx_error_relative_rms_values = map(block_keys) do block_key
            approx_error_rms_value =
                haskey(approx_blocks, block_key) ?
                rms(exact_blocks[block_key] - approx_blocks[block_key]) :
                rms(exact_blocks[block_key])
            approx_error_rms_value == 0 ? FT(0) :
            approx_error_rms_value / rms(exact_blocks[block_key])
        end
        @info "exact - $approx_name approx, relative RMS per block [unitless]:"
        pretty_table(approx_error_relative_rms_values; table_kwargs...)
    end
end
