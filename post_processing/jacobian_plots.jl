import ClimaComms
import ClimaCore.InputOutput: HDF5Reader
import LinearAlgebra: diag

function make_jacobian_plots(output_path, Yₜ_end, dt)
    context = ClimaComms.context(Yₜ_end.c)
    file_paths = filter(endswith(".hdf5"), readdir(output_path; join = true))
    for file_path in file_paths
        file_name = split(basename(file_path), '.')[1]
        contains(file_name, "jacobian") || continue
        time_series = read_time_series(file_path, context)
        plot_jacobian(output_path, time_series, file_name, Yₜ_end, dt)
        if startswith(file_name, "approx_jacobian")
            exact_file_path = joinpath(
                output_path,
                replace(basename(file_path), "approx" => "exact"),
            )
            if exact_file_path in file_paths
                exact_time_series = read_time_series(exact_file_path, context)
                error_time_series = map(
                    time_series,
                    exact_time_series,
                ) do (t, approx_matrix), (_, exact_matrix)
                    (t, approx_matrix .- exact_matrix)
                end
                error_file_name =
                    replace(file_name, "jacobian" => "jacobian_error")
                plot_jacobian(
                    output_path,
                    error_time_series,
                    error_file_name,
                    Yₜ_end,
                    dt,
                )
            end
        end
    end
end

# TODO: Use the description string from each file instead of recomputing it.
function read_time_series(file_path, context)
    reader = HDF5Reader(file_path, context)
    time_series = map(keys(reader.file)) do t_string
        # Take the transpose of each Jacobian matrix so that its rows are
        # plotted along the y-axis and its columns are plotted along the x-axis.
        (parse(Float64, t_string), reader.file[t_string][]')
    end
    Base.close(reader)
    return time_series
end

function plot_jacobian(output_path, time_series, file_name, Yₜ_end, dt)
    jacobian_times = first.(time_series)
    jacobian_matrices = last.(time_series)

    # TODO: Transform the axes so that every matrix block has the same size.
    field_names = collect(CA.scalar_field_names(Yₜ_end))
    index_ranges = collect(CA.scalar_field_index_ranges(Yₜ_end))
    first_tick_positions = first.(index_ranges)
    last_tick_positions = last.(index_ranges)
    center_tick_positions = (first_tick_positions .+ last_tick_positions) ./ 2
    boundary_tick_positions = [0.5, (last_tick_positions .+ 0.5)...]

    limit_padding = length(index_ranges[1]) / 20
    limit_positions =
        extrema(boundary_tick_positions) .+ (-limit_padding, limit_padding)

    tick_labels = map(field_names) do field_name
        replace(
            string(field_name),
            "@name(" => "",
            ".components.data" => "",
            ":(" => "",
            ')' => "",
        )
    end
    axis_kwargs = (;
        limits = (limit_positions, limit_positions),
        titlegap = 100,
        yreversed = true,
        xlabel = "Y index",
        ylabel = "Yₜ index",
        xlabelpadding = 60,
        ylabelpadding = 60,
        xticks = (center_tick_positions, tick_labels),
        yticks = (center_tick_positions, tick_labels),
        xticksvisible = false,
        yticksvisible = false,
        xticklabelrotation = pi / 4,
        xticklabelpad = 50,
        yticklabelpad = 80,
        xminorticks = boundary_tick_positions,
        yminorticks = boundary_tick_positions,
        xminorticksvisible = true,
        yminorticksvisible = true,
        xminorticksize = 50,
        yminorticksize = 50,
        xminortickwidth = 10,
        yminortickwidth = 10,
        xgridvisible = false,
        ygridvisible = false,
        xminorgridvisible = true,
        yminorgridvisible = true,
        xminorgridwidth = 10,
        yminorgridwidth = 10,
        spinewidth = 10,
    ) # Flip the y-axis so that the diagonal runs from top-left to bottom-right.
    colorbar_kwargs = (;
        size = 150,
        labelpadding = 60,
        ticklabelpad = 50,
        ticksize = 50,
        tickwidth = 10,
        spinewidth = 10,
    )
    figure_kwargs = (; figure_padding = 300, fontsize = 150)
    plot_size = 5000
    figure_size = (2.2, cld(length(jacobian_times), 2)) .* plot_size

    typical_tendency_values = Dict(
        CA.MatrixFields.@name(c.ρ) => 1e-8,
        CA.MatrixFields.@name(c.uₕ.components.data.:1) => 1e-6,
        CA.MatrixFields.@name(c.uₕ.components.data.:2) => 1e-6,
        CA.MatrixFields.@name(c.ρe_tot) => 1e-1,
        CA.MatrixFields.@name(c.ρq_tot) => 1e-7,
        CA.MatrixFields.@name(c.ρq_liq) => 1e-8,
        CA.MatrixFields.@name(c.ρq_ice) => 1e-8,
        CA.MatrixFields.@name(c.ρq_rai) => 1e-8,
        CA.MatrixFields.@name(c.ρq_sno) => 1e-8,
        CA.MatrixFields.@name(c.sgs⁰.ρatke) => 1e-3,
        CA.MatrixFields.@name(c.sgsʲs.:(1).ρa) => 1e-5,
        CA.MatrixFields.@name(c.sgsʲs.:(1).mse) => 1e-1,
        CA.MatrixFields.@name(c.sgsʲs.:(1).q_tot) => 1e-7,
        CA.MatrixFields.@name(c.sgsʲs.:(1).q_liq) => 1e-8,
        CA.MatrixFields.@name(c.sgsʲs.:(1).q_ice) => 1e-8,
        CA.MatrixFields.@name(c.sgsʲs.:(1).q_rai) => 1e-8,
        CA.MatrixFields.@name(c.sgsʲs.:(1).q_sno) => 1e-8,
        CA.MatrixFields.@name(f.u₃.components.data.:1) => 1e-2,
        CA.MatrixFields.@name(f.sgsʲs.:(1).u₃.components.data.:1) => 1e-2,
        CA.MatrixFields.@name(sfc.T) => 1e-3,
        CA.MatrixFields.@name(sfc.water) => 1e-7,
    )
    rescaling_vector = ClimaComms.allowscalar(
        Vector,
        ClimaComms.device(Yₜ_end.c), # ClimaComms.device(Yₜ_end),
        CA.Fields.column(Yₜ_end, 1, 1, 1),
    )
    for (field_name, index_range) in zip(field_names, index_ranges)
        block_max = CA.smooth_maximum(rescaling_vector[index_range])
        rescaling_vector[index_range] .=
            block_max < 1e-10 ? typical_tendency_values[field_name] : block_max
    end
    rescaling_matrix = (inv.(rescaling_vector) * rescaling_vector')'
    # Take the transpose of the rescaling matrix so that its rows and columns
    # are consistent with the Jacobian matrices.

    sign_or_nan(x) = iszero(x) ? typeof(x)(NaN) : sign(x)
    logabs_or_nan(x) = iszero(x) ? typeof(x)(NaN) : log10(abs(x))
    entry_sign(matrix) = sign_or_nan.(matrix)
    entry_logabs(matrix) = logabs_or_nan.(matrix)
    entry_logabs_rescaled(matrix) = entry_logabs(rescaling_matrix .* matrix)
    function block_logabs_rescaled(matrix)
        block_max_matrix = similar(matrix)
        for row_index_range in index_ranges, col_index_range in index_ranges
            block_max_matrix[row_index_range, col_index_range] .=
                CA.smooth_maximum(matrix[row_index_range, col_index_range])
        end
        return entry_logabs_rescaled(block_max_matrix)
    end
    function row_logabs_rescaled(matrix)
        row_max_matrix = similar(matrix)
        for row_index_range in index_ranges, col in axes(matrix, 1)
            row_max_matrix[row_index_range, col] .=
                CA.smooth_maximum(matrix[row_index_range, col])
        end
        return entry_logabs_rescaled(row_max_matrix)
    end
    function block_bandwidth(matrix)
        block_bandwidth_matrix = similar(matrix, Int)
        for row_index_range in index_ranges, col_index_range in index_ranges
            block = matrix[row_index_range, col_index_range]
            band_indices =
                (1 - length(row_index_range)):(length(col_index_range) - 1)
            nonempty_band_indices = filter(
                band_index -> any(!iszero, diag(block, band_index)),
                band_indices,
            )
            main_diagonal_index = (band_indices[1] + band_indices[end]) / 2
            block_bandwidth_matrix[row_index_range, col_index_range] .=
                isempty(nonempty_band_indices) ? 0 :
                maximum(
                    band_index -> 2 * abs(band_index - main_diagonal_index) + 1,
                    nonempty_band_indices,
                )
        end
        return block_bandwidth_matrix
    end

    bandwidth_matrices = map(block_bandwidth, jacobian_matrices)
    max_bandwidth = maximum(maximum, bandwidth_matrices)
    sign_matrices = map(entry_sign, jacobian_matrices)

    categorical(colormap, n) = CairoMakie.cgrad(colormap, n; categorical = true)
    main_colormap = categorical(:tol_iridescent, 21)
    bandwidth_colors = [CairoMakie.RGB(1, 1, 1), main_colormap[1:(end - 1)]...]
    bandwidth_colormap =
        categorical(bandwidth_colors, min(max_bandwidth, 9) + 1)
    sign_colormap = categorical(:RdBu_5, 2)
    rescaling_colors =
        setindex!(CairoMakie.to_colormap(:RdBu_11), CairoMakie.RGB(1, 1, 1), 6)
    rescaling_colormap = categorical(rescaling_colors, 21)

    first_column_field = CA.Fields.column(Yₜ_end.c, 1, 1, 1)
    coord_field =
        CA.Fields.coordinate_field(CA.Fields.level(first_column_field, 1))
    coord = ClimaComms.allowscalar(
        getindex,
        ClimaComms.device(Yₜ_end.c), # ClimaComms.device(Yₜ_end),
        coord_field,
    )
    round_value(value) = round(value; sigdigits = 3)
    column_loc_str = if coord isa CA.Geometry.XZPoint
        "x = $(round_value(coord.x)) Meters"
    elseif coord isa CA.Geometry.XYZPoint
        "x = $(round_value(coord.x)) Meters, y = $(round_value(coord.y)) Meters"
    elseif coord isa CA.Geometry.LatLongZPoint
        "lat = $(round_value(coord.lat))°, long = $(round_value(coord.long))°"
    else
        error("Unrecognized coordinate type $(typeof(coord))")
    end

    page_file_paths = map(n -> joinpath(output_path, "$file_name $n.pdf"), 1:6)
    full_file_path = joinpath(output_path, "$file_name.pdf")

    figure = CairoMakie.Figure(; size = figure_size, figure_kwargs...)
    for (index, (t, bandwidth_matrix, sign_matrix)) in
        enumerate(zip(jacobian_times, bandwidth_matrices, sign_matrices))
        grid_position = figure[cld(index, 2), (index - 1) % 2 + 1]
        grid_layout = CairoMakie.GridLayout(grid_position)
        title = "∂Yₜ/∂Y at $column_loc_str, t = $(CA.time_and_units_str(t))"
        axis = CairoMakie.Axis(grid_layout[1, 1]; title, axis_kwargs...)
        bandwidth_plot = CairoMakie.heatmap!(
            axis,
            boundary_tick_positions,
            boundary_tick_positions,
            bandwidth_matrix[first_tick_positions, first_tick_positions];
            colormap = bandwidth_colormap,
            colorrange = (-0.5, min(max_bandwidth, 9) + 0.5),
            (max_bandwidth > 9 ? (; highclip = :black) : (;))...,
        )
        CairoMakie.translate!(bandwidth_plot, 0, 0, -100) # Put plot under grid.
        sign_plot = CairoMakie.heatmap!(
            axis,
            sign_matrix;
            colormap = sign_colormap,
            colorrange = (-1, 1),
        )
        grid_sublayout = CairoMakie.GridLayout(grid_layout[1, 2])
        CairoMakie.Colorbar(
            grid_sublayout[1, 1],
            bandwidth_plot;
            label = "Bandwidth of matrix block",
            colorbar_kwargs...,
            (max_bandwidth < 4 ? (; ticks = [(0:max_bandwidth)...]) : (;))...,
        )
        CairoMakie.Colorbar(
            grid_sublayout[2, 1],
            sign_plot;
            label = "Sign of matrix entry",
            ticks = ([-0.5, 0.5], Makie.latexstring.(["-", "+"])),
            colorbar_kwargs...,
            labelpadding = 27,
        )
        CairoMakie.colgap!(grid_layout, CairoMakie.Relative(0.05))
        CairoMakie.rowgap!(grid_sublayout, CairoMakie.Relative(0.05))
    end
    CairoMakie.colsize!(figure.layout, 1, CairoMakie.Aspect(1, 1))
    CairoMakie.colsize!(figure.layout, 2, CairoMakie.Aspect(1, 1))
    CairoMakie.colgap!(figure.layout, CairoMakie.Relative(0.08))
    CairoMakie.rowgap!(figure.layout, CairoMakie.Relative(0.05))
    CairoMakie.save(page_file_paths[1], figure)

    colorbar_ticks =
        CairoMakie.WilkinsonTicks(5; k_min = 4, k_max = 6, niceness_weight = 1)
    colorbar_tickformat =
        Base.Fix1(map, x -> Makie.latexstring("10^{$(round(Int, x))}"))
    for (index, (logabs_transform, value_type_str, is_rescaled)) in enumerate((
        (block_logabs_rescaled, "block", true),
        (row_logabs_rescaled, "block row", true),
        (entry_logabs_rescaled, "entry", true),
        (entry_logabs, "entry", false),
    ))
        matrices = map(logabs_transform, jacobian_matrices)
        max_logabs =
            maximum(matrix -> maximum(filter(!isnan, matrix)), matrices)
        figure = CairoMakie.Figure(; size = figure_size, figure_kwargs...)
        for (index, (t, matrix)) in enumerate(zip(jacobian_times, matrices))
            grid_position = figure[cld(index, 2), (index - 1) % 2 + 1]
            grid_layout = CairoMakie.GridLayout(grid_position)
            title = "∂Yₜ/∂Y at $column_loc_str, t = $(CA.time_and_units_str(t))"
            axis = CairoMakie.Axis(grid_layout[1, 1]; title, axis_kwargs...)
            plot_args = if value_type_str == "block"
                submatrix = matrix[first_tick_positions, first_tick_positions]
                (boundary_tick_positions, boundary_tick_positions, submatrix)
            elseif value_type_str == "block row"
                submatrix = matrix[first_tick_positions, :]
                (boundary_tick_positions, 1:last_tick_positions[end], submatrix)
            else
                (matrix,)
            end
            colorbar_range = (max_logabs - 9, max_logabs)
            plot = CairoMakie.heatmap!(
                axis,
                plot_args...;
                colormap = main_colormap,
                colorrange = colorbar_range,
                lowclip = main_colormap[1],
            ) # Only color in the top 9 orders of magnitude.
            CairoMakie.translate!(plot, 0, 0, -100) # Put plot under grid.
            is_max = value_type_str != "entry"
            label = "Magnitude of$(is_rescaled ? " rescaled" : "") \
                     $(is_max ? "max of " : "")matrix \
                     $value_type_str$(is_rescaled ? " [s⁻¹]" : "")"
            colorbar_tick_values =
                Makie.get_tickvalues(colorbar_ticks, colorbar_range...)
            colorbar_tick_labels = colorbar_tickformat(colorbar_tick_values)
            if is_rescaled
                dt_tick_value = log10(1 / dt)
                dt_tick_label = Makie.latexstring("\\mathbf{Δt^{-1}}")
                min_tick_value_distance, colorbar_tick_index =
                    findmin(colorbar_tick_values) do colorbar_tick_value
                        abs(colorbar_tick_value - dt_tick_value)
                    end
                if min_tick_value_distance < 0.5
                    colorbar_tick_values[colorbar_tick_index] = dt_tick_value
                    colorbar_tick_labels[colorbar_tick_index] = dt_tick_label
                else
                    push!(colorbar_tick_values, dt_tick_value)
                    push!(colorbar_tick_labels, dt_tick_label)
                end
            end
            CairoMakie.Colorbar(
                grid_layout[1, 2],
                plot;
                label,
                ticks = (colorbar_tick_values, colorbar_tick_labels),
                colorbar_kwargs...,
            )
            CairoMakie.colgap!(grid_layout, CairoMakie.Relative(0.05))
        end
        CairoMakie.colsize!(figure.layout, 1, CairoMakie.Aspect(1, 1))
        CairoMakie.colsize!(figure.layout, 2, CairoMakie.Aspect(1, 1))
        CairoMakie.colgap!(figure.layout, CairoMakie.Relative(0.08))
        CairoMakie.rowgap!(figure.layout, CairoMakie.Relative(0.05))
        CairoMakie.save(page_file_paths[index + 1], figure)
    end

    figure = CairoMakie.Figure(; size = (2.2, 1) .* plot_size, figure_kwargs...)

    grid_position = figure[1, 1]
    title = "Rescaling vector for Yₜ"
    ylimits = (minimum(rescaling_vector) / 2, maximum(rescaling_vector) * 2)
    vector_axis_kwargs = (;
        limits = (limit_positions, ylimits),
        titlegap = 100,
        yscale = log10,
        xlabel = "Yₜ index",
        ylabel = "Vector entry",
        xlabelpadding = 60,
        ylabelpadding = 60,
        xticks = (center_tick_positions, tick_labels),
        xticksvisible = false,
        xticklabelrotation = pi / 4,
        xticklabelpad = 50,
        yticklabelpad = 50,
        xminorticks = boundary_tick_positions,
        xminorticksvisible = true,
        xminorticksize = 50,
        yticksize = 50,
        xminortickwidth = 10,
        ytickwidth = 10,
        xgridvisible = false,
        xminorgridvisible = true,
        xminorgridwidth = 10,
        ygridwidth = 10,
        spinewidth = 10,
    )
    axis = CairoMakie.Axis(grid_position; title, vector_axis_kwargs...)
    CairoMakie.lines!(
        axis,
        1:last_tick_positions[end],
        rescaling_vector;
        color = rescaling_colormap[end - 5],
        linewidth = 40,
    )

    matrix = logabs_or_nan.(rescaling_matrix)
    grid_layout = CairoMakie.GridLayout(figure[1, 2])
    title = "Rescaling matrix for ∂Yₜ/∂Y"
    axis = CairoMakie.Axis(grid_layout[1, 1]; title, axis_kwargs...)
    plot = CairoMakie.heatmap!(
        axis,
        boundary_tick_positions,
        boundary_tick_positions,
        matrix[first_tick_positions, first_tick_positions];
        colormap = rescaling_colormap,
    )
    CairoMakie.translate!(plot, 0, 0, -100) # Put plot under grid.
    CairoMakie.Colorbar(
        grid_layout[1, 2],
        plot;
        label = "Matrix entry",
        tickformat = colorbar_tickformat,
        colorbar_kwargs...,
    )
    CairoMakie.colgap!(grid_layout, CairoMakie.Relative(0.05))

    CairoMakie.colsize!(figure.layout, 1, CairoMakie.Aspect(1, 1))
    CairoMakie.colsize!(figure.layout, 2, CairoMakie.Aspect(1, 1))
    CairoMakie.colgap!(figure.layout, CairoMakie.Relative(0.15))
    CairoMakie.save(page_file_paths[end], figure)

    pdfunite() do unite
        run(Cmd([unite, page_file_paths..., full_file_path]))
    end
    for page_file_path in page_file_paths
        Filesystem.rm(page_file_path; force = true)
    end
end
