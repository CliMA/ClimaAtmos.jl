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
                ) do approx_data, exact_data
                    (t, approx_title, approx_∂Yₜ_∂Y, approx_rescaled_∂Yₜ_∂Y, Yₜ) = approx_data
                    (_, _, exact_∂Yₜ_∂Y, exact_rescaled_∂Yₜ_∂Y, _) =
                        exact_data
                    @assert exact_data[1] == t
                    @assert exact_data[5] == Yₜ
                    error_title = "Error of $approx_title"
                    error_∂Yₜ_∂Y = approx_∂Yₜ_∂Y .- exact_∂Yₜ_∂Y
                    error_rescaled_∂Yₜ_∂Y =
                        approx_rescaled_∂Yₜ_∂Y .- exact_rescaled_∂Yₜ_∂Y
                    (t, error_title, error_∂Yₜ_∂Y, error_rescaled_∂Yₜ_∂Y, Yₜ)
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

function read_time_series(file_path, context)
    time_series = nothing
    HDF5Reader(file_path, context) do reader
        t_strings =
            sort(keys(reader.file); by = t_string -> parse(Float64, t_string))
        time_series = map(t_strings) do t_string
            data_at_t = reader.file[t_string]
            title = data_at_t["title"][]
            ∂Yₜ_∂Y = data_at_t["∂Yₜ_∂Y"][]
            Yₜ = data_at_t["Yₜ"][]

            safe_inv(x) = iszero(x) || issubnormal(x) ? zero(x) : inv(x)
            rescaled_∂Yₜ_∂Y = safe_inv.(Yₜ) .* ∂Yₜ_∂Y .* Yₜ'

            # Take the transpose of each matrix so that its rows are plotted
            # along the y-axis and its columns are plotted along the x-axis.
            (parse(Float64, t_string), title, ∂Yₜ_∂Y', rescaled_∂Yₜ_∂Y', Yₜ)
        end
    end
    return time_series
end

function plot_jacobian(output_path, time_series, file_name, Yₜ_end, dt)
    times = getindex.(time_series, 1)
    titles = getindex.(time_series, 2)
    ∂Yₜ_∂Ys = getindex.(time_series, 3)
    rescaled_∂Yₜ_∂Ys = getindex.(time_series, 4)
    Yₜs = getindex.(time_series, 5)
    FT = eltype(Yₜ_end)

    field_names = collect(CA.scalar_field_names(Yₜ_end))
    tick_labels = map(field_names) do field_name
        replace(
            string(field_name),
            "@name(" => "",
            ".components.data" => "",
            ":(" => "",
            ')' => "",
        )
    end

    index_ranges = collect(CA.scalar_field_index_ranges(Yₜ_end))
    first_tick_positions = first.(index_ranges)
    last_tick_positions = last.(index_ranges)
    center_tick_positions = (first_tick_positions .+ last_tick_positions) ./ 2
    boundary_tick_positions = [FT(0.5), (last_tick_positions .+ FT(0.5))...]
    limit_padding = length(index_ranges[1]) / 20
    limit_positions =
        extrema(boundary_tick_positions) .+ (-limit_padding, limit_padding)

    sign_or_nan(x) = iszero(x) ? FT(NaN) : sign(x)
    logabs_or_nan(x) = iszero(x) ? FT(NaN) : log10(abs(x))
    function block_bandwidth(block)
        band_indices = (1 - size(block, 1)):(size(block, 2) - 1)
        nonempty_band_indices = filter(band_indices) do band_index
            any(!iszero, diag(block, band_index))
        end
        main_diagonal_index = (band_indices[1] + band_indices[end]) / 2
        return maximum(nonempty_band_indices; init = 0) do band_index
            2 * abs(band_index - main_diagonal_index) + 1
        end
    end

    entry_sign_transform(matrix) = sign_or_nan.(matrix)
    entry_logabs_transform(matrix) = logabs_or_nan.(matrix)
    function block_logabs_transform(matrix)
        block_max_matrix = similar(matrix)
        for row_index_range in index_ranges, col_index_range in index_ranges
            block_max_matrix[row_index_range, col_index_range] .=
                CA.smooth_maximum(matrix[row_index_range, col_index_range])
        end
        return entry_logabs_transform(block_max_matrix)
    end
    function block_row_logabs_transform(matrix)
        row_max_matrix = similar(matrix)
        for row_index_range in index_ranges, col in axes(matrix, 1)
            row_max_matrix[row_index_range, col] .=
                CA.smooth_maximum(matrix[row_index_range, col])
        end
        return entry_logabs_transform(row_max_matrix)
    end
    function block_bandwidth_transform(matrix)
        block_bandwidth_matrix = similar(matrix, Int)
        for row_index_range in index_ranges, col_index_range in index_ranges
            block_bandwidth_matrix[row_index_range, col_index_range] .=
                block_bandwidth(matrix[row_index_range, col_index_range])
        end
        return block_bandwidth_matrix
    end

    bandwidth_matrices = map(block_bandwidth_transform, ∂Yₜ_∂Ys)
    max_bandwidth = maximum(maximum, bandwidth_matrices)
    sign_matrices = map(entry_sign_transform, ∂Yₜ_∂Ys)

    categorical(colormap, n) = CairoMakie.cgrad(colormap, n; categorical = true)
    main_colormap = categorical(:tol_iridescent, 21)
    bandwidth_colors = [CairoMakie.RGB(1, 1, 1), main_colormap[1:(end - 1)]...]
    bandwidth_colormap =
        categorical(bandwidth_colors, min(max_bandwidth, 9) + 1)
    sign_colormap = categorical(:RdBu_5, 2)
    rescaling_colors =
        setindex!(CairoMakie.to_colormap(:RdBu_11), CairoMakie.RGB(1, 1, 1), 6)
    rescaling_colormap = categorical(rescaling_colors, 21)

    value_for_min(x) = isnan(x) ? FT(Inf) : x
    value_for_max(x) = isnan(x) ? FT(-Inf) : x

    abs_Yₜs = map(Yₜs) do Yₜ
        @. ifelse(iszero(Yₜ) || issubnormal(Yₜ), FT(NaN), abs(Yₜ))
    end
    if all(abs_Yₜ -> all(isnan, abs_Yₜ), abs_Yₜs)
        min_abs_Yₜ = FT(1e-9)
        max_abs_Yₜ = FT(1)
    else
        min_abs_Yₜ = minimum(abs_Yₜ -> minimum(value_for_min, abs_Yₜ), abs_Yₜs)
        max_abs_Yₜ = maximum(abs_Yₜ -> maximum(value_for_max, abs_Yₜ), abs_Yₜs)
    end

    figure_kwargs = (;
        size = (2.2, cld(length(times), 2)) .* 5000,
        figure_padding = 300,
        fontsize = 150,
    )
    colorbar_kwargs = (;
        size = 150,
        labelpadding = 60,
        ticklabelpad = 50,
        ticksize = 50,
        tickwidth = 10,
        spinewidth = 10,
    )
    axis_kwargs = (;
        titlegap = 100,
        xlabelpadding = 60,
        ylabelpadding = 60,
        xticks = (center_tick_positions, tick_labels),
        xticksvisible = false,
        xticklabelrotation = pi / 4,
        xticklabelpad = 50,
        yticklabelpad = 80,
        xminorticks = boundary_tick_positions,
        xminorticksvisible = true,
        xminorticksize = 50,
        yminorticksize = 50,
        xminortickwidth = 10,
        yminortickwidth = 10,
        xgridvisible = false,
        xminorgridvisible = true,
        xminorgridwidth = 10,
        yminorgridwidth = 10,
        spinewidth = 10,
    )
    ∂Yₜ_∂Y_axis_kwargs = (;
        axis_kwargs...,
        limits = (limit_positions, limit_positions),
        xlabel = "Y index",
        ylabel = "Yₜ index",
        yreversed = true,
        yticks = (center_tick_positions, tick_labels),
        yticksvisible = false,
        yminorticks = boundary_tick_positions,
        yminorticksvisible = true,
        ygridvisible = false,
        yminorgridvisible = true,
    ) # Flip the y-axis so that the diagonal runs from top-left to bottom-right.
    Yₜ_axis_kwargs = (;
        axis_kwargs...,
        limits = (limit_positions, (min_abs_Yₜ / 2, max_abs_Yₜ * 2)),
        xlabel = "Yₜ index",
        ylabel = "Yₜ magnitude",
        yscale = log10,
        yticksize = 50,
    )

    page_file_paths = map(n -> joinpath(output_path, "$file_name $n.pdf"), 1:6)
    full_file_path = joinpath(output_path, "$file_name.pdf")

    figure = CairoMakie.Figure(; figure_kwargs...)
    for (t_index, (t, title, bandwidth_matrix, sign_matrix)) in
        enumerate(zip(times, titles, bandwidth_matrices, sign_matrices))
        grid_position = figure[cld(t_index, 2), (t_index - 1) % 2 + 1]
        grid_layout = CairoMakie.GridLayout(grid_position)
        axis = CairoMakie.Axis(grid_layout[1, 1]; title, ∂Yₜ_∂Y_axis_kwargs...)
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

    for (page_index, (is_rescaled, transform, transform_string)) in enumerate((
        (true, block_logabs_transform, "block"),
        (true, block_row_logabs_transform, "block row"),
        (true, entry_logabs_transform, "entry"),
        (false, entry_logabs_transform, "entry"),
    ))
        matrices = map(transform, is_rescaled ? rescaled_∂Yₜ_∂Ys : ∂Yₜ_∂Ys)
        max_logabs =
            all(matrix -> all(isnan, matrix), matrices) ? FT(0) :
            maximum(matrix -> maximum(value_for_max, matrix), matrices)
        if is_rescaled
            dt_tick_value = log10(1 / dt)
            min_logabs = min(max_logabs - 9, dt_tick_value - FT(0.5))
        else
            min_logabs = max_logabs - 19
        end
        colorbar_range = (min_logabs, max_logabs)
        wilkinson_kwargs = (; k_min = 4, k_max = 6, niceness_weight = 1)
        colorbar_ticks = Makie.WilkinsonTicks(5; wilkinson_kwargs...)
        colorbar_tick_values =
            Makie.get_tickvalues(colorbar_ticks, colorbar_range...)
        colorbar_tick_labels = map(colorbar_tick_values) do tick_value
            Makie.latexstring("10^{$(round(Int, tick_value))}")
        end
        if is_rescaled
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
        figure = CairoMakie.Figure(; figure_kwargs...)
        for (t_index, (t, title, matrix)) in
            enumerate(zip(times, titles, matrices))
            grid_position = figure[cld(t_index, 2), (t_index - 1) % 2 + 1]
            grid_layout = CairoMakie.GridLayout(grid_position)
            axis =
                CairoMakie.Axis(grid_layout[1, 1]; title, ∂Yₜ_∂Y_axis_kwargs...)
            plot_args = if transform == block_logabs_transform
                submatrix = matrix[first_tick_positions, first_tick_positions]
                (boundary_tick_positions, boundary_tick_positions, submatrix)
            elseif transform == block_row_logabs_transform
                submatrix = matrix[first_tick_positions, :]
                (boundary_tick_positions, 1:last_tick_positions[end], submatrix)
            else
                (matrix,)
            end
            plot = CairoMakie.heatmap!(
                axis,
                plot_args...;
                colormap = main_colormap,
                colorrange = colorbar_range,
                lowclip = main_colormap[1],
            ) # Only color in the top 9 orders of magnitude.
            CairoMakie.translate!(plot, 0, 0, -100) # Put plot under grid.
            label = "Magnitude of$(is_rescaled ? " rescaled" : "") \
                     $(transform_string != "entry" ? "max of " : "")matrix \
                     $transform_string$(is_rescaled ? " [s⁻¹]" : "")"
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
        CairoMakie.save(page_file_paths[page_index + 1], figure)
    end

    figure = CairoMakie.Figure(; figure_kwargs...)
    for (t_index, (t, title, abs_Yₜ)) in enumerate(zip(times, titles, abs_Yₜs))
        grid_position = figure[cld(t_index, 2), (t_index - 1) % 2 + 1]
        Yₜ_title = "Yₜ, avg over all columns," * split(title, ',')[end]
        axis =
            CairoMakie.Axis(grid_position; title = Yₜ_title, Yₜ_axis_kwargs...)
        CairoMakie.lines!(axis, axes(abs_Yₜ, 1), abs_Yₜ; linewidth = 10)
    end
    CairoMakie.colsize!(figure.layout, 1, CairoMakie.Aspect(1, 1))
    CairoMakie.colsize!(figure.layout, 2, CairoMakie.Aspect(1, 1))
    CairoMakie.colgap!(figure.layout, CairoMakie.Relative(0.12))
    CairoMakie.rowgap!(figure.layout, CairoMakie.Relative(0.05))
    CairoMakie.save(page_file_paths[end], figure)

    pdfunite() do unite
        run(Cmd([unite, page_file_paths..., full_file_path]))
    end
    for page_file_path in page_file_paths
        Filesystem.rm(page_file_path; force = true)
    end
end
