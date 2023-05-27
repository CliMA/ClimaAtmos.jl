import ClimaAtmos: time_from_filename
import ClimaCore: Geometry, Spaces, Fields, InputOutput
import CairoMakie: Makie
import Statistics: mean

function best_time_unit(value_in_seconds)
    unit_options = (
        (60 * 60 * 24 * 365, "yr"),
        (60 * 60 * 24, "d"),
        (60 * 60, "h"),
        (60, "min"),
        (1, "s"),
    )
    minimum_value_in_units = 2
    index = findfirst(unit_options) do (seconds_per_unit, _)
        return value_in_seconds / seconds_per_unit >= minimum_value_in_units
    end
    return unit_options[isnothing(index) ? end : index]
end

function time_string_with_unit(value_in_seconds)
    seconds_per_unit, abbreviation = best_time_unit(value_in_seconds)
    value_in_units = round(value_in_seconds / seconds_per_unit; sigdigits = 6)
    return "$value_in_units $abbreviation"
end

# This is an extended version of the time tick locator and formatter from the
# Makie docs: https://docs.makie.org/stable/examples/blocks/axis/index.html.
struct TimeTicks end
function Makie.get_ticks(::TimeTicks, scale, formatter, vmin, vmax)
    seconds_per_unit, abbreviation = best_time_unit(max(abs(vmin), abs(vmax)))
    values_in_units = Makie.get_tickvalues(
        Makie.automatic,
        scale,
        vmin / seconds_per_unit,
        vmax / seconds_per_unit,
    )
    values_in_seconds = values_in_units .* seconds_per_unit
    labels =
        Makie.get_ticklabels(formatter, values_in_units) .* " $abbreviation"
    return values_in_seconds, labels
end

function contour_plot!(
    grid_position,
    var_col_time_series,
    ref_var_col_time_series,
    category,
    name,
    draw_ref_plot,
    label_ts,
    label_zs,
    negative_values_allowed,
)
    # There seems to be a bug in contourf! that occasionally causes the highest
    # or lowest contour band to disappear. Adding a tiny amount of padding to
    # the limits fixes this problem.
    padding_fraction = 1e-6 # Increase this if any contour bands aren't plotted.

    # There is a known issue in Makie that causes aliasing artifacts to appear
    # between adjacent bands drawn by contourf! (and apparently also Colorbar):
    # https://discourse.julialang.org/t/aliasing-with-makie-contourf/71783.
    # Redrawing plots on top of themselves fixes this problem.
    n_redraws = 2 # Increase this if any aliasing artifacts are observed.

    if isnothing(var_col_time_series)
        return
    end
    ts = getindex.(var_col_time_series, 1)
    zs = var_col_time_series[1][2]
    values = hcat(getindex.(var_col_time_series, 3)...)'
    if isnothing(ref_var_col_time_series)
        limits = extrema(values)
    else
        ref_ts = getindex.(ref_var_col_time_series, 1)
        ref_zs = ref_var_col_time_series[1][2]
        ref_values = hcat(getindex.(ref_var_col_time_series, 3)...)'
        limits = extrema((extrema(values)..., extrema(ref_values)...))
    end
    limits = limits .+ (-1, 1) .* (padding_fraction * (limits[2] - limits[1]))
    if negative_values_allowed
        extendlow = nothing
    else
        limits = max.(limits, 0)
        extendlow = :red
    end

    if limits[1] == limits[2]
        colorbar_ticks_kwarg = (; ticks = [limits[1]])

        # Since contourf! requires nonempty contours, the limits need to be
        # expanded. If negative values are allowed, expand the limits so that
        # the contour plot is filled with the color located in the middle of the
        # color bar; otherwise, expand them so that it is filled with the color
        # at the bottom of the color bar.
        limits = limits[1] .+ (negative_values_allowed ? (-1, 1) : (0, 1))
    else
        colorbar_ticks_kwarg = (;)
    end
    n_bands = 20
    levels = range(limits..., n_bands + 1)
    kwargs = (; levels, extendlow, colormap = :viridis)

    grid_layout = Makie.GridLayout(; default_rowgap = 10, default_colgap = 10)
    grid_position[:, :] = grid_layout

    axis = Makie.Axis(
        grid_layout[1, :];
        title = "$category $name",
        xticks = TimeTicks(),
        xlabel = label_ts && !draw_ref_plot ? "Time" : "",
        ylabel = label_zs ?
                 (draw_ref_plot ? "PR\n" : "") * "Elevation (km)" : "",
        xticksvisible = !draw_ref_plot,
        xticklabelsvisible = label_ts && !draw_ref_plot,
        yticklabelsvisible = label_zs,
    )
    first_plot = Makie.contourf!(axis, ts, zs, values; kwargs...)
    for _ in 1:n_redraws
        Makie.contourf!(axis, ts, zs, values; kwargs...)
    end

    if draw_ref_plot
        ref_axis = Makie.Axis(
            grid_layout[2, :];
            xticks = TimeTicks(),
            xlabel = label_ts ? "Time" : "",
            ylabel = label_zs ? "main\nElevation (km)" : "",
            xticklabelsvisible = label_ts,
            yticklabelsvisible = label_zs,
        )
        if isnothing(ref_var_col_time_series)
            Makie.contourf!(ref_axis, ts, zs, fill(NaN, size(values)))
        else
            for _ in 1:(n_redraws + 1)
                Makie.contourf!(ref_axis, ref_ts, ref_zs, ref_values; kwargs...)
            end
        end
    end

    for _ in 1:(n_redraws + 1)
        Makie.Colorbar(grid_layout[:, 2], first_plot; colorbar_ticks_kwarg...)
    end
end

function profile_plot!(
    grid_position,
    final_var_cols,
    ref_final_var_cols,
    categories,
    name,
    label_zs,
)
    # Use maximally distinguishable colors that are neither too bright nor too
    # dark (restrict their luminance range from [0, 100] to [30, 40]).
    colors = Makie.ColorSchemes.distinguishable_colors(6; lchoices = 30:40)
    color_indices = Dict(:gm => 1, :draft => 2, :env => 3)
    ref_color_indices = Dict(:gm => 4, :draft => 5, :env => 6)
    alpha = 0.6 # Use transparency to see when lines are on top of each other.

    axis = Makie.Axis(
        grid_position;
        xlabel = "$name",
        ylabel = label_zs ? "Elevation (km)" : "",
        yticksvisible = label_zs,
        yticklabelsvisible = label_zs,
    )
    any_refs_available = any(!isnothing, getindex.(ref_final_var_cols, 2))
    for (category, (zs, values)) in zip(categories, final_var_cols)
        isnothing(values) && continue
        color = (colors[color_indices[category]], alpha)
        label = (any_refs_available ? "PR " : "") * "$category"
        Makie.lines!(axis, values, zs; color, label)
    end
    for (category, (zs, values)) in zip(categories, ref_final_var_cols)
        isnothing(values) && continue
        color = (colors[ref_color_indices[category]], alpha)
        label = "main $category"
        Makie.lines!(axis, values, zs; color, label, linestyle = :dash)
    end
end

function contours_and_profiles(output_dir, ref_job_id = nothing)
    ##
    ## Reading in time series data:
    ##

    function sorted_hdf5_files(dir)
        file_paths = readdir(dir; join = true)
        filter!(endswith(".hdf5"), file_paths)
        sort!(file_paths; by = time_from_filename)
        return file_paths
    end

    function read_hdf5_file(file_path)
        reader = InputOutput.HDF5Reader(file_path)
        diagnostics = InputOutput.read_field(reader, "diagnostics")
        close(reader)
        return time_from_filename(file_path), diagnostics
    end

    time_series = map(read_hdf5_file, sorted_hdf5_files(output_dir))
    t_end = time_series[end][1]

    ref_time_series = nothing
    if !isnothing(ref_job_id) && haskey(ENV, "BUILDKITE_COMMIT")
        main_dir = "/central/scratch/esm/slurm-buildkite/climaatmos-main"
        isdir(main_dir) ||
            error("Unable to find $main_dir when running on Buildkite")
        commit_dirs = readdir(main_dir; join = true)

        # Only commits that increment the "ref_counter" currently generate hdf5
        # files. The files are zipped for TC cases, but not for other CI jobs.
        # TODO: Unify how reference data is stored and simplify this code.
        has_hdf5_files(commit_dir) =
            isdir(joinpath(commit_dir, ref_job_id)) &&
            any(endswith(".hdf5"), readdir(joinpath(commit_dir, ref_job_id)))
        zip_file(commit_dir) = joinpath(commit_dir, ref_job_id, "hdf5files.zip")
        filter!(commit_dirs) do commit_dir
            return has_hdf5_files(commit_dir) || isfile(zip_file(commit_dir))
        end
        if isempty(commit_dirs)
            @warn "Unable to find reference data for $ref_job_id in $main_dir"
        else
            latest_commit_dir =
                argmax(commit_dir -> stat(commit_dir).mtime, commit_dirs)
            commit_id = basename(latest_commit_dir)
            @info "Loading reference data from main branch (commit $commit_id)"
            if has_hdf5_files(latest_commit_dir)
                ref_files_dir = joinpath(latest_commit_dir, ref_job_id)
            else
                ref_files_dir = mktempdir(output_dir; prefix = "unzip_dir_")
                run(`unzip $(zip_file(latest_commit_dir)) -d $ref_files_dir`)
            end
            ref_file_paths = sorted_hdf5_files(ref_files_dir)
            _, last_index = findmin(ref_file_paths) do ref_file_path
                return abs(t_end - time_from_filename(ref_file_path))
            end # End the reference time series as close to t_end as possible.
            ref_time_series = map(read_hdf5_file, ref_file_paths[1:last_index])
            ref_t_end = ref_time_series[end][1]
        end
    end

    ##
    ## Extracting a column of data from each time series:
    ##

    # Assume that all variables are on the same horizontal space.
    horizontal_space(time_series) =
        Spaces.horizontal_space(axes(time_series[1][2].temperature))

    is_on_sphere(time_series) =
        eltype(Fields.coordinate_field(horizontal_space(time_series))) <:
        Geometry.LatLongPoint

    function column_view_time_series(time_series, column)
        ((i, j), h) = column
        is_extruded_field(object) =
            axes(object) isa Spaces.ExtrudedFiniteDifferenceSpace
        column_view(field) = Fields.column(field, i, j, h)
        column_zs(object) =
            is_extruded_field(object) ?
            vec(parent(Fields.coordinate_field(column_view(object)).z)) / 1000 :
            nothing
        column_values(object) =
            is_extruded_field(object) ? vec(parent(column_view(object))) :
            nothing
        col_time_series = map(time_series) do (t, diagnostics)
            objects = Fields._values(diagnostics)
            return t, map(column_zs, objects), map(column_values, objects)
        end

        # Assume that all variables have the same horizontal coordinates.
        coords =
            Fields.coordinate_field(column_view(time_series[1][2].temperature))

        coord_strings = map(filter(!=(:z), propertynames(coords))) do symbol
            # Add 0 to every horizontal coordinate value so that -0.0 gets
            # printed without the unnecessary negative sign.
            value = round(mean(getproperty(coords, symbol)); sigdigits = 6) + 0
            return "$symbol = $value"
        end
        col_string = "column data from $(join(coord_strings, ", "))"

        return col_time_series, col_string
    end

    get_column_1(time_series) =
        isnothing(time_series) ? (nothing, nothing) :
        column_view_time_series(time_series, ((1, 1), 1))

    column_at_latitude_getter(latitude) =
        time_series -> begin
            isnothing(time_series) && return (nothing, nothing)
            column = if is_on_sphere(time_series)
                horz_space = horizontal_space(time_series)
                horz_coords = Fields.coordinate_field(horz_space)
                FT = eltype(eltype(horz_coords))
                target_column_coord =
                    Geometry.LatLongPoint(FT(latitude), FT(0))
                distance_to_target(((i, j), h)) =
                    Geometry.great_circle_distance(
                        Spaces.column(horz_coords, i, j, h)[],
                        target_column_coord,
                        horz_space.global_geometry,
                    )
                argmin(distance_to_target, Spaces.all_nodes(horz_space))
            else
                ((1, 1), 1) # If the data is not on a sphere, extract column 1.
            end
            return column_view_time_series(time_series, column)
        end

    # TODO: Add averaging over multiple columns.
    column_info = if is_on_sphere(time_series)
        map((0, 30, 60, 90)) do latitude
            return column_at_latitude_getter(latitude), "_from_$(latitude)N_0E"
        end
    else
        ((get_column_1, ""),)
    end

    ##
    ## Extracting data for a specific variable from each column time series:
    ##

    function variable_column_time_series(col_time_series, category, name)
        isnothing(col_time_series) && return nothing
        symbol = category === :gm ? name : Symbol(category, :_, name)

        # TODO: Remove this workaround when the old EDMF model is deleted.
        deprecated_symbols = Dict(
            :env_temperature => :env_temperature,
            :draft_temperature => :bulk_up_temperature,
            :env_potential_temperature => :env_theta_liq_ice,
            :env_w_velocity => :face_env_w,
            :draft_w_velocity => :face_bulk_w,
            :draft_buoyancy => :bulk_up_buoyancy,
            :draft_q_liq => :bulk_up_q_liq,
            :draft_q_ice => :bulk_up_q_ice,
            :env_relative_humidity => :env_RH,
            :draft_cloud_fraction => :bulk_up_cloud_fraction,
            :draft_area => :bulk_up_area,
            :env_tke => :env_TKE,
        )
        if !hasproperty(col_time_series[1][2], symbol) &&
           symbol in keys(deprecated_symbols)
            symbol = deprecated_symbols[symbol]
        end

        hasproperty(col_time_series[1][2], symbol) || return nothing
        return map(col_time_series) do (t, zs, diagnostics)
            return t, getproperty(zs, symbol), getproperty(diagnostics, symbol)
        end
    end

    final_variable_columns(col_time_series, categories, name) =
        map(categories) do category
            var_col_time_series =
                variable_column_time_series(col_time_series, category, name)
            return isnothing(var_col_time_series) ? (nothing, nothing) :
                   var_col_time_series[end][2:3]
        end

    has_moisture = hasproperty(time_series[1][2], :q_vap)
    has_precipitation = hasproperty(time_series[1][2], :q_rai)
    has_sgs = hasproperty(time_series[1][2], :draft_area)

    negative_values_allowed_names =
        (:u_velocity, :v_velocity, :w_velocity, :buoyancy)

    env_or_gm = has_sgs ? :env : :gm
    draft_or_gm = has_sgs ? :draft : :gm
    contour_variables = (
        (draft_or_gm, :temperature),
        (:gm, :potential_temperature),
        (:gm, :u_velocity),
        (:gm, :v_velocity),
        (:gm, :w_velocity),
        (draft_or_gm, :buoyancy),
    )
    if has_moisture
        contour_variables = (
            contour_variables...,
            (env_or_gm, :relative_humidity),
            (draft_or_gm, :q_vap),
            (draft_or_gm, :q_liq),
            (draft_or_gm, :q_ice),
        )
    end
    if has_precipitation
        contour_variables =
            (contour_variables..., (draft_or_gm, :q_rai), (draft_or_gm, :q_sno))
    end
    if has_sgs
        if has_moisture
            contour_variables = (contour_variables..., (:gm, :cloud_fraction))
        end
        contour_variables =
            (contour_variables..., (:draft, :area), (:env, :tke))
    end

    all_categories = has_sgs ? (:gm, :env, :draft) : (:gm,)
    profile_variables = (
        (all_categories, :temperature),
        (all_categories, :potential_temperature),
        ((:gm,), :u_velocity),
        ((:gm,), :v_velocity),
        (all_categories, :w_velocity),
        (all_categories, :buoyancy),
    )
    if has_moisture
        profile_variables = (
            profile_variables...,
            (all_categories, :relative_humidity),
            (all_categories, :q_vap),
            (all_categories, :q_liq),
            (all_categories, :q_ice),
        )
    end
    if has_precipitation
        profile_variables = (
            profile_variables...,
            (all_categories, :q_rai),
            (all_categories, :q_sno),
        )
    end
    if has_sgs
        if has_moisture
            profile_variables =
                (profile_variables..., ((:gm,), :cloud_fraction))
        end
        profile_variables =
            (profile_variables..., ((:draft,), :area), ((:env,), :tke))
    end

    ##
    ## Generating the contour and profile plots:
    ##

    # Organize the variables into squares, or something close to squares.
    # If they are not squares, the contour plots should have more rows than
    # columns, whereas the profile plots should have more columns than rows.
    sqrt_factors(n) = minmax(round(Int, sqrt(n)), cld(n, round(Int, sqrt(n))))
    n_contour_cols, n_contour_rows = sqrt_factors(length(contour_variables))
    n_profile_rows, n_profile_cols = sqrt_factors(length(profile_variables))

    # Plot the variables by starting at the bottom of the first column and
    # moving upward, then going to the bottom of the second column and moving
    # upward, and so on. This ensures that the first column and the bottom row
    # are always filled, which means that axis ticks do not need to be drawn in
    # the remaining rows and columns.
    function row_and_col(index, n_rows)
        col, row = divrem(index - 1, n_rows, RoundDown) .+ 1
        return n_rows + 1 - row, col
    end

    contour_resolution = (700 * n_contour_cols, 400 * n_contour_rows)
    profile_resolution = (400 * n_profile_cols, 400 * n_profile_rows)

    for (get_column, filename_suffix) in column_info
        col_time_series, col_string = get_column(time_series)
        ref_col_time_series, ref_col_string = get_column(ref_time_series)

        contour_title =
            isnothing(ref_col_time_series) || ref_col_string == col_string ?
            col_string : "PR $col_string\nmain branch $ref_col_string"

        profile_title =
            isnothing(ref_col_time_series) ||
            (ref_col_string == col_string && ref_t_end == t_end) ?
            "$col_string after $(time_string_with_unit(t_end))" :
            "PR $col_string after $(time_string_with_unit(t_end))\nmain branch \
             $ref_col_string after $(time_string_with_unit(ref_t_end))"

        # Assume that all variables have the same vertical coordinate limits.
        z_limits = extrema(col_time_series[1][2].temperature)

        figure = Makie.Figure()
        for (index, (category, name)) in enumerate(contour_variables)
            row, col = row_and_col(index, n_contour_rows)
            contour_plot!(
                figure[row, col],
                variable_column_time_series(col_time_series, category, name),
                variable_column_time_series(
                    ref_col_time_series,
                    category,
                    name,
                ),
                category,
                name,
                !isnothing(ref_col_time_series),
                row == n_contour_rows,
                col == 1,
                name in negative_values_allowed_names,
            )
        end
        all_axes = filter(object -> object isa Makie.Axis, figure.content)
        foreach(axis -> Makie.limits!(axis, (0, t_end), z_limits), all_axes)
        Makie.Label(figure[0, :], contour_title; font = :bold, fontsize = 20)
        file_path = joinpath(output_dir, "contours$(filename_suffix).png")
        Makie.save(file_path, figure; resolution = contour_resolution)

        figure = Makie.Figure()
        for (index, (categories, name)) in enumerate(profile_variables)
            row, col = row_and_col(index, n_profile_rows)
            profile_plot!(
                figure[row, col],
                final_variable_columns(col_time_series, categories, name),
                final_variable_columns(ref_col_time_series, categories, name),
                categories,
                name,
                col == 1,
            )
        end
        all_axes = filter(object -> object isa Makie.Axis, figure.content)
        foreach(axis -> Makie.ylims!(axis, z_limits), all_axes)
        Makie.Legend(figure[0, :], all_axes[1]; orientation = :horizontal)
        Makie.Label(figure[-1, :], profile_title; font = :bold, fontsize = 20)
        file_path = joinpath(output_dir, "final_profiles$(filename_suffix).png")
        Makie.save(file_path, figure; resolution = profile_resolution)
    end
end
