import ClimaCore: Fields, InputOutput, Geometry
using Plots
import ClimaAtmos as CA

function plot_tc_profiles(folder; hdf5_filename, main_branch_data_path)
    args =
        (; tickfontsize = 13, guidefontsize = 16, legendfontsize = 10, lw = 3)

    # initialize all plot panes
    p1 = plot(; title = "area fraction", args...)
    p2 = plot(; title = "up qt", args...)
    p3 = plot(; title = "up ql", args...)
    p4 = plot(; title = "up qi", args...)
    p5a = plot(; title = "up w", args...)
    p5b = plot(; title = "en w", args...)
    p6 = plot(; title = "en qt", args...)
    p7 = plot(; title = "en ql", args...)
    p8 = plot(; title = "en qi", args...)
    p9 = plot(; title = "buoy", args...)
    p10 = plot(; title = "T", args...)
    p11 = plot(; title = "CF", args...)
    p12 = plot(; title = "en RH", args...)
    p13 = plot(; title = "en TKE", args...)
    p14 = plot(; title = "en Hvar", args...)
    p15 = plot(; title = "en QTvar", args...)
    p16 = plot(; title = "en HQTcov", args...)

    function add_to_plots!(input_filename; data_source)
        if !isfile(input_filename)
            @info "Data file `$input_filename` not found for data source `$data_source`."
            return
        end

        reader = InputOutput.HDF5Reader(input_filename)
        Y = InputOutput.read_field(reader, "Y")
        D = InputOutput.read_field(reader, "diagnostics")

        zc = parent(Fields.coordinate_field(Y.c).z)[:]
        zf = parent(Fields.coordinate_field(Y.f).z)[:]

        plot!(p1, parent(D.bulk_up_area)[:], zc; label = "up $data_source")
        plot!(p1, parent(D.env_area)[:], zc; label = "en $data_source")
        plot!(p2, parent(D.bulk_up_q_tot)[:], zc; label = "$data_source")
        plot!(p3, parent(D.bulk_up_q_liq)[:], zc; label = "$data_source")
        plot!(p4, parent(D.bulk_up_q_ice)[:], zc; label = "$data_source")
        plot!(
            p5a,
            parent(Geometry.WVector.(D.face_bulk_w))[:],
            zf;
            label = "$data_source",
        )
        plot!(
            p5b,
            parent(Geometry.WVector.(D.face_env_w))[:],
            zf;
            label = "$data_source",
        )
        plot!(p6, parent(D.env_q_tot)[:], zc; label = "$data_source")
        plot!(p7, parent(D.env_q_liq)[:], zc; label = "$data_source")
        plot!(p8, parent(D.env_q_ice)[:], zc; label = "$data_source")
        plot!(p9, parent(D.bulk_up_buoyancy)[:], zc; label = "up $data_source")
        plot!(p9, parent(D.env_buoyancy)[:], zc; label = "en $data_source")
        plot!(
            p10,
            parent(D.bulk_up_temperature)[:],
            zc;
            label = "up $data_source",
        )
        plot!(p10, parent(D.env_temperature)[:], zc; label = "en $data_source")
        plot!(
            p11,
            parent(D.bulk_up_cloud_fraction)[:],
            zc;
            label = "up $data_source",
        )
        plot!(
            p11,
            parent(D.env_cloud_fraction)[:],
            zc;
            label = "env $data_source",
        )
        plot!(p12, parent(D.env_RH)[:], zc; label = "$data_source")
        plot!(p13, parent(D.env_TKE)[:], zc; label = "$data_source")
        plot!(p14, parent(D.env_Hvar)[:], zc; label = "$data_source")
        plot!(p15, parent(D.env_QTvar)[:], zc; label = "$data_source")
        plot!(p16, parent(D.env_HQTcov)[:], zc; label = "$data_source")
    end

    PR_filename = joinpath(folder, hdf5_filename)
    add_to_plots!(PR_filename; data_source = "PR")
    if ispath(main_branch_data_path)
        main_filename = joinpath(main_branch_data_path, hdf5_filename)
        add_to_plots!(main_filename; data_source = "main")
    end

    more_args = (;
        size = (2400.0, 1500.0),
        bottom_margin = 20.0 * Plots.PlotMeasures.px,
        left_margin = 20.0 * Plots.PlotMeasures.px,
        layout = (4, 5),
    )
    p = plot(
        p1,
        p2,
        p3,
        p4,
        p5a,
        p5b,
        p6,
        p7,
        p8,
        p9,
        p10,
        p11,
        p12,
        p13,
        p14,
        p15,
        p16;
        more_args...,
    )

    # Save output
    output_filename = joinpath(
        folder,
        "____________________________________final_profiles.png",
    )
    png(p, output_filename)
end

function get_contours(input_filenames, plots; data_source, have_main)
    files_found = map(x -> isfile(x), input_filenames)
    if !all(files_found)
        @warn "Some data files were missing in data source `$data_source`."
        @warn "$(count(files_found)) files found out of $(length(input_filenames))"
        @warn "Filtering out only found files."
        input_filenames = filter(isfile, input_filenames)
    end

    data = map(input_filenames) do input_filename
        reader = InputOutput.HDF5Reader(input_filename)
        Y = InputOutput.read_field(reader, "Y")
        D = InputOutput.read_field(reader, "diagnostics")
        (Y, D)
    end
    (Ys, Ds) = first.(data), last.(data)
    t = CA.time_from_filename.(input_filenames)

    zc = parent(Fields.coordinate_field(first(Ys).c).z)[:]
    zf = parent(Fields.coordinate_field(first(Ys).f).z)[:]

    K = collect(keys(plots))
    n = length(K)
    width_to_height_ratio = have_main ? 22 / 18 : 22 / 18
    fig_height = 1800
    left_side = data_source == "main"
    l_margin = left_side ? 40 : -20
    r_margin = left_side ? -20 : -20

    contours = map(enumerate(K)) do (i, name)
        fn = plots[name].fn
        cdata = hcat(fn.(Ds)...)
        Plots.contourf!(
            plots[name].plot,
            t ./ 3600,
            zc ./ 10^3,
            cdata;
            xticks = i == n,
            yticks = left_side,
            colorbar = !left_side,
            c = :viridis,
            xlabel = i == n ? "Time (hours)" : "",
            ylabel = left_side ? "Height (km)" : "",
            left_margin = l_margin * Plots.PlotMeasures.px,
            right_margin = r_margin * Plots.PlotMeasures.px,
            bottom_margin = 0 * Plots.PlotMeasures.px,
            top_margin = 0 * Plots.PlotMeasures.px,
            size = (width_to_height_ratio * fig_height, fig_height), # extra space for colorbar
            title = "$name ($data_source)",
        )
    end
    return contours
end

function hdf5_files(path, name_match)
    files = filter(x -> endswith(x, ".hdf5"), readdir(path, join = true))
    filter!(x -> occursin(name_match, x), files)
end

function plot_tc_contours(folder; main_branch_data_path, name_match)
    PR_filenames = CA.sort_files_by_time(hdf5_files(folder, name_match))
    main_filenames = if ispath(main_branch_data_path)
        files = hdf5_files(main_branch_data_path, name_match)
        if any(isfile.(files))
            CA.sort_files_by_time(files)
        else
            nothing
        end
    else
        nothing
    end
    _plot_tc_contours(folder; PR_filenames, main_filenames)
end

function get_plots(vars)
    plots = map(vars) do name_fn
        name = first(name_fn)
        fn = last(name_fn)
        cplot = Plots.contourf()
        Pair(name, (; plot = cplot, fn))
    end
    plots = Dict(plots...)
    return plots
end

function _plot_tc_contours(folder; PR_filenames, main_filenames)

    vars = [
        ("area fraction", D -> parent(D.bulk_up_area)[:]),
        ("up qt", D -> parent(D.bulk_up_q_tot)[:]),
        ("up ql", D -> parent(D.bulk_up_q_liq)[:]),
        ("up qi", D -> parent(D.bulk_up_q_ice)[:]),
        ("up w", D -> parent(Geometry.WVector.(D.face_bulk_w))[:]),
        ("en qt", D -> parent(D.env_q_tot)[:]),
        ("en TKE", D -> parent(D.env_TKE)[:]),
        # ("up qr", D-> parent(D.)[:])
    ]

    have_main = main_filenames ≠ nothing
    contours_PR = get_contours(
        PR_filenames,
        get_plots(vars);
        data_source = "PR",
        have_main,
    )
    if have_main
        contours_main = get_contours(
            main_filenames,
            get_plots(vars);
            data_source = "main",
            have_main,
        )
        # TODO: get and reset clims

        P = map(collect(zip(contours_main, contours_PR))) do z
            Plots.plot(z...; layout = Plots.grid(1, 2; widths = [0.45, 0.55]))
        end
        p = Plots.plot(P...; layout = Plots.grid(2 * length(contours_main), 1))
    else
        @warn "No main branch to compare against"
        p = Plots.plot(
            contours_PR...;
            layout = Plots.grid(length(contours_PR), 1),
        )
    end

    # Save output
    output_filename = joinpath(
        folder,
        "____________________________________final_contours.png",
    )
    png(p, output_filename)
end
