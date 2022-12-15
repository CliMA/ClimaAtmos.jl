import ClimaCore: Fields, InputOutput, Geometry
ENV["GKSwstype"] = "nul";
import Plots
import ClimaAtmos as CA

function plot_tc_profiles(folder; hdf5_filename, main_branch_data_path)
    args =
        (; tickfontsize = 13, guidefontsize = 16, legendfontsize = 10, lw = 3)

    # initialize all plot panes
    p1 = Plots.plot(; title = "area fraction", args...)
    p2 = Plots.plot(; title = "up qt", args...)
    p3 = Plots.plot(; title = "up ql", args...)
    p4 = Plots.plot(; title = "up qi", args...)
    p5a = Plots.plot(; title = "up w", args...)
    p5b = Plots.plot(; title = "en w", args...)
    p6 = Plots.plot(; title = "en qt", args...)
    p7 = Plots.plot(; title = "en ql", args...)
    p8 = Plots.plot(; title = "en qi", args...)
    p9 = Plots.plot(; title = "buoy", args...)
    p10 = Plots.plot(; title = "T", args...)
    p11 = Plots.plot(; title = "CF", args...)
    p12 = Plots.plot(; title = "en RH", args...)
    p13 = Plots.plot(; title = "en TKE", args...)
    p14 = Plots.plot(; title = "en Hvar", args...)
    p15 = Plots.plot(; title = "en QTvar", args...)
    p16 = Plots.plot(; title = "en HQTcov", args...)

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
    p = Plots.plot(
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
    Plots.png(p, output_filename)
end

"""
    get_contours(vars, input_filenames; data_source, have_main)

An array of contour plots, given
 - `vars::Tuple{String,Function}` a Tuple of variable name and function
    that obtains the field given the diagnostics
 - `input_filenames::Array{String}` an array of input filenames
 - `data_source::String` a string containing the data source
 - `have_main::Bool` indicates whether the main branch data is available
    in order to adjust the margins
"""
function get_contours(vars, input_filenames; data_source, have_main)
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

    clims = Dict()
    contours = get_empty_contours(vars)
    K = collect(keys(contours))
    n = length(K)
    fig_width = 4500
    fig_height = 3000
    left_side = data_source == "main"
    if have_main
        l_margin = left_side ? 100 : 0
        r_margin = left_side ? 0 : 0 # else case doesn't seem to be working
    else
        l_margin = 100
        r_margin = -100
    end
    parent_data(data) = parent(data)[:]

    for (i, name) in enumerate(K)
        fn = contours[name].fn
        space = axes(fn(first(Ds)))
        z = parent(Fields.coordinate_field(space).z)[:]
        Ds_parent = parent_data.(fn.(Ds))
        cdata = hcat(Ds_parent...)
        clims[name] = (minimum(cdata), maximum(cdata))
        @info "clims[$name] ($data_source) = $(clims[name])"
        Plots.contourf!(
            contours[name].plot,
            t ./ 3600,
            z ./ 10^3,
            cdata;
            xticks = i == n,
            yticks = !have_main || left_side,
            colorbar = !have_main || !left_side,
            c = :viridis,
            xlabel = i == n ? "Time (hours)" : "",
            ylabel = left_side || !have_main ? "Height (km)" : "",
            left_margin = l_margin * Plots.PlotMeasures.px,
            right_margin = r_margin * Plots.PlotMeasures.px,
            bottom_margin = i == n ? 50 * Plots.PlotMeasures.px :
                            0 * Plots.PlotMeasures.px,
            top_margin = 0 * Plots.PlotMeasures.px,
            size = (fig_width, fig_height), # extra space for colorbar
            title = "$name ($data_source)",
        )
    end
    return contours, clims
end

hdf5_files(path) = filter(x -> endswith(x, ".hdf5"), readdir(path, join = true))

function zip_and_cleanup_output(path, zip_file)
    files = basename.(hdf5_files(path))
    cd(path) do
        if !isfile(zip_file)
            zipname = first(splitext(basename(zip_file)))
            run(pipeline(Cmd(["zip", zipname, files...]), stdout = IOBuffer()))
        end
        for f in files
            rm(f)
        end
    end
end

function unzip_file_in_path(path, zip_file, unzip_path)
    if !ispath(path)
        @warn "Path $path not found."
        return nothing
    end
    if !isfile(joinpath(path, zip_file))
        @warn "Zip file does not exist"
        @show path
        @show zip_file
        @show readdir(path, join = true)
        return nothing
    end

    cp(joinpath(path, zip_file), joinpath(unzip_path, zip_file))
    @info "Unzipping files in `$path`"
    cd(unzip_path) do
        zipname = first(splitext(basename(zip_file)))
        run(pipeline(Cmd(["unzip", zipname]); stdout = IOBuffer()))
    end
    files = hdf5_files(unzip_path)
    @assert !isempty(files)
end

function get_main_filenames(main_branch_data_path)
    if ispath(main_branch_data_path)
        files = hdf5_files(main_branch_data_path)
        if any(isfile.(files))
            CA.sort_files_by_time(files)
        else
            nothing
        end
    else
        nothing
    end
end

function debug_filenames(filenames, data_source)
    filenames == nothing && return
    println("---------------- Files for $data_source")
    for f in filenames
        println("    File: $f")
    end
    println("---- Additional info")
    @show allunique(filenames)
    @show allunique(hdf5_files(dirname(first(filenames))))
    println("----------------")
end

function plot_tc_contours(folder; main_branch_data_path)
    PR_filenames = CA.sort_files_by_time(hdf5_files(folder))
    main_filenames = get_main_filenames(main_branch_data_path)
    # debug_filenames(PR_filenames, "PR")
    # debug_filenames(main_filenames, "main")
    _plot_tc_contours(folder; PR_filenames, main_filenames)
end

function get_empty_contours(vars)
    plots = map(vars) do name_fn
        name = first(name_fn)
        fn = last(name_fn)
        cplot = Plots.contourf()
        Pair(name, (; plot = cplot, fn))
    end
    plots = Dict(plots...)
    return plots
end

union_clims(a::Tuple{T, T}, b::Tuple{T, T}) where {T} =
    (min(first(a), first(b)), max(last(a), last(b)))

is_trivial_clims(tup::Tuple{T, T}) where {T} = tup[1] == tup[2] && tup[1] == 0

function union_clims(a::Dict, b::Dict)
    clims_dict = Dict()
    for k in union(keys(a), keys(b))
        clims_tup = if haskey(a, k) && haskey(b, k)
            union_clims(a[k], b[k])
        elseif haskey(a, k)
            a[k]
        else
            b[k]
        end
        # Avoid https://github.com/JuliaPlots/Plots.jl/issues/3924
        clims_dict[k] =
            is_trivial_clims(clims_tup) ? NamedTuple() : (; clims = clims_tup)
        @info "union_clims[$k] = $(clims_dict[k])"
    end
    return clims_dict
end

function _plot_tc_contours(folder; PR_filenames, main_filenames)

    vars = [
        ("area fraction", D -> D.bulk_up_area),
        ("up qt", D -> D.bulk_up_q_tot),
        ("up ql", D -> D.bulk_up_q_liq),
        ("up qi", D -> D.bulk_up_q_ice),
        ("up w", D -> Geometry.WVector.(D.face_bulk_w)),
        ("en qt", D -> D.env_q_tot),
        ("en TKE", D -> D.env_TKE),
        # ("up qr", D-> parent(D.)[:])
    ]

    have_main = main_filenames ≠ nothing
    contours_PR, clims_PR =
        get_contours(vars, PR_filenames; data_source = "PR", have_main)
    if have_main
        contours_main, clims_main =
            get_contours(vars, main_filenames; data_source = "main", have_main)
        clims = union_clims(clims_main, clims_PR)
        for k in keys(contours_main)
            Plots.contourf!(contours_main[k].plot; clims[k]...)
        end
        for k in keys(contours_PR)
            Plots.contourf!(contours_PR[k].plot; clims[k]...)
        end
        # Get array of plots from dict of NamedTuples
        contours_main =
            map(k -> contours_main[k].plot, collect(keys(contours_main)))
        contours_PR = map(k -> contours_PR[k].plot, collect(keys(contours_PR)))
        P = Iterators.flatten(collect(zip(contours_main, contours_PR)))
        p = Plots.plot(
            P...;
            layout = Plots.grid(
                length(contours_main),
                2;
                widths = [0.45, 0.55],
            ),
        )
    else
        @warn "No main branch to compare against"
        # Get array of plots from dict of NamedTuples
        contours_PR = map(k -> contours_PR[k].plot, collect(keys(contours_PR)))
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
    Plots.png(p, output_filename)
end
