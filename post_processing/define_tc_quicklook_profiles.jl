import ClimaCore: Fields, InputOutput, Geometry
ENV["GKSwstype"] = "nul";
import Plots
import ClimaAtmos as CA

function getfun(D, sym::Symbol)
    if !hasproperty(D, sym)
        @warn "Property $(sym) not in Diagnostics FieldVector."
        return nothing
    else
        return (Y, D) -> getproperty(D, sym)
    end
end
getfun(D, p::Function) = p

function plot_tc_profiles(folder; hdf5_filename, main_branch_data_path)
    args =
        (; tickfontsize = 13, guidefontsize = 16, legendfontsize = 10, lw = 3)

    # initialize all plot panes

#! format: off
    vars = Dict()
    # vars[title] = (; fn = func to get variable(s), pfx = label prefix(es))
    vars["area fraction"] =     (;fn=(:bulk_up_area, :env_area), pfx = ("up", "en"))
    vars["buoy"] =              (;fn=(:bulk_up_buoyancy, :env_buoyancy), pfx = ("up", "en"))
    vars["T"] =                 (;fn=(:bulk_up_temperature, :env_temperature), pfx = ("up", "en"))
    vars["CF"] =                (;fn=(:bulk_up_cloud_fraction, :env_cloud_fraction), pfx = ("up", "env"))

    vars["up w"]  =             (;fn=(Y, D) -> Geometry.WVector.(D.face_bulk_w), pfx="")
    vars["en w"] =              (;fn=(Y, D) -> Geometry.WVector.(D.face_env_w), pfx="")
    vars["gm u"] =              (;fn=(Y, D) -> Geometry.UVector.(Y.c.uₕ), pfx = "")
    vars["gm v"] =              (;fn=(Y, D) -> Geometry.VVector.(Y.c.uₕ), pfx = "")

    vars["up qt"] =             (;fn=:bulk_up_q_tot, pfx = "")
    vars["up ql"] =             (;fn=:bulk_up_q_liq, pfx = "")
    vars["up qi"] =             (;fn=:bulk_up_q_ice, pfx = "")
    vars["en qt"] =             (;fn=:env_q_tot, pfx = "")
    vars["en ql"] =             (;fn=:env_q_liq, pfx = "")
    vars["en qi"] =             (;fn=:env_q_ice, pfx = "")
    vars["gm theta"] =          (;fn=:potential_temperature, pfx = "")
    vars["en RH"] =             (;fn=:env_RH, pfx = "")
    vars["en TKE"] =            (;fn=:env_TKE, pfx = "")
    vars["en Hvar"] =           (;fn=:env_Hvar, pfx = "")
    vars["en QTvar"] =          (;fn=:env_QTvar, pfx = "")
    vars["en HQTcov"] =         (;fn=:env_HQTcov, pfx = "")
    vars["up filter flag 1"] =  (;fn=:bulk_up_filter_flag_1, pfx = "")
    vars["up filter flag 2"] =  (;fn=:bulk_up_filter_flag_2, pfx = "")
    vars["up filter flag 3"] =  (;fn=:bulk_up_filter_flag_3, pfx = "")
    vars["up filter flag 4"] =  (;fn=:bulk_up_filter_flag_4, pfx = "")
#! format: on

    p_all = map(k -> Plots.plot(; title = k, args...), collect(keys(vars)))

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

        for (i, (k, v)) in enumerate(vars)
            if v.pfx isa Tuple
                for j in 1:length(v.pfx)
                    fn = getfun(D, v.fn[j])
                    isnothing(fn) && continue
                    var = fn(Y, D)
                    z = parent(Fields.coordinate_field(axes(var)).z)[:]
                    vardata = parent(var)[:]
                    prefix = isempty(v.pfx[j]) ? "" : "$(v.pfx[j]) "
                    plot!(p_all[i], vardata, z; label = "$prefix$data_source")
                end
            elseif v.pfx isa String
                fn = getfun(D, v.fn)
                isnothing(fn) && continue
                var = fn(Y, D)
                z = parent(Fields.coordinate_field(axes(var)).z)[:]
                vardata = parent(var)[:]
                prefix = isempty(v.pfx) ? "" : "$(v.pfx) "
                plot!(p_all[i], vardata, z; label = "$prefix$data_source")
            end
        end
    end

    PR_filename = joinpath(folder, hdf5_filename)
    add_to_plots!(PR_filename; data_source = "PR")
    if ispath(main_branch_data_path)
        main_filename = joinpath(main_branch_data_path, hdf5_filename)
        add_to_plots!(main_filename; data_source = "main")
    end

    n_cols = 5
    more_args = (;
        size = (2400.0, 1500.0),
        bottom_margin = 20.0 * Plots.PlotMeasures.px,
        left_margin = 20.0 * Plots.PlotMeasures.px,
        layout = (ceil(Int, length(p_all) / n_cols), n_cols),
    )
    p = Plots.plot(p_all...; more_args...)

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
    fig_height = 5000
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
        fn = getfun(first(Ds), contours[name].fn)
        isnothing(fn) && continue
        space = axes(fn(first(Ys), first(Ds)))
        z = parent(Fields.coordinate_field(space).z)[:]
        Ds_parent = parent_data.(fn.(Ys, Ds))
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

is_trivial_clims(tup::Tuple{T, T}) where {T} = tup[1] == tup[2]

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
        ("area fraction", :bulk_up_area),
        ("up qt", :bulk_up_q_tot),
        ("up ql", :bulk_up_q_liq),
        ("up qi", :bulk_up_q_ice),
        ("en qt", :env_q_tot),
        ("en TKE", :env_TKE),
        ("gm theta", :potential_temperature),
        ("up filter flag 1", :bulk_up_filter_flag_1),
        ("up filter flag 2", :bulk_up_filter_flag_2),
        ("up filter flag 3", :bulk_up_filter_flag_3),
        ("up filter flag 4", :bulk_up_filter_flag_4),
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
            haskey(clims, k) || continue # need to skip when adding new variables
            Plots.contourf!(contours_main[k].plot; clims[k]...)
        end
        for k in keys(contours_PR)
            haskey(clims, k) || continue # need to skip when adding new variables
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
