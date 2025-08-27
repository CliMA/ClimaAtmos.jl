script_folder = joinpath(pkgdir(CA), "examples", "perturbed_dycoms_rf02")
include(joinpath(script_folder, "all_paths.jl"))
include(joinpath(pkgdir(CA), "post_processing", "ci_plots.jl"))

DiagEDMF1M = [
    Val{Symbol(replace(f, ".yml" => ""))} for f in readdir(diagnostic_1M_config)
]

DiagEDMF2M = [
    Val{Symbol(replace(f, ".yml" => ""))} for f in readdir(diagnostic_2M_config)
]

ProgEDMF1M = [
    Val{Symbol(replace(f, ".yml" => ""))} for f in readdir(prognostic_1M_config)
]

ProgEDMF2M = [
    Val{Symbol(replace(f, ".yml" => ""))} for f in readdir(prognostic_2M_config)
]

function plot_edmf_vert_profile_and_time!(grid_loc, var_group)

    if all(short_name(var) in ("lwp", "rwp") for var in var_group)
        # Time series.
        t = ClimaAnalysis.times(var_group[1])
        units = ClimaAnalysis.units(var_group[1])
        t_units = ClimaAnalysis.dim_units(var_group[1], "time")
        ax_time = CairoMakie.Axis(
            grid_loc[1, 1],
            xlabel = "t [$t_units]",
            ylabel = "$(short_name(var_group[1])) [$units]",
            title = parse_var_attributes(var_group[1]),
        )

        for var in var_group
            CairoMakie.lines!(ax_time, t, var.data, label = short_name(var))
        end

        if length(var_group) > 1
            Makie.axislegend(ax_time)
        end
    else
        # Vertical profiles.
        z = ClimaAnalysis.altitudes(var_group[1])
        units = ClimaAnalysis.units(var_group[1])
        z_units = ClimaAnalysis.dim_units(var_group[1], "z")
        ax_vert = CairoMakie.Axis(
            grid_loc[1, 1],
            ylabel = "z [$z_units]",
            xlabel = "$(short_name(var_group[1])) [$units]",
            title = parse_var_attributes(var_group[1]),
        )

        for var in var_group
            CairoMakie.lines!(ax_vert, var.data, z, label = short_name(var))
        end

        if length(var_group) > 1
            Makie.axislegend(ax_vert)
        end
    end
end

function make_plots(
    sim_type::Union{DiagEDMF1M..., ProgEDMF1M...},
    output_paths::Vector{<:AbstractString},
)
    simdirs = SimDir.(output_paths)

    if sim_type in ProgEDMF1M
        precip_names =
            ("husra", "hussn", "husraup", "hussnup", "husraen", "hussnen")
    else
        precip_names = ("husra", "hussn", "husraup", "hussnup")
    end

    short_names = [
        "wa",
        "waup",
        "ta",
        "taup",
        "hus",
        "husup",
        "arup",
        "tke",
        "ua",
        "thetaa",
        "thetaaup",
        "ha",
        "haup",
        "hur",
        "hurup",
        "lmix",
        "cl",
        "clw",
        "clwup",
        "cli",
        "cliup",
        precip_names...,
    ]

    time_var = ["lwp", "rwp"]

    reduction = "inst"

    available_periods = ClimaAnalysis.available_periods(
        simdirs[1];
        short_name = short_names[1],
        reduction,
    )
    if "5m" in available_periods
        period = "5m"
    elseif "10m" in available_periods
        period = "10m"
    elseif "30m" in available_periods
        period = "30m"
    end

    short_name_tuples = pair_edmf_names(short_names)
    var_groups_zt =
        map_comparison(simdirs, short_name_tuples) do simdir, name_tuple
            return [
                slice(
                    get(simdir; short_name, reduction, period),
                    x = 0.0,
                    y = 0.0,
                ) for short_name in name_tuple
            ]
        end

    var_groups_z = [
        ([slice(v, time = LAST_SNAP) for v in group]...,) for
        group in var_groups_zt
    ]

    var_groups_t =
        map_comparison(simdirs, pair_edmf_names(time_var)) do simdir, name_tuple
            return [
                slice(
                    get(simdir; short_name, reduction, period),
                    x = 0.0,
                    y = 0.0,
                ) for short_name in name_tuple
            ]
        end

    tmp_file = make_plots_generic(
        output_paths,
        output_name = "tmp",
        vcat(var_groups_t, var_groups_z);
        plot_fn = plot_edmf_vert_profile_and_time!,
        MAX_NUM_COLS = 2,
        MAX_NUM_ROWS = 4,
    )

    make_plots_generic(
        output_paths,
        vcat(var_groups_zt...),
        plot_fn = plot_parsed_attribute_title!,
        summary_files = [tmp_file],
        MAX_NUM_COLS = 2,
        MAX_NUM_ROWS = 4,
    )
end

function make_plots(
    sim_type::Union{DiagEDMF2M..., ProgEDMF2M...},
    output_paths::Vector{<:AbstractString},
)
    simdirs = SimDir.(output_paths)

    if sim_type in ProgEDMF2M
        precip_names =
            ("husra", "hussn", "husraup", "hussnup", "husraen", "hussnen")
        cloud_droplets = ("cdnc", "cdncup", "cdncen")
    else
        precip_names = ("husra", "hussn", "husraup", "hussnup")
        cloud_droplets = ("cdnc", "cdncup")
    end

    short_names = [
        "wa",
        "waup",
        "ta",
        "taup",
        "hus",
        "husup",
        "arup",
        "tke",
        "ua",
        "thetaa",
        "thetaaup",
        "ha",
        "haup",
        "hur",
        "hurup",
        "lmix",
        "cl",
        "clw",
        "clwup",
        "cli",
        "cliup",
        precip_names...,
        cloud_droplets...,
    ]

    time_var = ["lwp", "rwp"]

    reduction = "inst"

    available_periods = ClimaAnalysis.available_periods(
        simdirs[1];
        short_name = short_names[1],
        reduction,
    )
    if "5m" in available_periods
        period = "5m"
    elseif "10m" in available_periods
        period = "10m"
    elseif "30m" in available_periods
        period = "30m"
    end

    short_name_tuples = pair_edmf_names(short_names)
    var_groups_zt =
        map_comparison(simdirs, short_name_tuples) do simdir, name_tuple
            return [
                slice(
                    get(simdir; short_name, reduction, period),
                    x = 0.0,
                    y = 0.0,
                ) for short_name in name_tuple
            ]
        end

    var_groups_z = [
        ([slice(v, time = LAST_SNAP) for v in group]...,) for
        group in var_groups_zt
    ]

    var_groups_t =
        map_comparison(simdirs, pair_edmf_names(time_var)) do simdir, name_tuple
            return [
                slice(
                    get(simdir; short_name, reduction, period),
                    x = 0.0,
                    y = 0.0,
                ) for short_name in name_tuple
            ]
        end

    tmp_file = make_plots_generic(
        output_paths,
        output_name = "tmp",
        vcat(var_groups_t, var_groups_z);
        plot_fn = plot_edmf_vert_profile_and_time!,
        MAX_NUM_COLS = 2,
        MAX_NUM_ROWS = 4,
    )

    make_plots_generic(
        output_paths,
        vcat(var_groups_zt...),
        plot_fn = plot_parsed_attribute_title!,
        summary_files = [tmp_file],
        MAX_NUM_COLS = 2,
        MAX_NUM_ROWS = 4,
    )
end
