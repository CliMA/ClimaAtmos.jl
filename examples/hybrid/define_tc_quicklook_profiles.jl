import ClimaCore: Fields, InputOutput
using Plots

function plot_tc_profiles(folder; hdf5_filename, main_branch_data_path)

    PR_filename = joinpath(folder, hdf5_filename)
    main_filename = joinpath(main_branch_data_path, hdf5_filename)

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
        plot!(p5a, parent(D.face_bulk_w)[:], zf; label = "$data_source")
        plot!(p5b, parent(D.face_env_w)[:], zf; label = "$data_source")
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

    add_to_plots!(PR_filename; data_source = "PR")
    add_to_plots!(main_filename; data_source = "main")

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
