import ClimaCore: Fields, InputOutput
using Plots

function plot_tc_profiles(folder, hdf5_filename)

    input_filename = joinpath(folder, hdf5_filename)
    output_filename = joinpath(folder, "final_profiles.png")

    reader = InputOutput.HDF5Reader(input_filename)
    Y = InputOutput.read_field(reader, "Y")
    D = InputOutput.read_field(reader, "diagnostics")

    zc = Fields.coordinate_field(Y.c).z
    zf = Fields.coordinate_field(Y.f).z

    args =
        (; tickfontsize = 13, guidefontsize = 16, legendfontsize = 10, lw = 3)

    p1 = plot(
        parent(D.bulk_up_area)[:],
        parent(zc)[:],
        label = "up area";
        args...,
    )
    p1 = plot!(parent(D.env_area)[:], parent(zc)[:], label = "en area"; args...)
    p2 = plot(
        parent(D.bulk_up_q_tot)[:],
        parent(zc)[:],
        label = "up qt";
        args...,
    )
    p3 = plot(
        parent(D.bulk_up_q_liq)[:],
        parent(zc)[:],
        label = "up ql";
        args...,
    )
    p4 = plot(
        parent(D.bulk_up_q_ice)[:],
        parent(zc)[:],
        label = "up qi";
        args...,
    )
    p5 = plot(parent(D.face_bulk_w)[:], parent(zf)[:], label = "up w"; args...)
    p5 = plot!(parent(D.face_env_w)[:], parent(zf)[:], label = "en w"; args...)
    p6 = plot(parent(D.env_q_tot)[:], parent(zc)[:], label = "en qt"; args...)
    p7 = plot(parent(D.env_q_liq)[:], parent(zc)[:], label = "en ql"; args...)
    p8 = plot(parent(D.env_q_ice)[:], parent(zc)[:], label = "en qi"; args...)
    p9 = plot(
        parent(D.bulk_up_buoyancy)[:],
        parent(zc)[:],
        label = "up buoy";
        args...,
    )
    p9 = plot!(
        parent(D.env_buoyancy)[:],
        parent(zc)[:],
        label = "en buoy";
        args...,
    )
    p10 = plot(
        parent(D.bulk_up_temperature)[:],
        parent(zc)[:],
        label = "up T";
        args...,
    )
    p10 = plot!(
        parent(D.env_temperature)[:],
        parent(zc)[:],
        label = "en T";
        args...,
    )
    p11 = plot(
        parent(D.bulk_up_cloud_fraction)[:],
        parent(zc)[:],
        label = "up CF";
        args...,
    )
    p11 = plot!(
        parent(D.env_cloud_fraction)[:],
        parent(zc)[:],
        label = "env CF";
        args...,
    )
    p12 = plot(parent(D.env_RH)[:], parent(zc)[:], label = "en RH"; args...)
    p13 = plot(parent(D.env_TKE)[:], parent(zc)[:], label = "en TKE"; args...)
    p14 = plot(parent(D.env_Hvar)[:], parent(zc)[:], label = "en Hvar"; args...)
    p15 =
        plot(parent(D.env_QTvar)[:], parent(zc)[:], label = "en QTvar"; args...)
    p16 = plot(
        parent(D.env_HQTcov)[:],
        parent(zc)[:],
        label = "en HQTcov";
        args...,
    )

    p = plot(
        p1,
        p2,
        p3,
        p4,
        p5,
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
        p16,
        size = (2400.0, 1500.0),
        bottom_margin = 20.0 * Plots.PlotMeasures.px,
        left_margin = 20.0 * Plots.PlotMeasures.px,
    )
    png(p, output_filename)
end
