#!/usr/bin/env julia
#
# Tables + figures comparing stretched-grid face heights for a toy column
# (z_max=30 km, baseline z_elem=30, dz_bottom=20 m) vs coarsened surface spacing.
# Strategies: baseline | (A) stretch-only | (B) z_elem÷2 | (C) top-Δz search (resolution_ladder.jl).
#
# Default outputs two coarsening levels: dz_bottom=40 m (2×) and 100 m (5× surface spacing).
# Each tier uses an explicit file suffix (`_dzb40`, `_dzb100`) so neither is privileged.
# Also writes one combined 3×2-panel figure (both tiers side by side) plus per-tier figures.
#
# Usage (from this directory):
#   julia --project=. resolution_ladder_grid_comparison.jl
#   julia --project=. resolution_ladder_grid_comparison.jl /path/to/out_dir

import ClimaCore.CommonGrids as CG
import ClimaCore.Meshes as Meshes
import ClimaCore.Geometry as Geometry
using CairoMakie
using Printf

"""Thickness of cell `i` (between face `i` and face `i+1`), in m; `nothing` if `i` is the top face."""
function cell_dz_m(zvec::Vector{Float64}, i::Int)
    i < length(zvec) || return nothing
    return zvec[i + 1] - zvec[i]
end

"""Per-cell Δz (m), length = n_faces - 1."""
function cell_dz_profile(zvec::Vector{Float64})
    return [zvec[i + 1] - zvec[i] for i in 1:(length(zvec) - 1)]
end

"""Cell-center height (m) for each layer: midpoint between face `i` and face `i+1`."""
function cell_z_mid_m(zvec::Vector{Float64})
    return [(zvec[i] + zvec[i + 1]) / 2 for i in 1:(length(zvec) - 1)]
end

function face_heights_m(z_max::Float64, z_elem::Int, dz_bottom::Float64)
    FT = Float64
    mesh = CG.DefaultZMesh(
        FT;
        z_min = FT(0),
        z_max = FT(z_max),
        z_elem,
        stretch = Meshes.HyperbolicTangentStretching{FT}(FT(dz_bottom)),
    )
    return [Float64(Geometry.component(f, 1)) for f in mesh.faces]
end

function _path_with_suffix(path::String, suffix::AbstractString)
    isempty(suffix) && return path
    d, f = splitdir(path)
    base, ext = splitext(f)
    return joinpath(d, base * suffix * ext)
end

"""
    compute_resolution_ladder_tier(z_max, z0, db0, db_coarse; nelem_search=4:120)

Face heights and labels for baseline vs strategies A/B/C for one coarsened `dz_bottom`.
"""
function compute_resolution_ladder_tier(
    z_max::Float64,
    z0::Int,
    db0::Float64,
    db_coarse::Float64;
    nelem_search = 4:120,
)
    z_elem_B = max(4, round(Int, z0 / 2))

    v_base = face_heights_m(z_max, z0, db0)
    target_top_dz = v_base[end] - v_base[end-1]

    best_n_C = -1
    best_err_C = Inf
    for n in nelem_search
        v = face_heights_m(z_max, n, db_coarse)
        err = abs((v[end] - v[end-1]) - target_top_dz)
        if err < best_err_C
            best_err_C, best_n_C = err, n
        end
    end

    labels = ("baseline", "A_same_ne", "B_ne_half", "C_match_topdz")
    zs = (
        v_base,
        face_heights_m(z_max, z0, db_coarse),
        face_heights_m(z_max, z_elem_B, db_coarse),
        face_heights_m(z_max, best_n_C, db_coarse),
    )

    err_s = @sprintf("%.2f", best_err_C)
    dbc_i = round(Int, db_coarse)
    leg_z = [
        "baseline | z_elem=$z0 dz_bottom=$(db0)m",
        "A stretch | z_elem=$z0 dz_bottom=$(dbc_i)m",
        "B z÷2 | z_elem=$z_elem_B dz_bottom=$(dbc_i)m",
        "C top-Δz | z_elem=$best_n_C dz_bottom=$(dbc_i)m err=$(err_s)m",
    ]

    return (
        zs = zs,
        leg_z = leg_z,
        labels = labels,
        dbc_i = dbc_i,
        z_elem_B = z_elem_B,
        best_n_C = best_n_C,
        best_err_C = best_err_C,
        z_max = z_max,
        z0 = z0,
        db0 = db0,
        db_coarse = db_coarse,
        nelem_search = nelem_search,
    )
end

function _write_resolution_ladder_csv(tier, csv_path::String)
    zs = tier.zs
    nmax = maximum(length.(zs))
    z_m_at(i, zvec) = i <= length(zvec) ? zvec[i] : NaN

    open(csv_path, "w") do io
        println(
            io,
            "face_i,baseline_z_m,baseline_dz_m,A_same_ne_z_m,A_same_ne_dz_m,B_ne_half_z_m,B_ne_half_dz_m,C_match_topdz_z_m,C_match_topdz_dz_m",
        )
        for i in 1:nmax
            @printf(io, "%d", i)
            for zvec in zs
                v = z_m_at(i, zvec)
                if isnan(v)
                    print(io, ",,")
                else
                    @printf(io, ",%.6f", v)
                    d = cell_dz_m(zvec, i)
                    if d === nothing
                        print(io, ",")
                    else
                        @printf(io, ",%.6f", d)
                    end
                end
            end
            println(io)
        end
    end
    return nothing
end

function _write_resolution_ladder_table(tier, tbl_path::String)
    zs = tier.zs
    labels = tier.labels
    z_max = tier.z_max
    z0 = tier.z0
    db0 = tier.db0
    db_coarse = tier.db_coarse
    nmax = maximum(length.(zs))

    open(tbl_path, "w") do io
        factor = db_coarse / db0
        println(
            io,
            "Toy column: z_max=$(z_max/1000) km; baseline z_elem=$z0 dz_bottom=$db0 m; coarsened tier dz_bottom=$db_coarse m ($(round(factor, digits=2))× baseline surface spacing).",
        )
        println(io, "Column keys: baseline | A_same_ne (stretch only) | B_ne_half (naive) | C_match_topdz (argmin |Δz_top−baseline|).")
        println(io, "")
        println(io, "UNITS: all lengths in meters (SI). z_*_m = face height; *_dz_m = cell thickness on that face.")
        println(io, "       Stretched column: thin BL cells vs thick top cell. Model top z ≈ $(round(Int, z_max)) m.")
        println(io, "")
        wz = 14
        wd = 14
        gsp = 2
        println(io, "— Face table: z and Δz for the layer on face i (blank Δz on the model-top face). —")
        println(io, "")
        col_pair(lab) =
            rpad(string(lab) * "_z_m", wz) * rpad(string(lab) * "_dz_m", wd)
        header = rpad("face", 6) * join([col_pair(lab) for lab in labels], " "^gsp)
        println(io, header)
        println(io, "-"^min(140, length(header)))
        for i in 1:nmax
            print(io, rpad(string(i), 6))
            for (zi, zvec) in enumerate(zs)
                if i <= length(zvec)
                    print(io, rpad(@sprintf("%.2f", zvec[i]), wz))
                    d = cell_dz_m(zvec, i)
                    s = d === nothing ? "" : @sprintf("%.2f", d)
                    print(io, rpad(s, wd))
                else
                    print(io, rpad("", wz + wd))
                end
                zi < length(zs) && print(io, " "^gsp)
            end
            println(io)
        end

        ncell_max = maximum(length.(zs)) - 1
        println(io, "")
        println(io, "— Cell table: Δz by cell index j (layer j = faces j → j+1), m. —")
        println(io, "")
        wc = 14
        h2 = rpad("cell_j", 8) * join([rpad(string(lab) * "_dz_m", wc) for lab in labels], " "^gsp)
        println(io, h2)
        println(io, "-"^length(h2))
        for j in 1:ncell_max
            print(io, rpad(string(j), 8))
            for (zi, zvec) in enumerate(zs)
                if j < length(zvec)
                    d = cell_dz_m(zvec, j)
                    print(io, rpad(@sprintf("%.2f", d::Float64), wc))
                else
                    print(io, rpad("", wc))
                end
                zi < length(zs) && print(io, " "^gsp)
            end
            println(io)
        end
    end
    return nothing
end

const _WONG = Makie.wong_colors()

function _plot_face_heights!(ax, tier; title::String, subtitle::String)
    z_max = tier.z_max
    zs = tier.zs
    leg_z = tier.leg_z
    z_plot_lo = 5.0
    for (k, zvec) in enumerate(zs)
        x = collect(2:length(zvec))
        y = zvec[2:end]
        scatterlines!(
            ax,
            x,
            y;
            color = _WONG[k],
            linewidth = 1.75,
            markersize = 9,
            marker = :circle,
            strokewidth = 1.25,
            strokecolor = :white,
            label = leg_z[k],
        )
    end
    ax.xlabel = "face index (surface z=0 omitted from this panel)"
    ax.ylabel = "height z (m)"
    ax.yscale = log10
    ax.title = title
    ax.subtitle = subtitle
    ylims!(ax, z_plot_lo, z_max * 1.02)
    return ax
end

function _plot_dz_vs_index!(ax, tier; title::String, subtitle::String)
    zs = tier.zs
    leg_z = tier.leg_z
    for (k, zvec) in enumerate(zs)
        dzv = cell_dz_profile(zvec)
        x = collect(1:length(dzv))
        scatterlines!(
            ax,
            x,
            dzv;
            color = _WONG[k],
            linewidth = 1.75,
            markersize = 9,
            marker = :circle,
            strokewidth = 1.25,
            strokecolor = :white,
            label = leg_z[k],
        )
    end
    ax.xlabel = "cell index (1 = surface layer)"
    ax.ylabel = "Δz (m)"
    ax.yscale = log10
    ax.title = title
    ax.subtitle = subtitle
    ylims!(ax, 1.0, nothing)
    return ax
end

function _plot_dz_vs_z!(ax, tier; title::String, subtitle::String)
    z_max = tier.z_max
    zs = tier.zs
    leg_z = tier.leg_z
    for (k, zvec) in enumerate(zs)
        dzv = cell_dz_profile(zvec)
        xm = cell_z_mid_m(zvec)
        scatterlines!(
            ax,
            dzv,
            xm;
            color = _WONG[k],
            linewidth = 1.75,
            markersize = 9,
            marker = :circle,
            strokewidth = 1.25,
            strokecolor = :white,
            label = leg_z[k],
        )
    end
    ax.xlabel = "Δz (m)"
    ax.ylabel = "cell-center height z (m)"
    ax.xscale = log10
    ax.yscale = log10
    ax.title = title
    ax.subtitle = subtitle
    xlims!(ax, 1.0, nothing)
    ylims!(ax, 1.0, z_max * 1.02)
    return ax
end

"""
    resolution_ladder_tier_outputs(out_dir, z_max, z0, db0, db_coarse; nelem_search=4:120, file_suffix="_dzb40")

Write CSV, text table, combined figure (faces, Δz vs index, Δz vs cell-center z), and Δz-only figure for one coarsened `dz_bottom = db_coarse`.
`file_suffix` (e.g. `\"_dzb40\"`, `\"_dzb100\"`) is inserted before the extension; use explicit suffixes for every tier.
"""
function resolution_ladder_tier_outputs(
    out_dir::AbstractString,
    z_max::Float64,
    z0::Int,
    db0::Float64,
    db_coarse::Float64;
    nelem_search = 4:120,
    file_suffix::AbstractString,
)
    tier = compute_resolution_ladder_tier(z_max, z0, db0, db_coarse; nelem_search)

    csv_path = _path_with_suffix(joinpath(out_dir, "resolution_ladder_face_heights.csv"), file_suffix)
    tbl_path = _path_with_suffix(joinpath(out_dir, "resolution_ladder_face_heights_table.txt"), file_suffix)
    _write_resolution_ladder_csv(tier, csv_path)
    _write_resolution_ladder_table(tier, tbl_path)

    dbc_i = tier.dbc_i
    ne = tier.nelem_search
    leg_fs = 9

    fig = Figure(size = (1280, 1380))
    _plot_face_heights!(
        Axis(fig[1, 1]),
        tier;
        title = "Face heights — DefaultZMesh (dz_bottom=$(dbc_i) m tier)",
        subtitle = "log₁₀ z. C: argmin over z_elem∈[$(first(ne)),$(last(ne))] of |Δz_top−baseline|.",
    )
    _plot_dz_vs_index!(
        Axis(fig[2, 1]),
        tier;
        title = "Δz vs cell index",
        subtitle = "log₁₀ Δz; same meshes as top panel",
    )
    _plot_dz_vs_z!(
        Axis(fig[3, 1]),
        tier;
        title = "Layer thickness vs height (cell-center z on vertical axis)",
        subtitle = "log–log; z_mid = (z_face[i]+z_face[i+1])/2 per layer",
    )

    ax_ref = contents(fig[1, 1])[1]
    Legend(fig[1:3, 2], ax_ref; framevisible = true, fontsize = leg_fs)
    colgap!(fig.layout, 20)
    colsize!(fig.layout, 1, Auto(1))
    colsize!(fig.layout, 2, Auto(0.28))
    rowgap!(fig.layout, 20)

    fig_path = _path_with_suffix(joinpath(out_dir, "resolution_ladder_face_heights.png"), file_suffix)
    save(fig_path, fig)

    fig_dz = Figure(size = (1280, 1000))
    _plot_dz_vs_index!(
        Axis(fig_dz[1, 1]),
        tier;
        title = "Δz vs cell index — dz_bottom=$(dbc_i) m tier",
        subtitle = "Toy column z_max=$(tier.z_max/1000) km; log₁₀ Δz",
    )
    _plot_dz_vs_z!(
        Axis(fig_dz[2, 1]),
        tier;
        title = "Layer thickness vs height (z on vertical axis)",
        subtitle = "log–log; z_mid = (z_face[i]+z_face[i+1])/2",
    )

    ax_dz_ref = contents(fig_dz[1, 1])[1]
    Legend(fig_dz[1:2, 2], ax_dz_ref; framevisible = true, fontsize = leg_fs)
    colgap!(fig_dz.layout, 20)
    colsize!(fig_dz.layout, 1, Auto(1))
    colsize!(fig_dz.layout, 2, Auto(0.28))
    rowgap!(fig_dz.layout, 16)
    fig_dz_path = _path_with_suffix(joinpath(out_dir, "resolution_ladder_cell_dz.png"), file_suffix)
    save(fig_dz_path, fig_dz)

    return (
        csv = csv_path,
        table = tbl_path,
        fig_heights = fig_path,
        fig_dz = fig_dz_path,
        tier = tier,
    )
end

"""
    resolution_ladder_combined_both_tiers(out_dir, tier_left, tier_right; filename=\"resolution_ladder_face_heights_both_tiers.png\")

3 rows × 2 columns of data panels: left = `tier_left`, right = `tier_right` (e.g. dzb40 vs dzb100).

Two legend columns (not one): baseline / A / B / C use the **same colors** in both columns, but C’s optimal `z_elem` (and thus point counts) differ by tier, so each column gets its own legend built from that column’s top axis.
"""
function resolution_ladder_combined_both_tiers(
    out_dir::AbstractString,
    tier_left,
    tier_right;
    filename::String = "resolution_ladder_face_heights_both_tiers.png",
)
    dbc_L = tier_left.dbc_i
    dbc_R = tier_right.dbc_i
    ne_L = tier_left.nelem_search
    leg_fs = 7

    fig = Figure(size = (1920, 1380))

    _plot_face_heights!(
        Axis(fig[1, 1]),
        tier_left;
        title = "Face heights — dz_bottom=$(dbc_L) m",
        subtitle = "C: argmin z_elem∈[$(first(ne_L)),$(last(ne_L))]",
    )
    _plot_dz_vs_index!(
        Axis(fig[2, 1]),
        tier_left;
        title = "Δz vs cell index",
        subtitle = "log₁₀ Δz",
    )
    _plot_dz_vs_z!(
        Axis(fig[3, 1]),
        tier_left;
        title = "Δz vs cell-center z",
        subtitle = "log–log",
    )

    ne_R = tier_right.nelem_search
    _plot_face_heights!(
        Axis(fig[1, 2]),
        tier_right;
        title = "Face heights — dz_bottom=$(dbc_R) m",
        subtitle = "C: argmin z_elem∈[$(first(ne_R)),$(last(ne_R))]",
    )
    _plot_dz_vs_index!(
        Axis(fig[2, 2]),
        tier_right;
        title = "Δz vs cell index",
        subtitle = "log₁₀ Δz",
    )
    _plot_dz_vs_z!(
        Axis(fig[3, 2]),
        tier_right;
        title = "Δz vs cell-center z",
        subtitle = "log–log",
    )

    ax_L = contents(fig[1, 1])[1]
    ax_R = contents(fig[1, 2])[1]
    Legend(fig[1:3, 3], ax_L; framevisible = true, fontsize = leg_fs, title = "dz_bottom=$(dbc_L) m", titlefontsize = leg_fs + 1)
    Legend(fig[1:3, 4], ax_R; framevisible = true, fontsize = leg_fs, title = "dz_bottom=$(dbc_R) m", titlefontsize = leg_fs + 1)
    colgap!(fig.layout, 12)
    colsize!(fig.layout, 1, Auto(1))
    colsize!(fig.layout, 2, Auto(1))
    colsize!(fig.layout, 3, Auto(0.2))
    colsize!(fig.layout, 4, Auto(0.2))
    rowgap!(fig.layout, 18)

    out_path = joinpath(out_dir, filename)
    save(out_path, fig)
    return out_path
end

function main()
    out_dir = get(ARGS, 1, joinpath(@__DIR__, "..", "analysis", "figures", "resolution_ladder"))
    mkpath(out_dir)

    z_max = 30000.0
    z0, db0 = 30, 20.0

    p40 = resolution_ladder_tier_outputs(out_dir, z_max, z0, db0, 40.0; file_suffix = "_dzb40")
    p100 = resolution_ladder_tier_outputs(out_dir, z_max, z0, db0, 100.0; file_suffix = "_dzb100")

    combined_path = resolution_ladder_combined_both_tiers(out_dir, p40.tier, p100.tier)

    println("Wrote (2× tier, dz_bottom=40 m, suffix _dzb40):")
    println("  ", p40.csv)
    println("  ", p40.table)
    println("  ", p40.fig_heights)
    println("  ", p40.fig_dz)
    println("Wrote (5× tier, dz_bottom=100 m, suffix _dzb100):")
    println("  ", p100.csv)
    println("  ", p100.table)
    println("  ", p100.fig_heights)
    println("  ", p100.fig_dz)
    println("Wrote (combined 3×2 both tiers):")
    println("  ", combined_path)
    return nothing
end

main()
