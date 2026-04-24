#!/usr/bin/env julia
#
# Side-by-side (or overlaid) comparison: HadGEM2 July forcing `hur` (siteNN) vs SCM output `hur_10m_inst.nc`.
# Forcing `hur` is CMIP-style percent; SCM diagnostic is fraction — both plotted as fraction [0,1] for comparison.
#
# Usage (from experiment root or pass absolute paths):
#   julia --project=../.. analysis/scripts/plot_compare_hur_forcing_vs_sim.jl \
#       /path/to/HadGEM2-A_amip.2004-2008.07.nc site11 \
#       /path/to/simulation_output/.../hur_10m_inst.nc \
#       /path/to/out.png
#
import NCDatasets as NC
import CairoMakie as CM
import Statistics: mean

function forcing_mean_hur_profile(forcing_path::AbstractString, site::AbstractString)
    ds = NC.Dataset(forcing_path)
    g = ds.group[site]
    hur = g["hur"][:, :]   # lev × time (HadGEM2 cfSite groups)
    lev = g["lev"][:]      # m
    # time-mean profile; hur stored as percent in CMIP-style files
    μ = vec(mean(hur, dims = 2))
    NC.close(ds)
    return lev, μ ./ 100  # fraction
end

function sim_mean_hur_profile(sim_hur_path::AbstractString)
    ds = NC.Dataset(sim_hur_path)
    h = ds["hur"][:, :, :, :]  # time × x × y × z
    z = ds["z"][:]
    # mean over time and horizontal columns
    μ = vec(dropdims(mean(mean(mean(h, dims = 1), dims = 2), dims = 3), dims = (1, 2, 3)))
    NC.close(ds)
    return z, μ  # already fraction
end

function main()
    length(ARGS) >= 3 || error(
        "Usage: julia plot_compare_hur_forcing_vs_sim.jl FORCING_NC SITE sim_hur.nc [out.png]",
    )
    forcing_path = ARGS[1]
    site = ARGS[2]
    sim_path = ARGS[3]
    outpath = length(ARGS) >= 4 ? ARGS[4] : joinpath(@__DIR__, "..", "figures", "hur_forcing_vs_sim_$(site).png")

    lev_f, hur_f = forcing_mean_hur_profile(forcing_path, site)
    z_s, hur_s = sim_mean_hur_profile(sim_path)

    zmax = max(maximum(skipmissing(lev_f)), maximum(z_s))
    hur_f_f = Float64[c for c in hur_f]
    lev_f_f = Float64[c for c in lev_f]
    hur_s_f = Float64[c for c in hur_s]
    z_s_f = Float64[c for c in z_s]

    fig = CM.Figure(; size = (1000, 520))
    tit = "HadGEM2 July ($site) vs SCM | $(basename(sim_path))"
    CM.Label(fig[0, :], tit; fontsize = 14)

    ax1 = CM.Axis(
        fig[1, 1];
        xlabel = "RH (fraction)",
        ylabel = "Height (m)",
        title = "Forcing: time-mean hur (÷100)",
    )
    CM.lines!(ax1, hur_f_f, lev_f_f; linewidth = 2.5, color = :steelblue)
    CM.ylims!(ax1, 0, zmax)
    CM.xlims!(ax1, 0, max(1.05, maximum(hur_f_f) * 1.05))

    ax2 = CM.Axis(
        fig[1, 2];
        xlabel = "RH (fraction)",
        ylabel = "Height (m)",
        title = "SCM: mean over t,x,y",
    )
    CM.lines!(ax2, hur_s_f, z_s_f; linewidth = 2.5, color = :orangered)
    CM.ylims!(ax2, 0, zmax)
    CM.xlims!(ax2, 0, max(1.05, maximum(hur_s_f) * 1.05))

    axo = CM.Axis(
        fig[2, :];
        xlabel = "RH (fraction)",
        ylabel = "Height (m)",
        title = "Overlay (same axes)",
    )
    CM.lines!(axo, hur_f_f, lev_f_f; label = "Forcing", linewidth = 2.5)
    CM.lines!(axo, hur_s_f, z_s_f; label = "SCM", linewidth = 2.5, linestyle = :dash)
    CM.axislegend(axo; position = :rt)
    CM.ylims!(axo, 0, zmax)
    CM.xlims!(axo, 0, max(1.05, max(maximum(hur_f_f), maximum(hur_s_f)) * 1.05))

    CM.rowsize!(fig.layout, 2, CM.Auto(0.35))
    CM.save(outpath, fig)
    @info "Wrote" outpath
    return nothing
end

main()
