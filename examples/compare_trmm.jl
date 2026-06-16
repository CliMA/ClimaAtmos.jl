using NCDatasets
using CairoMakie

const LES_FILE = "/Users/zshen/Work/clima_repo/les_data/Stats.TRMM_LBA.nc"

function les_timeseries(varname)
    ds = NCDataset(LES_FILE)
    ts = ds.group["timeseries"]
    t = Array(ts["t"][:])
    v = Array(ts[varname][:])
    close(ds)
    return t, v
end

function model_timeseries(output_dir, varname; period = "10m_inst")
    f = joinpath(output_dir, "$(varname)_$(period).nc")
    ds = NCDataset(f)
    t = Array(ds["time"][:])
    v = Array(ds[varname][:])
    close(ds)
    return t, v
end

"""
    rmse_against_les(output_dir, varname; period = "10m_inst")

Interpolate the model timeseries onto the LES time grid (restricted to the
overlapping time range) and compute the RMSE, plus peak magnitudes for both.
"""
function rmse_against_les(output_dir, varname; period = "10m_inst")
    t_les, v_les = les_timeseries(varname)
    t_mod, v_mod = model_timeseries(output_dir, varname; period)

    t_max = min(maximum(t_les), maximum(t_mod))
    mask = t_les .<= t_max
    t_les_c, v_les_c = t_les[mask], v_les[mask]

    v_interp = map(t_les_c) do t
        i = searchsortedlast(t_mod, t)
        i = clamp(i, 1, length(t_mod) - 1)
        t0, t1 = t_mod[i], t_mod[i + 1]
        v0, v1 = v_mod[i], v_mod[i + 1]
        frac = t1 == t0 ? 0.0 : (t - t0) / (t1 - t0)
        return v0 + frac * (v1 - v0)
    end

    rmse = sqrt(sum((v_interp .- v_les_c) .^ 2) / length(v_les_c))
    return (;
        rmse,
        model_max = maximum(v_mod),
        les_max = maximum(v_les_c),
        model_mean = sum(v_mod) / length(v_mod),
        les_mean = sum(v_les_c) / length(v_les_c),
    )
end

function compare_plot(output_dirs::Vector{<:Pair}, varnames; period = "10m_inst", outfile = "trmm_compare.png")
    fig = Figure(size = (500, 350 * length(varnames)))
    for (i, varname) in enumerate(varnames)
        ax = Axis(fig[i, 1]; xlabel = "time (s)", ylabel = varname, title = varname)
        t_les, v_les = les_timeseries(varname)
        lines!(ax, t_les, v_les; label = "LES", color = :black, linewidth = 3)
        for (label, dir) in output_dirs
            t_mod, v_mod = model_timeseries(dir, varname; period)
            lines!(ax, t_mod, v_mod; label = label)
        end
        axislegend(ax; position = :rt)
    end
    save(outfile, fig)
    return outfile
end

if abspath(PROGRAM_FILE) == @__FILE__
    output_dir = length(ARGS) >= 1 ? ARGS[1] : "output/prognostic_edmfx_trmm_column/output_active"
    for varname in ("iwp", "swp", "lwp", "rwp")
        stats = rmse_against_les(output_dir, varname)
        println(
            "$varname: rmse=$(round(stats.rmse, sigdigits=4)) " *
            "model_max=$(round(stats.model_max, sigdigits=4)) les_max=$(round(stats.les_max, sigdigits=4)) " *
            "model_mean=$(round(stats.model_mean, sigdigits=4)) les_mean=$(round(stats.les_mean, sigdigits=4))",
        )
    end
end
