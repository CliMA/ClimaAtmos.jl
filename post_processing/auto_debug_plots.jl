import ClimaAtmos as CA
import ClimaComms
import ClimaCore.InputOutput
import ClimaCore.Fields
import ClimaCore.Spaces
import CairoMakie as MK
import REPL.TerminalMenus as TM
function auto_debug_plots(integrator)
    debugger = CA.get_run_mode(integrator)
    comms_ctx = ClimaComms.context(integrator.u.c)
    (; dt, t) = integrator
    auto_debug_plots(debugger, comms_ctx, dt, t)
end

hdf5day(t) = floor(Int, t / (60 * 60 * 24))
hdf5sec(t) = floor(Int, t % (60 * 60 * 24))
hdf5filename(output_dir, t) =
    joinpath(output_dir, "day$(hdf5day(t)).$(hdf5sec(t)).hdf5")

isinfornans(x) = isnan(x) || isinf(x)
function auto_debug_lat_long(space, colidx)
    col_space = Spaces.column(space, colidx)
    ᶜlg_col = Fields.local_geometry_field(col_space).coordinates
    (ᶜlg_col.lat, ᶜlg_col.long)
end

"""
    get_suspect_columns

Find suspect columns. Notes:
 - they can vary per variable
 - they are likely those that explode the earliest in time
 - its possible that a field may contain very large numbers,
   resulting in a a crashed simulation, before Inf or NaNs
   are reached.

So, per variable, we find the first column to exhibit
exploding behavior (+Inf, -Inf, NaN). To do this, we
march forward in time:
 - If a variable does _not_ have Inf/NaNs anywhere, then find
   the column with the extrema.
 - If a variable has multiple Inf/NaNs, stop, and use the last
   extrema.
 - If a variable has Inf/NaNs in a single column, use that column
   and stop.
"""
function get_suspect_columns(debugger, comms_ctx, dt, t_end)
    if comms_ctx isa ClimaComms.CUDADevice
        @error "get_suspect_columns functionally using ByColumn, which is not supported on the GPU."
        return
    end
    # prob_cols = Dict()
    sus_cols_min = Dict()
    sus_cols_max = Dict()

    (; t_boost_diagnostics, output_dir) = debugger
    YT = map(t_boost_diagnostics:dt:t_end) do t
        reader = InputOutput.HDF5Reader(hdf5filename(output_dir, t), comms_ctx)
        Y = InputOutput.read_field(reader, "Y")
        _t = InputOutput.HDF5.read_attribute(reader.file, "time")
        close(reader)
        @assert t == _t "Time mismatch"
        (Y, _t)
    end
    Ys = first.(YT)
    times = last.(YT)
    reached_inf_or_nans = Dict()
    for prop_chain in Fields.property_chains(first(Ys))
        var_name = join(prop_chain, "_")
        reached_inf_or_nans[var_name] = false
        for (Y, t) in zip(Ys, times) # loop in time
            if reached_inf_or_nans[var_name]
                break
            end
            var = Fields.single_field(Y, prop_chain)
            # First, look for Inf / NaNs
            c = CA.count_columns(var, f -> any(x -> isinfornans(x), f))
            # Let's store c in (sus_cols_min,sus_cols_max) so that
            # we know what case we're in:
            #  - No NaNs
            #  - A single column with NaNs
            #  - Multiple columns with NaNs
            if c == 0
                (colidx, found) =
                    CA.find_column(var, f -> any(x -> x == minimum(var), f))
                sus_cols_min[var_name] = (colidx, c)
                (colidx, found) =
                    CA.find_column(var, f -> any(x -> x == maximum(var), f))
                sus_cols_max[var_name] = (colidx, c)
            elseif c == 1
                (colidx, found) =
                    CA.find_column(var, f -> any(x -> isinfornans(x), f))
                sus_cols_min[var_name] = (colidx, c)
                (colidx, found) =
                    CA.find_column(var, f -> any(x -> isinfornans(x), f))
                sus_cols_max[var_name] = (colidx, c)
                reached_inf_or_nans[var_name] = true
                break
            else
                reached_inf_or_nans[var_name] = true
                break
            end
        end
    end

    return (Ys, times, sus_cols_min, sus_cols_max)
end

function auto_debug_plots(debugger, comms_ctx, dt, t)
    (Ys, times, sus_cols_min, sus_cols_max) =
        get_suspect_columns(debugger, comms_ctx, dt, t)
    multiple_times_per_line_plot(debugger, Ys, times, sus_cols_min, "min")
    multiple_times_per_line_plot(debugger, Ys, times, sus_cols_max, "max")
end

function multiple_times_per_line_plot(debugger, Ys, times, sus_cols, name)
    (; output_dir) = debugger

    @info "Plotting for $name in $output_dir"
    Nt = length(times)
    ps = Fields.property_chains(first(Ys))
    Nvars = length(ps)
    fig = MK.Figure(; size = (1200, 600))
    axs = map(collect(enumerate(ps))) do (i, prop_chain)
        var_name = join(prop_chain, "_")
        ylabel_nt = i == 1 ? (; ylabel = "z [km]") : (;)
        col_idx, c = sus_cols[var_name]
        title = "$name @ col , count(Inf/Nan)=$c"
        MK.Axis(fig[1, i]; ylabel_nt..., xlabel = "$var_name", title)
    end

    for (i, prop_chain) in enumerate(ps)
        var_name = join(prop_chain, "_")
        for (j, (Y, t)) in enumerate(zip(Ys, times)) # loop in time
            var = Fields.single_field(Y, prop_chain)
            (colidx, c) = sus_cols[var_name]
            ϕcol = Fields.column(var, colidx)
            MK.lines!(
                axs[i],
                vec(parent(ϕcol)),
                vec(parent(Fields.coordinate_field(ϕcol).z)) ./ 1e3;
                color = t,
                colormap = :viridis,
                colorrange = (minimum(times), maximum(times)),
            )
        end
    end
    MK.Colorbar(
        fig[1, Nvars + 1];
        limits = (minimum(times), maximum(times)),
        colormap = :viridis,
    )
    MK.save(joinpath(output_dir, "suspect_cols_$name.png"), fig)
end
