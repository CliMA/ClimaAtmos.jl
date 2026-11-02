# post_processing/rico_numerics_sensitivity.jl
#
# Time-series numerics-sensitivity overlay for the 6
# `prognostic_edmfx_rico_sens_<i>` runs. Produces a single 6-panel PDF
# overlaying `lwp(t)`, `rwp(t)`, `waup(t)` at z=1km and z=2km, and `arup(t)`
# at z=1km and z=2km.
#
# All dependencies live in the shared .buildkite environment; always use
# --project=.buildkite (the CI Initialize step pre-installs it).
#
# Usage (from the repo root):
#   julia --project=.buildkite \
#     post_processing/rico_numerics_sensitivity.jl \
#     [--sim_root <dir>]   # root that contains the <job_id>/output_active/nc_files/
#                          # subdirectories; defaults to the current working dir
#     [--output <file>]    # output PDF path; default rico_numerics_sensitivity.pdf

import ArgParse: ArgParseSettings, @add_arg_table!, parse_args
import NCDatasets: NCDataset
import CairoMakie

const S = ArgParseSettings()
#! format: off
@add_arg_table! S begin
    "--sim_root"
        help = "Root with <job_id>/output_active/nc_files/ subdirs"
        default = "."
    "--output"
        help = "Output PDF path"
        default = "rico_numerics_sensitivity.pdf"
end
#! format: on
const ARGS_PARSED = parse_args(S)

# The six runs and short descriptive labels. The order sets the legend order.
const RUNS = [
    ("prognostic_edmfx_rico_sens_1",
        "1: dt=10s, picard=2, cache=stage, exp, N=1"),
    ("prognostic_edmfx_rico_sens_2",
        "2: dt=120s, picard=0, cache=step, exp, N=1"),
    ("prognostic_edmfx_rico_sens_3",
        "3: dt=120s, picard=2, cache=step, exp, N=1"),
    ("prognostic_edmfx_rico_sens_4",
        "4: dt=120s, picard=2, cache=stage, exp, N=1"),
    ("prognostic_edmfx_rico_sens_5",
        "5: dt=120s, picard=2, cache=stage, imp, N=1"),
    ("prognostic_edmfx_rico_sens_6",
        "6: dt=120s, picard=2, cache=stage, imp, N=3"),
]

const PERIOD_TAG = "10m"
const REDUCTION_TAG = "inst"

const COLORS = (
    CairoMakie.RGBf(0.00, 0.45, 0.70),   # blue
    CairoMakie.RGBf(0.90, 0.60, 0.00),   # orange
    CairoMakie.RGBf(0.00, 0.60, 0.50),   # teal
    CairoMakie.RGBf(0.80, 0.40, 0.00),   # vermillion
    CairoMakie.RGBf(0.80, 0.40, 0.70),   # magenta
    CairoMakie.RGBf(0.35, 0.35, 0.35),   # dark grey
)

sim_dir(root, job_id) = joinpath(root, job_id, "output_active", "nc_files")

sim_nc_path(root, job_id, short_name) =
    joinpath(sim_dir(root, job_id), "$(short_name)_$(PERIOD_TAG)_$(REDUCTION_TAG).nc")

# Load `(t_hours, values)` for a scalar time series. Returns `(nothing, nothing)`
# if the file is missing or unreadable.
function load_ts(root, job_id, short_name)
    path = sim_nc_path(root, job_id, short_name)
    isfile(path) || return (nothing, nothing)
    try
        ds = NCDataset(path)
        t = ds["time"].var[:]
        y = ds[short_name][:]
        close(ds)
        return t ./ 3600.0, y
    catch e
        @warn "Could not load timeseries $short_name from $path: $e"
        return (nothing, nothing)
    end
end

# Load `(t_hours, values_at_z)` for a profile diagnostic linearly interpolated
# to a target height `z_target` (meters). Returns `(nothing, nothing)` if the
# file is missing or unreadable.
function load_profile_at_z(root, job_id, short_name, z_target)
    path = sim_nc_path(root, job_id, short_name)
    isfile(path) || return (nothing, nothing)
    try
        ds = NCDataset(path)
        t = ds["time"].var[:]
        z = ds["z"][:]
        data = ds[short_name][:, :]
        close(ds)
        iu = findfirst(zi -> zi >= z_target, z)
        if iu === nothing
            @warn "z_target=$z_target above domain top for $short_name in $path"
            return (nothing, nothing)
        end
        if iu == 1
            vals = data[:, 1]
        else
            il = iu - 1
            w = (z_target - z[il]) / (z[iu] - z[il])
            vals = (1 - w) .* data[:, il] .+ w .* data[:, iu]
        end
        return t ./ 3600.0, vec(vals)
    catch e
        @warn "Could not load profile $short_name from $path: $e"
        return (nothing, nothing)
    end
end

const Z1 = 1000.0
const Z2 = 2000.0

fig = CairoMakie.Figure(size = (1200, 1500))
ax_lwp = CairoMakie.Axis(fig[1, 1];
    title = "liquid water path (lwp)",
    xlabel = "time [h]", ylabel = "LWP [g/m²]")
ax_rwp = CairoMakie.Axis(fig[1, 2];
    title = "rain water path (rwp)",
    xlabel = "time [h]", ylabel = "RWP [g/m²]")
ax_wa1 = CairoMakie.Axis(fig[2, 1];
    title = "updraft w at z=$(Int(Z1)) m",
    xlabel = "time [h]", ylabel = "w [m/s]")
ax_wa2 = CairoMakie.Axis(fig[2, 2];
    title = "updraft w at z=$(Int(Z2)) m",
    xlabel = "time [h]", ylabel = "w [m/s]")
ax_ar1 = CairoMakie.Axis(fig[3, 1];
    title = "updraft area at z=$(Int(Z1)) m",
    xlabel = "time [h]", ylabel = "a [-]")
ax_ar2 = CairoMakie.Axis(fig[3, 2];
    title = "updraft area at z=$(Int(Z2)) m",
    xlabel = "time [h]", ylabel = "a [-]")

for (i, (job_id, label)) in enumerate(RUNS)
    color = COLORS[i]

    (t_lwp, lwp) = load_ts(ARGS_PARSED["sim_root"], job_id, "lwp")
    if !isnothing(t_lwp)
        # kg/m² → g/m² for human-readable LWP
        CairoMakie.lines!(ax_lwp, t_lwp, lwp .* 1e3; color, linewidth = 2, label)
    end

    (t_rwp, rwp) = load_ts(ARGS_PARSED["sim_root"], job_id, "rwp")
    if !isnothing(t_rwp)
        CairoMakie.lines!(ax_rwp, t_rwp, rwp .* 1e3; color, linewidth = 2, label)
    end

    (t_w1, w1) = load_profile_at_z(ARGS_PARSED["sim_root"], job_id, "waup", Z1)
    if !isnothing(t_w1)
        CairoMakie.lines!(ax_wa1, t_w1, w1; color, linewidth = 2, label)
    end

    (t_w2, w2) = load_profile_at_z(ARGS_PARSED["sim_root"], job_id, "waup", Z2)
    if !isnothing(t_w2)
        CairoMakie.lines!(ax_wa2, t_w2, w2; color, linewidth = 2, label)
    end

    (t_a1, a1) = load_profile_at_z(ARGS_PARSED["sim_root"], job_id, "arup", Z1)
    if !isnothing(t_a1)
        CairoMakie.lines!(ax_ar1, t_a1, a1; color, linewidth = 2, label)
    end

    (t_a2, a2) = load_profile_at_z(ARGS_PARSED["sim_root"], job_id, "arup", Z2)
    if !isnothing(t_a2)
        CairoMakie.lines!(ax_ar2, t_a2, a2; color, linewidth = 2, label)
    end
end

CairoMakie.axislegend(ax_lwp; position = :rt, framevisible = false, labelsize = 10)

mkpath(dirname(ARGS_PARSED["output"]) == "" ? "." : dirname(ARGS_PARSED["output"]))
CairoMakie.save(ARGS_PARSED["output"], fig)
@info "Wrote $(ARGS_PARSED["output"])"
