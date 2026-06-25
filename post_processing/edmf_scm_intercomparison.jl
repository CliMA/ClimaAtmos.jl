# post_processing/edmf_scm_intercomparison.jl
#
# General EDMF intercomparison: profiles + timeseries comparing ClimaAtmos
# (blue) against PyCLES reference (orange). Variables absent from PyCLES are
# silently skipped on that side; the plot still shows ClimaAtmos data.
#
# Layout (12-column grid):
#   Row 1 profiles : θ_l | q_t | q_l | q_r | q_i | q_s
#   Row 2 profiles : a_up | w_up | TKE | CF  | w_mean | u & v
#   Row 3 timeseries: LWP [1:3] | RWP [4:6] | IWP [7:9] | SWP [10:12]
#
# Usage:
#   julia --project=post_processing post_processing/edmf_scm_intercomparison.jl \
#       --case BOMEX --sim_dir <nc_files_dir> \
#       [--ref_nc <PyCLES.nc>]  # omit to use the pycles_<case> artifact
#       [--output <out.pdf>]
#
# When --ref_nc is omitted the script looks for a single artifact named
# "pycles_scm_les_data" (populated from post_processing/Artifacts.toml) and
# expects the reference NC file at <artifact_dir>/<case>.nc.

import ArgParse: ArgParseSettings, @add_arg_table!, parse_args
import NCDatasets: NCDataset
import CairoMakie
import Statistics: mean, std

# ── CLI ───────────────────────────────────────────────────────────────────────
s = ArgParseSettings()
@add_arg_table! s begin
    "--sim_dir"
    help = "Path to ClimaAtmos nc_files directory"
    default = ""
    "--ref_nc"
    help = "Path to PyCLES reference NetCDF file (omit to use Julia artifact)"
    default = ""
    "--output"
    help = "Output PDF path (defaults to <case>_intercomparison.pdf in cwd)"
    default = ""
    "--case"
    help = "Case name, e.g. Bomex, GABLS, DYCOMS_RF01 (required)"
    required = true
end
args = parse_args(s)

sim_dir_path = args["sim_dir"]
ref_nc_path = args["ref_nc"]
out_path = args["output"]
case_name = args["case"]

# ── Default output path ───────────────────────────────────────────────────────
if isempty(out_path)
    out_path = "$(lowercase(case_name))_intercomparison.pdf"
end

# ── Per-case geostrophic wind offsets ─────────────────────────────────────────
# PyCLES stores horizontal wind *anomaly* from the case geostrophic wind, while
# ClimaAtmos stores the absolute wind.  Subtracting (u_geo, v_geo) from the
# ClimaAtmos profiles before plotting puts both models in the same frame.
# Values from AtmosphericProfilesLibrary / the original DYCOMS papers.
const UV_GEO = Dict(
    "DYCOMS_RF01" => (u = 7.0,  v = -5.5),   # APL.Dycoms_RF01_u0, Dycoms_RF01_v0
    "DYCOMS_RF02" => (u = 0.0,  v =  0.0),   # RF02 uses divergence-driven wind, no offset needed
)
const uv_offset = get(UV_GEO, case_name, (u = 0.0, v = 0.0))

# Per-case hard floor on the u/v x-axis lower bound (m/s, in the PyCLES anomaly frame).
# Useful when absolute geostrophic winds push all data to one side of zero.
const UV_XMIN = Dict(
    "GABLS" => -10.0,   # GABLS geostrophic wind is ~8 m/s; extend left for context
)

# ── Resolve reference NC via artifact if --ref_nc not given ──────────────────
if isempty(ref_nc_path)
    artifact_name = "pycles_scm_les_data"
    try
        # Use find_artifacts_toml to walk up to the repo-root Artifacts.toml,
        # matching the same convention used by @clima_artifact / @artifact_str.
        import Pkg.Artifacts: ensure_artifact_installed, artifact_meta
        import Artifacts: find_artifacts_toml
        toml_path = find_artifacts_toml(dirname(@__DIR__))
        isnothing(toml_path) && error("No Artifacts.toml found above $(@__DIR__)")
        meta = artifact_meta(artifact_name, toml_path)
        isnothing(meta) && error("No artifact '$artifact_name' found in $toml_path")
        adir = ensure_artifact_installed(artifact_name, meta, toml_path)
        global ref_nc_path = joinpath(adir, "$(case_name).nc")
    catch e
        error(
            "No --ref_nc supplied and artifact lookup for '$artifact_name' failed: $e\n" *
            "Either pass --ref_nc or ensure the root Artifacts.toml is populated.",
        )
    end
end

# ── Reference data ────────────────────────────────────────────────────────────
println("Loading reference: $ref_nc_path")
ref_ds = NCDataset(ref_nc_path)
prof = ref_ds.group["profiles"]
ts_grp = ref_ds.group["timeseries"]

# Auto-detect PyCLES export format:
#   Old format: t and z_half are top-level; density is prof["rho"].
#   New format (Stats.*.nc): t and z_half live inside the profiles group;
#   density is in the reference group as rho0; water paths are direct
#   timeseries variables (lwp, rwp, iwp, swp).
ref_fmt_new = !haskey(ref_ds, "t")
ref_t = (ref_fmt_new ? prof["t"] : ref_ds["t"])[:]           # seconds
ref_z = (ref_fmt_new ? prof["z_half"] : ref_ds["z_half"])[:]  # metres
ref_rho = ref_fmt_new ? ref_ds.group["reference"]["rho0"][:] : prof["rho"][:]

# Averaging window: last LAST_N_HOURS of the run, anchored at the ClimaAtmos
# t_end when sim data is available.  This ensures both sides average the same
# absolute time period even when PyCLES ran longer than ClimaAtmos.
LAST_N_HOURS = 3.0

# Sniff ClimaAtmos t_end from any available profile file (before opening ref).
function _sim_t_end(sim_dir)
    isempty(sim_dir) && return nothing
    !isdir(sim_dir) && return nothing
    # Try a handful of common profile names to find the first available file.
    for sn in ("thetaa", "hus", "ta", "wa", "ua")
        p = joinpath(sim_dir, "$(sn)_10m_inst.nc")
        if isfile(p)
            try
                ds = NCDataset(p)
                t  = ds["time"][end]
                close(ds)
                return Float64(t)
            catch
            end
        end
    end
    return nothing
end

_atmos_t_end = _sim_t_end(sim_dir_path)

# Anchor the PyCLES window at the ClimaAtmos t_end (if known and earlier than
# the PyCLES run end), so profiles are time-matched between the two models.
ref_t_end     = ref_t[end]
ref_t_avg_end = isnothing(_atmos_t_end) ? ref_t_end : min(ref_t_end, _atmos_t_end)
ref_t_start_avg = ref_t_avg_end - LAST_N_HOURS * 3600.0
ref_mask = (ref_t .>= ref_t_start_avg) .& (ref_t .<= ref_t_avg_end)
if sum(ref_mask) == 0
    ref_mask = ref_t .>= ref_t[end] / 2
    @warn "Last-$(LAST_N_HOURS)h window empty; using last half of run."
end
println("Reference: averaging last $(LAST_N_HOURS)h (",
    round(ref_t_start_avg / 3600, digits = 1), "–",
    round(ref_t_avg_end / 3600, digits = 1), "h)")

# NCDatasets returns (z, t) for profile groups → transpose to (t, z)
function load_prof(varname)
    return permutedims(prof[varname][:, :])
end

function ref_stats(field_tz; scale = 1.0)
    w = field_tz[ref_mask, :]
    mu = mean(w, dims = 1)[1, :] .* scale
    sig = std(w, dims = 1)[1, :] .* scale
    return mu, sig
end

# Graceful loader: returns (nothing, nothing) when variable absent
function try_ref_stats(varname; scale = 1.0)
    haskey(prof, varname) || return nothing, nothing
    return ref_stats(load_prof(varname); scale)
end

# Core profiles (always present in PyCLES output)
ref_thetali, ref_thetali_s = ref_stats(load_prof("thetali_mean"))
ref_qt, ref_qt_s = ref_stats(load_prof("qt_mean"); scale = 1e3)
ref_ql, ref_ql_s = ref_stats(load_prof("ql_mean"); scale = 1e3)
ref_aup, ref_aup_s = ref_stats(load_prof("updraft_fraction"))
ref_wup, ref_wup_s = ref_stats(load_prof("updraft_w"))
ref_tke, ref_tke_s = ref_stats(load_prof("tke_mean"))

# Optional profiles (case dependent; GABLS has no cloud_fraction etc.)
ref_cf, ref_cf_s = try_ref_stats("cloud_fraction")
ref_qr, ref_qr_s = try_ref_stats("qr_mean"; scale = 1e3)
ref_qi, ref_qi_s = try_ref_stats("qi_mean"; scale = 1e3)
ref_qs, ref_qs_s = try_ref_stats("qs_mean"; scale = 1e3)
ref_wmean, ref_wmean_s = try_ref_stats("w_mean")
ref_umean, ref_umean_s = try_ref_stats("u_mean")
ref_vmean, ref_vmean_s = try_ref_stats("v_mean")

# Reference timeseries
ref_t_h = ref_t ./ 3600.0

# LWP: old format uses 'lwp_mean', new format uses 'lwp'
ref_lwp = if haskey(ts_grp, "lwp_mean")
    ts_grp["lwp_mean"][:] .* 1e3   # g/m²
elseif haskey(ts_grp, "lwp")
    ts_grp["lwp"][:] .* 1e3        # g/m²
else
    zeros(length(ref_t))
end

# Rain water path: read directly from new-format timeseries, or integrate profile
ref_rwp = if haskey(ts_grp, "rwp")
    ts_grp["rwp"][:] .* 1e3        # g/m² (new format)
elseif !isnothing(ref_qr)
    dz = diff(ref_z)
    dz = vcat(dz[1:1], (dz[1:(end - 1)] .+ dz[2:end]) ./ 2, dz[end:end])
    clamp.(load_prof("qr_mean") * (ref_rho .* dz), 0.0, Inf) .* 1e3   # g/m²
else
    nothing
end

# Ice and snow water paths (available in new-format timeseries)
ref_iwp_ts = haskey(ts_grp, "iwp") ? ts_grp["iwp"][:] .* 1e3 : nothing  # g/m²
ref_swp_ts = haskey(ts_grp, "swp") ? ts_grp["swp"][:] .* 1e3 : nothing  # g/m²

close(ref_ds)

# ── ClimaAtmos simulation ─────────────────────────────────────────────────────
has_sim = !isempty(sim_dir_path) && isdir(sim_dir_path)

function sim_nc_path(short_name; reduction = "inst", period = "10m")
    joinpath(sim_dir_path, "$(short_name)_$(period)_$(reduction).nc")
end

# Returns (z_m, mean, std) averaged over [ref_t_avg_end - LAST_N_HOURS, ref_t_avg_end];
# nothing-tuple if unavailable.  Using the shared anchor keeps both models
# time-matched even when their run durations differ.
function load_and_avg(short_name; scale = 1.0)
    path = sim_nc_path(short_name)
    isfile(path) || return nothing, nothing, nothing
    try
        ds   = NCDataset(path)
        t_ax = ds["time"][:]
        z_ax = ds["z"][:]
        data = ds[short_name][:, :] .* scale
        close(ds)
        t0   = ref_t_avg_end - LAST_N_HOURS * 3600.0
        mask = (t_ax .>= t0) .& (t_ax .<= ref_t_avg_end)
        sum(mask) == 0 && (mask = t_ax .>= t_ax[end] / 2)
        w = data[mask, :]
        return z_ax, mean(w, dims = 1)[1, :], std(w, dims = 1)[1, :]
    catch e
        @warn "Could not load profile $short_name: $e"
        return nothing, nothing, nothing
    end
end

# Returns (t_h, values) for a scalar timeseries clipped to ref_t_avg_end;
# nothing-tuple if unavailable.
function load_ts(short_name; scale = 1.0)
    path = sim_nc_path(short_name)
    isfile(path) || return nothing, nothing
    try
        ds   = NCDataset(path)
        t    = ds["time"][:]
        data = ds[short_name][:] .* scale
        close(ds)
        mask = t .<= ref_t_avg_end
        return t[mask] ./ 3600.0, data[mask]
    catch e
        @warn "Could not load timeseries $short_name: $e"
        return nothing, nothing
    end
end

# Running mean + std over a centred ±WINDOW_H/2 window
WINDOW_H = 1.0
function running_stats(t_h, y; window_h = WINDOW_H)
    n = length(t_h)
    mu = similar(y, Float64)
    sig = similar(y, Float64)
    for i in 1:n
        mask = abs.(t_h .- t_h[i]) .<= window_h / 2
        pts = y[mask]
        mu[i] = mean(pts)
        sig[i] = length(pts) > 1 ? std(pts) : 0.0
    end
    return mu, sig
end

# ── Figure ─────────────────────────────────────────────────────────────────────
# 12-column grid so profile panels (×6) each span 2 cols and timeseries panels
# (×4) each span 3 cols, keeping widths equal within each row type.

const REF_COLOR = "#e07b39"   # orange – PyCLES u
const V_REF_COLOR = "#c0392b"   # red    – PyCLES v
const SIM_COLOR = "#2e6da4"   # blue   – ClimaAtmos u
const V_SIM_COLOR = "#27ae60"   # green  – ClimaAtmos v
const LWIDTH = 2.5

fig = CairoMakie.Figure(size = (1900, 1050))
fig[0, 1:12] = CairoMakie.Label(
    fig,
    "$case_name intercomparison  ·  last $(Int(LAST_N_HOURS))h average  ·  " *
    (has_sim ? "ClimaAtmos (blue) vs PyCLES (orange)" : "PyCLES reference only"),
    fontsize = 14, tellwidth = false,
)

z_km = ref_z ./ 1e3

function make_profile_ax(pos, xlabel, title)
    ax = CairoMakie.Axis(pos;
        ylabel = "z (km)", xlabel = xlabel, title = title,
        titlesize = 11, xlabelsize = 10, ylabelsize = 10)
    CairoMakie.ylims!(ax, 0, maximum(z_km))
    CairoMakie.hidespines!(ax, :t, :r)
    return ax
end

function make_ts_ax(pos, ylabel, title)
    ax = CairoMakie.Axis(pos;
        xlabel = "time (h)", ylabel = ylabel, title = title,
        titlesize = 11, xlabelsize = 10, ylabelsize = 10)
    CairoMakie.xlims!(ax, 0, ref_t_avg_end / 3600)   # clip to shared run end
    CairoMakie.hidespines!(ax, :t, :r)
    return ax
end

# Row 1: thermodynamic + water species profiles (2 cols each)
ax1 = make_profile_ax(fig[1, 1:2], "θ_l (K)", "Liq-ice pot. temperature")
ax2 = make_profile_ax(fig[1, 3:4], "q_t (g/kg)", "Total water")
ax3 = make_profile_ax(fig[1, 5:6], "q_l (g/kg)", "Liquid cloud water")
ax4 = make_profile_ax(fig[1, 7:8], "q_r (g/kg)", "Rain water")
ax5 = make_profile_ax(fig[1, 9:10], "q_i (g/kg)", "Cloud ice")
ax6 = make_profile_ax(fig[1, 11:12], "q_s (g/kg)", "Snow")

# Row 2: dynamics + turbulence profiles (2 cols each)
ax7 = make_profile_ax(fig[2, 1:2], "a_up (–)", "Updraft area fraction")
ax8 = make_profile_ax(fig[2, 3:4], "w_up (m/s)", "Updraft velocity")
ax9 = make_profile_ax(fig[2, 5:6], "TKE (m²/s²)", "Turbulent kinetic energy")
ax10 = make_profile_ax(fig[2, 7:8], "cloud fraction (–)", "Cloud fraction")
ax11 = make_profile_ax(fig[2, 9:10], "w (m/s)", "Grid-mean vert. velocity")
ax12 = make_profile_ax(fig[2, 11:12], "u, v (m/s)", "Horiz. velocity")

# Row 3: water path timeseries (3 cols each)
ax13 = make_ts_ax(fig[3, 1:3], "LWP (g/m²)", "Liquid water path")
ax14 = make_ts_ax(fig[3, 4:6], "RWP (g/m²)", "Rain water path")
ax15 = make_ts_ax(fig[3, 7:9], "IWP (g/m²)", "Ice water path")
ax16 = make_ts_ax(fig[3, 10:12], "SWP (g/m²)", "Snow water path")

# ── Drawing helpers ───────────────────────────────────────────────────────────
# Profile shading band (polygon, horizontal at each z level)
function profile_band!(ax, mu, sig, z, color; nsigma = 2, alpha = 0.25)
    isnothing(mu) && return
    lo = mu .- nsigma .* sig
    hi = mu .+ nsigma .* sig
    xs = vcat(lo, reverse(hi))
    ys = vcat(z, reverse(z))
    CairoMakie.poly!(ax, [CairoMakie.Point2f(x, y) for (x, y) in zip(xs, ys)];
        color = (color, alpha), strokewidth = 0)
end

# Timeseries shading band (standard orientation)
function ts_band!(ax, t, mu, sig, color; nsigma = 2, alpha = 0.25)
    CairoMakie.band!(ax, t, mu .- nsigma .* sig, mu .+ nsigma .* sig;
        color = (color, alpha))
end

function ref_line!(ax, x, z; label = "PyCLES", color = REF_COLOR, kw...)
    CairoMakie.lines!(ax, x, z; color, linewidth = LWIDTH, label, kw...)
end
function sim_line!(ax, x, z; label = "ClimaAtmos", color = SIM_COLOR, kw...)
    CairoMakie.lines!(ax, x, z; color, linewidth = LWIDTH, label, kw...)
end
function ref_ts_line!(ax, t, y; label = "PyCLES")
    CairoMakie.lines!(ax, t, y; color = REF_COLOR, linewidth = LWIDTH, label)
end
function sim_ts_line!(ax, t, y; label = "ClimaAtmos")
    CairoMakie.lines!(ax, t, y; color = SIM_COLOR, linewidth = LWIDTH, label)
end

# ── PyCLES profiles ───────────────────────────────────────────────────────────
# Core (always present across all cases)
for (ax, mu, sig) in [
    (ax1, ref_thetali, ref_thetali_s),
    (ax2, ref_qt, ref_qt_s),
    (ax3, ref_ql, ref_ql_s),
    (ax7, ref_aup, ref_aup_s),
    (ax8, ref_wup, ref_wup_s),
    (ax9, ref_tke, ref_tke_s),
]
    profile_band!(ax, mu, sig, z_km, REF_COLOR)
    ref_line!(ax, mu, z_km)
end

# Optional (may be absent — silently skip)
for (ax, mu, sig) in [
    (ax10, ref_cf, ref_cf_s),
    (ax4, ref_qr, ref_qr_s),
    (ax5, ref_qi, ref_qi_s),
    (ax6, ref_qs, ref_qs_s),
    (ax11, ref_wmean, ref_wmean_s),
]
    isnothing(mu) && continue
    profile_band!(ax, mu, sig, z_km, REF_COLOR)
    ref_line!(ax, mu, z_km)
end

# u & v: separate solid lines with distinct colors per component
if !isnothing(ref_umean)
    profile_band!(ax12, ref_umean, ref_umean_s, z_km, REF_COLOR)
    ref_line!(ax12, ref_umean, z_km; label = "u  PyCLES")
end
if !isnothing(ref_vmean)
    profile_band!(ax12, ref_vmean, ref_vmean_s, z_km, V_REF_COLOR)
    ref_line!(ax12, ref_vmean, z_km; label = "v  PyCLES", color = V_REF_COLOR)
end

# PyCLES timeseries — clipped to the shared run-end anchor (raw, no smoothing)
ref_ts_mask = ref_t .<= ref_t_avg_end
ref_t_h_plot = ref_t_h[ref_ts_mask]
ref_ts_line!(ax13, ref_t_h_plot, ref_lwp[ref_ts_mask])

if !isnothing(ref_rwp)
    ref_ts_line!(ax14, ref_t_h_plot, ref_rwp[ref_ts_mask])
end

if !isnothing(ref_iwp_ts)
    ref_ts_line!(ax15, ref_t_h_plot, ref_iwp_ts[ref_ts_mask])
end

if !isnothing(ref_swp_ts)
    ref_ts_line!(ax16, ref_t_h_plot, ref_swp_ts[ref_ts_mask])
end

# ── ClimaAtmos profiles ───────────────────────────────────────────────────────
if has_sim
    for (ax, sn, scale) in [
        (ax1, "thetaa", 1.0),
        (ax2, "hus", 1e3),
        (ax3, "clw", 1e3),
        (ax4, "husra", 1e3),
        (ax5, "cli", 1e3),
        (ax6, "hussn", 1e3),
        (ax7, "arup", 1.0),
        (ax8, "waup", 1.0),
        (ax9, "tke", 1.0),
        (ax11, "wa", 1.0),
    ]
        z_s, mu_s, sig_s = load_and_avg(sn; scale)
        isnothing(z_s) && continue
        profile_band!(ax, mu_s, sig_s, z_s ./ 1e3, SIM_COLOR)
        sim_line!(ax, mu_s, z_s ./ 1e3)
    end

    # Cloud fraction: ClimaAtmos outputs % → convert to 0–1 fraction
    _z_cl, _mu_cl, _sig_cl = load_and_avg("cl")
    if !isnothing(_z_cl)
        profile_band!(ax10, _mu_cl ./ 100, _sig_cl ./ 100, _z_cl ./ 1e3, SIM_COLOR)
        sim_line!(ax10, _mu_cl ./ 100, _z_cl ./ 1e3)
    end

    # u: blue solid, v: green solid
    # Subtract geostrophic offset so ClimaAtmos is in the same frame as PyCLES
    # (PyCLES stores wind anomaly; ClimaAtmos stores absolute wind).
    _z_u, _mu_u, _sig_u = load_and_avg("ua")
    if !isnothing(_z_u)
        _mu_u_plot  = _mu_u  .- uv_offset.u
        profile_band!(ax12, _mu_u_plot, _sig_u, _z_u ./ 1e3, SIM_COLOR)
        sim_line!(ax12, _mu_u_plot, _z_u ./ 1e3; label = "u  ClimaAtmos")
    end
    _z_v, _mu_v, _sig_v = load_and_avg("va")
    if !isnothing(_z_v)
        _mu_v_plot  = _mu_v  .- uv_offset.v
        profile_band!(ax12, _mu_v_plot, _sig_v, _z_v ./ 1e3, V_SIM_COLOR)
        sim_line!(ax12, _mu_v_plot, _z_v ./ 1e3; label = "v  ClimaAtmos", color = V_SIM_COLOR)
    end

    # Timeseries (raw, no smoothing — noise is intentionally visible)
    for (ax, sn) in [
        (ax13, "lwp"),
        (ax14, "rwp"),
        (ax15, "iwp"),
        (ax16, "swp"),
    ]
        t_s, y_s = load_ts(sn; scale = 1e3)
        isnothing(t_s) && continue
        sim_ts_line!(ax, t_s, y_s)
    end
end

# ── X-axis limits ─────────────────────────────────────────────────────────────
# Helper: set xlims from a dataset's mean ± 2σ with a margin
function set_xlims_from!(ax, mu, sig; nsigma = 2, margin_frac = 0.05)
    isnothing(mu) && return
    lo = minimum(mu .- nsigma .* sig)
    hi = maximum(mu .+ nsigma .* sig)
    m = max(margin_frac * (hi - lo), 1e-6)
    CairoMakie.xlims!(ax, lo - m, hi + m)
end

# Core axes: always PyCLES-limited
for (ax, mu, sig) in [
    (ax1, ref_thetali, ref_thetali_s),
    (ax2, ref_qt, ref_qt_s),
    (ax3, ref_ql, ref_ql_s),
    (ax7, ref_aup, ref_aup_s),
    (ax8, ref_wup, ref_wup_s),
    (ax9, ref_tke, ref_tke_s),
]
    set_xlims_from!(ax, mu, sig)
end

# Cloud fraction: PyCLES-limited when present, else ClimaAtmos
if !isnothing(ref_cf)
    set_xlims_from!(ax10, ref_cf, ref_cf_s)
end

# Optional axes (rain/ice/snow/w_mean): xlims from ClimaAtmos data, which is
# more meaningful than PyCLES zeros (warm cases, missing vars, etc.)
if has_sim
    for (ax, sn, scale) in [
        (ax4, "husra", 1e3),
        (ax5, "cli", 1e3),
        (ax6, "hussn", 1e3),
        (ax11, "wa", 1.0),
    ]
        _, mu_s, sig_s = load_and_avg(sn; scale)
        isnothing(mu_s) || set_xlims_from!(ax, mu_s, sig_s)
    end
end

# u/v axis: union of PyCLES and ClimaAtmos ranges so neither side is clipped.
# ClimaAtmos values are shifted by the geostrophic offset (same frame as PyCLES).
let uv_data = Tuple{Vector{Float64}, Vector{Float64}}[]
    # PyCLES reference (anomaly frame)
    !isnothing(ref_umean) && push!(uv_data, (ref_umean, ref_umean_s))
    !isnothing(ref_vmean) && push!(uv_data, (ref_vmean, ref_vmean_s))
    # ClimaAtmos (shifted into anomaly frame)
    if has_sim
        _, mu_u, sig_u = load_and_avg("ua")
        _, mu_v, sig_v = load_and_avg("va")
        !isnothing(mu_u) && push!(uv_data, (mu_u .- uv_offset.u, sig_u))
        !isnothing(mu_v) && push!(uv_data, (mu_v .- uv_offset.v, sig_v))
    end
    if !isempty(uv_data)
        all_lo = minimum(minimum(v .- 2 .* s) for (v, s) in uv_data)
        all_hi = maximum(maximum(v .+ 2 .* s) for (v, s) in uv_data)
        m = max(0.05 * (all_hi - all_lo), 1e-6)
        # Apply per-case hard floor (e.g. GABLS: start at -10 m/s)
        lo_floor = get(UV_XMIN, case_name, all_lo - m)
        CairoMakie.xlims!(ax12, min(all_lo - m, lo_floor), all_hi + m)
    end
end

# ── Legend ────────────────────────────────────────────────────────────────────
CairoMakie.axislegend(ax1; position = :rt, labelsize = 9)
CairoMakie.axislegend(ax12; position = :lt, labelsize = 8)

CairoMakie.save(out_path, fig)
println("Saved → $out_path")
