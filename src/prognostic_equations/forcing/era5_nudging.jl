#####
##### Pre-forecast analysis nudging toward ERA5 for global (non-column) runs.
#####
##### Standard analysis nudging (cf. MPAS-A/WRF): the resolved large-scale state
##### (u, v, T, qₜ) is relaxed toward a sequence of ERA5 snapshots over a short
##### window preceding the forecast start, while the rest of the model (density,
##### surface pressure, w, clouds, TKE, EDMF, precipitation, land coupling) runs
##### normally and equilibrates. The targets are the same per-time, model-z-level
##### ERA5 files used for `WeatherModel` initial conditions. See `ERA5Nudging`.
#####

import Dates
import ClimaComms
import ClimaCore.Fields as Fields
import ClimaCore.Spaces as Spaces
import ClimaUtilities.SpaceVaryingInputs
import Interpolations as Intp

"""
    era5_nudging_snapshot_files(file_dir, start_date, window, interval)

Discover the processed ERA5 snapshot files used as nudging targets.

Candidate snapshot times are generated at `interval` spacing (in seconds) from
`start_date` up to and including `start_date + window`. For each candidate the
expected file `era5_init_processed_internal_YYYYMMDD_HHMM.nc` is looked up in
`file_dir`; only existing files are kept.

Returns `(times, paths)`, where `times` are the snapshot times in seconds
relative to `start_date` and `paths` the corresponding file paths, both sorted
in increasing time.
"""
function era5_nudging_snapshot_files(file_dir, start_date, window, interval)
    isdir(file_dir) ||
        error("ERA5Nudging: `era5_initial_condition_dir` $(file_dir) is not a directory")

    n = Int(floor(window / interval))
    times = Float64[]
    paths = String[]
    for k in 0:n
        offset_seconds = k * interval
        dt = start_date + Dates.Second(round(Int, offset_seconds))
        fname =
            "era5_init_processed_internal_" *
            Dates.format(dt, "yyyymmdd_HHMM") *
            ".nc"
        fpath = joinpath(file_dir, fname)
        if isfile(fpath)
            push!(times, offset_seconds)
            push!(paths, fpath)
        else
            @warn "ERA5Nudging: expected snapshot file not found, skipping" fpath
        end
    end

    length(paths) >= 2 || error(
        "ERA5Nudging requires at least two ERA5 snapshot files in $(file_dir) " *
        "covering [start_date, start_date + window]; found $(length(paths)).",
    )
    return times, paths
end

"""
    era5_nudging_weight(z, surface_weight, ramp_bottom, ramp_top, z_taper_start, z_top)

Height-dependent nudging weight `W(z) ∈ [0, 1]`.

`W` is the product of a near-surface taper and an upper (sponge) taper:

  - surface taper: `surface_weight` at/below `ramp_bottom`, cosine ramp to `1`
    at `ramp_top`, then `1` above (keeps the boundary layer relatively free so
    the land model and surface layer can equilibrate);
  - upper taper: `1` at/below `z_taper_start`, cosine ramp to `0` at `z_top`
    (so nudging does not fight the model-top sponge). Passing
    `z_taper_start == z_top` disables the upper taper.
"""
function era5_nudging_weight(
    z,
    surface_weight,
    ramp_bottom,
    ramp_top,
    z_taper_start,
    z_top,
)
    w_sfc =
        z <= ramp_bottom ? surface_weight :
        (
            z >= ramp_top ? one(z) :
            surface_weight +
            (one(z) - surface_weight) *
            (1 - cos(π * (z - ramp_bottom) / (ramp_top - ramp_bottom))) / 2
        )
    w_top =
        z <= z_taper_start ? one(z) :
        (
            z >= z_top ? zero(z) :
            (1 + cos(π * (z - z_taper_start) / (z_top - z_taper_start))) / 2
        )
    return w_sfc * w_top
end

"""
    era5_nudging_cache(Y, external_forcing::ERA5Nudging, start_date, z_taper_start)

Build the cache for pre-forecast ERA5 nudging: regrid every discovered ERA5
snapshot (`t`, `q`, `u`, `v`) onto the model center grid once, store them, and
materialize the height-dependent inverse relaxation timescales
`W(z) / τ_uvT` (used for `u`, `v`, `T`) and `W(z) / τ_q` (used for `qₜ`). The
per-step target fields are blended from the stored snapshots in the tendency.
"""
function era5_nudging_cache(
    Y,
    external_forcing::ERA5Nudging,
    start_date,
    z_taper_start,
)
    FT = Spaces.undertype(axes(Y.c))
    center_space = axes(Y.c)

    (; file_dir, window, interval) = external_forcing

    times, paths =
        era5_nudging_snapshot_files(file_dir, start_date, window, interval)
    snapshot_times = FT.(times)

    @info "ERA5Nudging: relaxing (u, v, T) with τ = $(external_forcing.τ_uvT) s " *
          "and qₜ with τ = $(external_forcing.τ_q) s toward $(length(paths)) " *
          "ERA5 snapshot(s) over the first $(window) s of the run."

    # Same regridding path as the ERA5 initial condition (`overwrite_from_file!`)
    regridder_type = :InterpolationsRegridder
    extrapolation_bc = (Intp.Periodic(), Intp.Flat(), Intp.Flat())
    interpolation_method = Intp.Linear()
    svi_kwargs = (;
        regridder_type,
        regridder_kwargs = (; extrapolation_bc, interpolation_method),
    )
    regrid(file_path, varname) = SpaceVaryingInputs.SpaceVaryingInput(
        file_path,
        varname,
        center_space;
        svi_kwargs...,
    )

    T_snapshots = [regrid(path, "t") for path in paths]
    qt_snapshots = [regrid(path, "q") for path in paths]
    u_snapshots = [regrid(path, "u") for path in paths]
    v_snapshots = [regrid(path, "v") for path in paths]

    # Domain top (face) for the upper taper.
    z_top = maximum(Fields.coordinate_field(Y.f).z)

    ᶜz = Fields.coordinate_field(Y.c).z
    ᶜinv_τ_uvT = similar(ᶜz, FT)
    ᶜinv_τ_q = similar(ᶜz, FT)
    @. ᶜinv_τ_uvT =
        era5_nudging_weight(
            ᶜz,
            external_forcing.surface_weight,
            external_forcing.ramp_bottom,
            external_forcing.ramp_top,
            z_taper_start,
            z_top,
        ) / external_forcing.τ_uvT
    @. ᶜinv_τ_q =
        era5_nudging_weight(
            ᶜz,
            external_forcing.surface_weight,
            external_forcing.ramp_bottom,
            external_forcing.ramp_top,
            z_taper_start,
            z_top,
        ) / external_forcing.τ_q

    return (;
        snapshot_times,
        t_last = snapshot_times[end],
        T_snapshots,
        qt_snapshots,
        u_snapshots,
        v_snapshots,
        ᶜT_nudge = similar(Y.c, FT),
        ᶜqt_nudge = similar(Y.c, FT),
        ᶜu_nudge = similar(Y.c, FT),
        ᶜv_nudge = similar(Y.c, FT),
        ᶜinv_τ_uvT,
        ᶜinv_τ_q,
    )
end

# Locate the bracketing snapshot indices `(i, i+1)` and the linear-in-time
# weight `α ∈ [0, 1]` such that the target at `t_sec` is
# `(1 - α) * snapshot[i] + α * snapshot[i+1]`. Runs on the host (few snapshots).
function _era5_bracket(snapshot_times, t_sec)
    n = length(snapshot_times)
    t_sec <= snapshot_times[1] && return (1, 1, zero(t_sec))
    t_sec >= snapshot_times[n] && return (n, n, zero(t_sec))
    i = 1
    while i < n && snapshot_times[i + 1] < t_sec
        i += 1
    end
    t0 = snapshot_times[i]
    t1 = snapshot_times[i + 1]
    α = (t_sec - t0) / (t1 - t0)
    return (i, i + 1, α)
end

# Blend the two bracketing snapshots into the target fields at time `t_sec`.
function blend_era5_snapshots!(cache, t_sec)
    i, j, α_raw = _era5_bracket(cache.snapshot_times, t_sec)
    α = eltype(cache.ᶜT_nudge)(α_raw)
    @. cache.ᶜT_nudge =
        (1 - α) * cache.T_snapshots[i] + α * cache.T_snapshots[j]
    @. cache.ᶜqt_nudge =
        (1 - α) * cache.qt_snapshots[i] + α * cache.qt_snapshots[j]
    @. cache.ᶜu_nudge =
        (1 - α) * cache.u_snapshots[i] + α * cache.u_snapshots[j]
    @. cache.ᶜv_nudge =
        (1 - α) * cache.v_snapshots[i] + α * cache.v_snapshots[j]
    return nothing
end

"""
    external_forcing_tendency!(Yₜ, Y, p, t, ::ERA5Nudging)

Apply pre-forecast ERA5 nudging. While the model time `t` is within the nudging
window (`t ≤ t_last`, the last snapshot / forecast start), relax `u`, `v`, `T`,
and `qₜ` toward the time-interpolated ERA5 target with the height-dependent
inverse timescales built in `era5_nudging_cache`. Temperature and humidity
tendencies are converted to `ρe_tot`/`ρq_tot` (density and surface pressure are
left untouched), and momentum is relaxed as a velocity tendency on `Y.c.uₕ`.
Past the window the nudging tendency is simply not applied.
"""
function external_forcing_tendency!(Yₜ, Y, p, t, ::ERA5Nudging)
    cache = p.external_forcing
    t_sec = float(t)

    # Nudging window closed (forecast phase): apply no nudging tendency.
    t_sec > cache.t_last && return nothing

    blend_era5_snapshots!(cache, t_sec)

    (; ᶜT) = p.precomputed
    ᶜdTdt = p.scratch.ᶜtemp_scalar
    ᶜdqtdt = p.scratch.ᶜtemp_scalar_2
    @. ᶜdTdt = -(ᶜT - cache.ᶜT_nudge) * cache.ᶜinv_τ_uvT
    @. ᶜdqtdt =
        -(specific(Y.c.ρq_tot, Y.c.ρ) - cache.ᶜqt_nudge) * cache.ᶜinv_τ_q
    apply_Tq_forcing!(Yₜ, Y, p, ᶜdTdt, ᶜdqtdt)

    nudge_uv!(Yₜ, Y, p, cache.ᶜu_nudge, cache.ᶜv_nudge, cache.ᶜinv_τ_uvT)
    return nothing
end
