#####
##### Online ML tendency correction, see `MLTendencyCorrection`.
#####
##### Every `refresh_period` the model state is brought to the network's grid the
##### same way the training data was produced (pressure-level interpolation as in
##### the pressure-coordinate diagnostics, then bilinear remapping to the regular
##### latitude-longitude grid), the network runs on the root process, and its
##### correction is regridded back to every process and interpolated to model
##### levels. Between refreshes the cached correction is applied every step.
#####

import ClimaCore.Remapping as Remapping
import ClimaUtilities.Regridders

# Below-ground temperature extrapolation of the training preprocessing.
const ML_CORRECTION_LAPSE_RATE = 0.0065 # [K/m]
const ML_CORRECTION_R_D = 287.05 # [J/kg/K]
const ML_CORRECTION_GRAV = 9.80665 # [m/s²]

"""
    ml_correction_cache(Y, atmos, precomputed, start_date)

Build `p.ml_correction`, the cache read by `ml_correction_update!` and
`ml_correction_tendency!`. Returns an empty cache when the correction is disabled.

For an `MLTendencyCorrection` this loads the network (on the root process only),
builds the pressure interpolator on `precomputed.ᶜp` and the remapper to the
network's latitude-longitude grid, the regridder back to the model's horizontal
space, and the center fields holding the cached temperature and humidity
corrections [K/s], [1/s].
"""
ml_correction_cache(Y, atmos::AtmosModel, precomputed, start_date) =
    ml_correction_cache(Y, atmos.ml_correction, atmos, precomputed, start_date)

ml_correction_cache(Y, ::Nothing, atmos, precomputed, start_date) = (;)

function ml_correction_cache(
    Y,
    ml_correction::MLTendencyCorrection,
    atmos,
    precomputed,
    start_date,
)
    atmos.microphysics_model isa DryModel &&
        error("MLTendencyCorrection requires a moist model")
    FT = Spaces.undertype(axes(Y.c))
    context = ClimaComms.context(Y.c)
    ArrayType = ClimaComms.array_type(ClimaComms.device(context))
    is_root = ClimaComms.iamroot(context)

    grid = column_net_grid(ml_correction.path, FT)
    "t" in grid.targets && "q" in grid.targets || error(
        "ML correction network $(ml_correction.path) must predict `t` and `q`, \
        got $(grid.targets)",
    )
    net = is_root ? load_column_net(ml_correction.path, FT, ArrayType) : nothing
    levels = grid.pressure
    nlon, nlat, nlevels = length(grid.lon), length(grid.lat), length(levels)

    pressure_interpolator = Remapping.PressureInterpolator(precomputed.ᶜp, levels)
    pressure_space = Remapping.pressure_space(pressure_interpolator)
    # Temperature, eastward wind, northward wind, total specific humidity, pressure.
    ᴾfields = [fill(zero(FT), pressure_space) for _ in 1:5]
    hcoords = [Geometry.LatLongPoint(lat, lon) for lon in grid.lon, lat in grid.lat]
    zcoords = Geometry.PPoint.(sort(levels))
    remapper = Remapping.Remapper(
        pressure_space,
        hcoords,
        zcoords;
        horizontal_method = Remapping.BilinearRemapping(),
        buffer_length = length(ᴾfields),
    )
    # The multi-field `Remapping.interpolate!` takes a view of its destination on
    # every process, so it is allocated everywhere; only the root's is filled.
    remapped = ArrayType(zeros(FT, nlon, nlat, nlevels, length(ᴾfields)))
    regridder =
        Regridders.InterpolationsRegridder(Spaces.horizontal_space(axes(Y.c)))
    ncolumns = length(Fields.field2array(Fields.level(Y.c.ρ, 1)))

    ClimaComms.iamroot(context) && @info(
        "ML tendency correction enabled",
        path = ml_correction.path,
        members = length(net.members),
        gain = ml_correction.gain,
        refresh_period_hours = ml_correction.refresh_period / 3600,
        correct_temperature = ml_correction.correct_temperature,
        correct_humidity = ml_correction.correct_humidity,
    )

    return (;
        net,
        pressure_interpolator,
        ᴾfields,
        remapper,
        remapped,
        regridder,
        lon = Array(grid.lon),
        lat = Array(grid.lat),
        window = grid.window,
        log_levels = ArrayType(log.(levels)),
        level_correction = ArrayType(zeros(FT, nlevels, ncolumns)),
        ᶜu_east = similar(Y.c.ρ),
        ᶜv_north = similar(Y.c.ρ),
        ᶜq_tot = similar(Y.c.ρ),
        ᶜlog_p = similar(Y.c.ρ),
        ᶜdTdt = zero(Y.c.ρ),
        ᶜdq_totdt = zero(Y.c.ρ),
        start_date,
    )
end

"""
    ml_correction_update!(Y, p, t, ml_correction)

Recompute the cached ML correction `p.ml_correction.ᶜdTdt` [K/s] and
`p.ml_correction.ᶜdq_totdt` [1/s] from the current state. A no-op for `nothing`.

Steps, in order:

 1. Interpolate `T`, `u`, `v`, `q_tot`, and `p` to the network's pressure levels
    on model columns (`ClimaCore.Remapping.PressureInterpolator`, flat
    extrapolation), then bilinearly remap them to the network's grid; the result
    lands on the root process.
 2. On the root process, evaluate `ml_correction_on_grid`, the correction on the
    network's grid, and broadcast it to every process.
 3. On every process, regrid each level to the model's horizontal space and
    interpolate in `log p` to the model levels, with flat extrapolation outside
    the network's levels, and taper to zero between `p_full` and `p_zero`
    (`ml_correction_pressure_taper`). The humidity correction is limited so that
    it cannot remove more than the local `q_tot` within one `refresh_period`.

Called by `ml_correction_callback!` on the `refresh_period` cadence, which also
runs once at initialization.
"""
ml_correction_update!(Y, p, t, ::Nothing) = nothing

NVTX.@annotate function ml_correction_update!(
    Y,
    p,
    t,
    ml_correction::MLTendencyCorrection,
)
    cache = p.ml_correction
    FT = Spaces.undertype(axes(Y.c))
    (; ᶜT, ᶜp, ᶜu) = p.precomputed
    (; ᶜu_east, ᶜv_north, ᶜq_tot, ᴾfields, pressure_interpolator) = cache
    @. ᶜu_east = u_component(UVec(ᶜu))
    @. ᶜv_north = v_component(VVec(ᶜu))
    @. ᶜq_tot = specific(Y.c.ρq_tot, Y.c.ρ)

    Remapping.update!(pressure_interpolator)
    for (dest, field) in zip(ᴾfields, (ᶜT, ᶜu_east, ᶜv_north, ᶜq_tot, ᶜp))
        Remapping.interpolate_pressure!(dest, field, pressure_interpolator)
    end
    Remapping.interpolate!(cache.remapped, cache.remapper, ᴾfields)

    context = ClimaComms.context(Y.c)
    date = cache.start_date + Dates.Millisecond(round(Int, 1000 * time_to_seconds(t)))
    correction = if ClimaComms.iamroot(context)
        Array(ml_correction_on_grid(cache.net, cache.remapped, date, ml_correction))
    else
        nothing
    end
    correction = ClimaComms.bcast(context, correction)

    (; regridder, level_correction, log_levels, ᶜlog_p, lon, lat) = cache
    @. ᶜlog_p = log(pressure_interpolator.scratch_center_pressure_field)
    for (i, ᶜcorrection) in enumerate((cache.ᶜdTdt, cache.ᶜdq_totdt))
        for k in axes(correction, 3)
            level = Regridders.regrid(regridder, correction[:, :, k, i], (lon, lat))
            view(level_correction, k, :) .= vec(Fields.field2array(level))
        end
        interpolate1d!(
            Fields.field2array(ᶜcorrection),
            log_levels,
            Fields.field2array(ᶜlog_p),
            level_correction,
            Linear(),
            Flat();
            reverse = true,
        )
    end
    (; p_full, p_zero) = ml_correction
    @. cache.ᶜdTdt *= ml_correction_pressure_taper(ᶜp, p_full, p_zero)
    @. cache.ᶜdq_totdt *= ml_correction_pressure_taper(ᶜp, p_full, p_zero)
    neg_inv_refresh = -FT(1 / ml_correction.refresh_period)
    @. cache.ᶜdq_totdt = max(cache.ᶜdq_totdt, neg_inv_refresh * ᶜq_tot)
    return nothing
end

"""
    ml_correction_on_grid(net, remapped, date, ml_correction)

Return the ML correction on the network's grid, `(lon, lat, level, 2)` holding the
temperature [K/s] and total-specific-humidity [1/s] tendencies, as a device array.

`remapped` is the `(lon, lat, level, field)` output of the remapper, with levels
ordered top to surface and fields `T, u, v, q_tot, p`; the reference surface
pressure is `p` at the highest-pressure level, i.e. the lowest model level
pressure capped at that level, as in the training data. Below-ground temperatures
are replaced by a 6.5 K/km lapse-rate extrapolation from that level, as in the
training preprocessing. The cos-zenith phases are taken at the start, middle, and
end of a training-length window centered on the coming refresh interval.

The correction is `-gain` times the (optionally smoothed) ensemble-mean prediction,
limited to `±t_cap` and `±q_cap`, with disabled variables set to zero, and
masked (below-ground or terrain-edge) levels filled with the value of the lowest
valid level above them.
"""
function ml_correction_on_grid(net::ColumnNet, remapped, date, ml_correction)
    FT = eltype(remapped)
    nlon, nlat, nlevels, _ = size(remapped)
    x = reverse(view(remapped, :, :, :, 1:4); dims = 3)
    ps = remapped[:, :, end, 5]
    p_levels = level_pressure(net, ps)
    t = view(x, :, :, :, 1)
    t_ref = x[:, :, 1:1, 1]
    exponent = FT(ML_CORRECTION_R_D * ML_CORRECTION_LAPSE_RATE / ML_CORRECTION_GRAV)
    @. t = ifelse(p_levels > ps, t_ref * (p_levels / ps)^exponent, t)

    window = Dates.Millisecond(round(Int, 1000 * net.window))
    center = date + Dates.Millisecond(round(Int, 500 * ml_correction.refresh_period))
    phases = (center - window ÷ 2, center, center + window ÷ 2)
    cz_host = cat((cos_zenith_noaa(d, net.lat, net.lon) for d in phases)...; dims = 3)
    cz = copyto!(similar(ps, nlon, nlat, 3), cz_host)

    pred, m = column_net_predict(net, x, ps, cz)
    pred = gaussian_smooth(pred, m, ml_correction.smoothing_sigma)

    correction = similar(pred, nlon, nlat, nlevels, 2)
    it = findfirst(==("t"), net.targets)
    iq = findfirst(==("q"), net.targets)
    scale = -ml_correction.gain / FT(3600)
    dT = view(correction, :, :, :, 1)
    dq = view(correction, :, :, :, 2)
    if ml_correction.correct_temperature
        t_cap = ml_correction.t_cap
        dT .= clamp.(scale .* view(pred, :, :, :, it), -t_cap, t_cap)
    else
        fill!(dT, zero(FT))
    end
    if ml_correction.correct_humidity
        q_cap = ml_correction.q_cap
        dq .= clamp.(scale .* view(pred, :, :, :, iq), -q_cap, q_cap)
    else
        fill!(dq, zero(FT))
    end
    for k in (nlevels - 1):-1:1
        valid = view(m, :, :, k)
        c = view(correction, :, :, k, :)
        above = view(correction, :, :, k + 1, :)
        c .= ifelse.(valid, c, above)
    end
    return correction
end

"""
    ml_correction_pressure_taper(p, p_full, p_zero)

Vertical weight of the ML correction at pressure `p`: 1 for `p ≥ p_full`, 0 for
`p ≤ p_zero`, and a half cosine in between. It confines the correction to the
troposphere and lower stratosphere, away from the sponge layer and from the
levels above the network's top, where the interpolated correction is only
extrapolated.
"""
@inline function ml_correction_pressure_taper(p, p_full, p_zero)
    s = clamp((p - p_zero) / (p_full - p_zero), zero(p), one(p))
    return (1 - cospi(s)) / 2
end

"""
    ml_correction_ramp(t, ml_correction)

Return the strength `α(t) ∈ [0, 1]` of the ML correction at time `t` [s]: 0 before
`start_time`, then a linear ramp to 1 over `ramp_time` (immediately 1 when
`ramp_time` is 0).
"""
function ml_correction_ramp(t, ml_correction::MLTendencyCorrection)
    FT = typeof(ml_correction.gain)
    τ = FT(time_to_seconds(t)) - ml_correction.start_time
    τ < 0 && return FT(0)
    iszero(ml_correction.ramp_time) && return FT(1)
    return min(FT(1), τ / ml_correction.ramp_time)
end

"""
    ml_correction_tendency!(Yₜ, Y, p, t, ml_correction)

Add the cached ML correction, scaled by `ml_correction_ramp`, to the total-energy
and total-water tendencies through `apply_Tq_forcing!`. A no-op for `nothing` and
before `start_time`. Called from `additional_tendency!`; treated explicitly.
"""
ml_correction_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

function ml_correction_tendency!(Yₜ, Y, p, t, ml_correction::MLTendencyCorrection)
    α = ml_correction_ramp(t, ml_correction)
    iszero(α) && return nothing
    (; ᶜdTdt, ᶜdq_totdt) = p.ml_correction
    apply_Tq_forcing!(
        Yₜ,
        Y,
        p,
        @.(lazy(α * ᶜdTdt)),
        @.(lazy(α * ᶜdq_totdt)),
    )
    return nothing
end
