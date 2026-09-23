#####
##### Global ERA5 relaxation (pre-forecast nudging), see `ERA5Relaxation`.
#####
##### The model is relaxed toward ERA5 over a pre-forecast window so the
##### atmosphere can adjust to the model's own dynamics and physics before a free
##### forecast begins. Temperature and total specific humidity are relaxed through
##### the independent pair `(T, q_tot)` and reconstructed into `ρe_tot`/`ρq_tot`
##### with `apply_Tq_forcing!`; the horizontal wind is relaxed with `nudge_uv!`.
#####

import ClimaCore.Fields as Fields
import ClimaCore.Spaces as Spaces
import ClimaUtilities.TimeVaryingInputs:
    TimeVaryingInput, LinearInterpolation, evaluate!
import Interpolations as Intp

"""
    era5_relaxation_target_levels(Y; dz = 300)

Return the altitude levels [m] the ERA5 relaxation targets are interpolated onto,
spanning the model column from the surface to its top with spacing `dz`.
"""
function era5_relaxation_target_levels(Y; dz = 300)
    z_arr = Array(Fields.field2array(Fields.coordinate_field(Y.c).z))
    z_top = round(maximum(z_arr))
    return collect(0.0:float(dz):z_top)
end

"""
    era5_relaxation_cache(Y, atmos, start_date)

Build `p.era5_relaxation`, the cache read by `era5_relaxation_tendency!`.

For an [`ERA5Relaxation`](@ref) this resolves (generating if necessary) the
combined `lon`-`lat`-`z`-`time` ERA5 file with `era5_relaxation_data_path`, builds
one regridding `TimeVaryingInput` per relaxed variable (`t`, `q`, `u`, `v`)
anchored at `start_date`, and allocates the center fields those inputs are
evaluated into. Returns an empty cache when relaxation is disabled.
"""
era5_relaxation_cache(Y, atmos::AtmosModel, start_date) =
    era5_relaxation_cache(Y, atmos.era5_relaxation, start_date)

era5_relaxation_cache(Y, ::Nothing, start_date) = (;)

function era5_relaxation_cache(Y, era5_relaxation::ERA5Relaxation, start_date)
    FT = Spaces.undertype(axes(Y.c))
    context = ClimaComms.context(Y.c)

    target_levels = era5_relaxation_target_levels(Y)
    file_path = era5_relaxation_data_path(
        era5_relaxation.data_dir,
        start_date,
        era5_relaxation.window,
        target_levels,
        FT;
        context,
    )

    ClimaComms.iamroot(context) && @info(
        "ERA5 relaxation enabled",
        start_date,
        window_hours = era5_relaxation.window / 3600,
        τ_temperature_hours = era5_relaxation.τ_temperature / 3600,
        τ_humidity_hours = era5_relaxation.τ_humidity / 3600,
        τ_wind_hours = era5_relaxation.τ_wind / 3600,
        taper_begin_hours = era5_relaxation.taper_begin / 3600,
        taper_end_hours = era5_relaxation.taper_end / 3600,
        file_path,
    )

    extrapolation_bc = (Intp.Periodic(), Intp.Flat(), Intp.Flat())
    input(name) = TimeVaryingInput(
        file_path,
        name,
        axes(Y.c);
        reference_date = start_date,
        regridder_type = :InterpolationsRegridder,
        regridder_kwargs = (; extrapolation_bc),
        method = LinearInterpolation(),
    )

    return (;
        ᶜT_era5 = similar(Y.c.ρ),
        ᶜq_era5 = similar(Y.c.ρ),
        ᶜu_era5 = similar(Y.c.ρ),
        ᶜv_era5 = similar(Y.c.ρ),
        input_T = input("t"),
        input_q = input("q"),
        input_u = input("u"),
        input_v = input("v"),
    )
end

"""
    era5_relaxation_taper(t, era5_relaxation)

Return the relaxation taper `α(t) ∈ [0, 1]` at model time `t` [s].

`α = 1` until `taper_begin`, ramps linearly to 0 between `taper_begin` and
`taper_end`, and is 0 afterward (so the free forecast, `t ≥ taper_end`, is
untouched).
"""
function era5_relaxation_taper(t, era5_relaxation::ERA5Relaxation)
    FT = typeof(era5_relaxation.window)
    tf = FT(t)
    t_begin = era5_relaxation.taper_begin
    t_end = era5_relaxation.taper_end
    tf <= t_begin && return FT(1)
    tf >= t_end && return FT(0)
    return (t_end - tf) / (t_end - t_begin)
end

"""
    era5_relaxation_tendency!(Yₜ, Y, p, t, era5_relaxation)

Add the ERA5 relaxation tendency at model time `t`, dispatching on
`era5_relaxation` (a no-op for `nothing`).

The taper `α(t)` scales all rates; once it reaches 0 (past the window) the
tendency returns immediately, leaving the free forecast untouched. Otherwise the
ERA5 targets are refreshed from their `TimeVaryingInput`s and the tendency
`-α (X - X_ERA5)/τ_X` is applied: temperature and total specific humidity through
the shared `(dT, dq)` buffers and `apply_Tq_forcing!`, the horizontal wind through
`nudge_uv!`. Called from `additional_tendency!`; treated explicitly.
"""
era5_relaxation_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

function era5_relaxation_tendency!(Yₜ, Y, p, t, era5_relaxation::ERA5Relaxation)
    α = era5_relaxation_taper(t, era5_relaxation)
    α == 0 && return nothing

    (; ᶜT_era5, ᶜq_era5, ᶜu_era5, ᶜv_era5) = p.era5_relaxation
    (; input_T, input_q, input_u, input_v) = p.era5_relaxation
    evaluate!(ᶜT_era5, input_T, t)
    evaluate!(ᶜq_era5, input_q, t)
    evaluate!(ᶜu_era5, input_u, t)
    evaluate!(ᶜv_era5, input_v, t)

    FT = eltype(Y.c.ρ)
    inv_τ_T = FT(α / era5_relaxation.τ_temperature)
    inv_τ_q = FT(α / era5_relaxation.τ_humidity)
    inv_τ_uv = FT(α / era5_relaxation.τ_wind)

    (; ᶜT) = p.precomputed
    ᶜdTdt = p.scratch.ᶜtemp_scalar
    ᶜdqtdt = p.scratch.ᶜtemp_scalar_2
    @. ᶜdTdt = -(ᶜT - ᶜT_era5) * inv_τ_T
    @. ᶜdqtdt = -(specific(Y.c.ρq_tot, Y.c.ρ) - ᶜq_era5) * inv_τ_q
    apply_Tq_forcing!(Yₜ, Y, p, ᶜdTdt, ᶜdqtdt)

    nudge_uv!(Yₜ, Y, p, ᶜu_era5, ᶜv_era5, inv_τ_uv)
    return nothing
end
