# ERA5 Relaxation (Pre-forecast Spin-up)

Initializing a forecast directly from a reanalysis at ``t_0`` forces the model
onto a state that is not balanced against its own dynamics and physics, which
produces an initialization shock and early drift. ERA5 relaxation reduces that
shock by starting the run *earlier*, at ``t_0 - \\Delta``, and relaxing
(nudging) the model toward ERA5 over the pre-forecast window ``[t_0 - \\Delta,
t_0]`` before letting it run free.

The model is still initialized from ERA5 model levels in the usual way (see
[`WeatherModel`](@ref ClimaAtmos.WeatherModel)); the relaxation is an *additional*
tendency applied during the window only.

## The relaxation tendency

For each relaxed variable ``X \\in \\{T, q_{tot}, u, v\\}`` a Newtonian
relaxation tendency is added,

```math
\\left. \\frac{\\partial X}{\\partial t} \\right|_{\\mathrm{relax}}
  = -\\,\\alpha(t)\\, \\frac{X - X_{\\mathrm{ERA5}}(t)}{\\tau_X},
```

where

  - ``X_{\\mathrm{ERA5}}(t)`` is the ERA5 field regridded to the model grid and
    linearly interpolated in time between the (6-hourly) ERA5 snapshots,
  - ``\\tau_X`` is the per-variable relaxation timescale, and
  - ``\\alpha(t) \\in [0, 1]`` is a taper that ramps the relaxation off toward the
    end of the window.

Temperature and total specific humidity are relaxed through the physically
independent pair ``(T, q_{tot})`` and then reconstructed into the prognostic
`ρe_tot` and `ρq_tot` with the model's own thermodynamic routines
(`apply_Tq_forcing!`), so the redundant thermodynamic variables (density,
pressure, energy) are never nudged independently. The horizontal wind is relaxed
directly on `Y.c.uₕ` (`nudge_uv!`). Vertical velocity and surface pressure are not
nudged.

With `t = 0` at initialization (the coupler sets this to ``t_0 - \\Delta``),

```math
\\alpha(t) = \\begin{cases}
1, & t \\le t_{\\mathrm{begin}} \\\\
\\dfrac{t_{\\mathrm{end}} - t}{t_{\\mathrm{end}} - t_{\\mathrm{begin}}},
  & t_{\\mathrm{begin}} < t < t_{\\mathrm{end}} \\\\
0, & t \\ge t_{\\mathrm{end}}
\\end{cases}
```

so the free forecast (``t \\ge t_{\\mathrm{end}}``, which should be the window
length) is left completely untouched.

## Configuration

The scheme is off by default. Enable it by pointing `era5_relaxation` at the
directory of 6-hourly ERA5 pressure-level files
(`era5_pressure_levels_<yyyymmdd>_<HHMM>.nc`, containing `z`, `t`, `q`, `u`,
`v`). All tuning is exposed through configuration keys (in hours and fractions),
so the timescales and window can be changed from the coupler without touching
ClimaParams:

| Key | Meaning | Default |
|-----|---------|---------|
| `era5_relaxation` | Directory of ERA5 pressure-level files (or `~` to disable) | `~` |
| `era5_relaxation_window_hours` | Window length ``\\Delta`` from `t = 0` | `12.0` |
| `era5_relaxation_tau_temperature_hours` | ``\\tau_T`` | `6.0` |
| `era5_relaxation_tau_humidity_hours` | ``\\tau_{q}`` | `6.0` |
| `era5_relaxation_tau_wind_hours` | ``\\tau_u = \\tau_v`` | `6.0` |
| `era5_relaxation_taper_begin_frac` | Fraction of window where ``\\alpha`` starts decreasing | `0.5` |
| `era5_relaxation_taper_end_frac` | Fraction of window where ``\\alpha`` reaches 0 | `1.0` |

The run must be initialized at ``t_0 - \\Delta`` (the coupler is responsible for
choosing `start_date` and, if it wishes, the extended-back ERA5 initial
condition). A moist model is required, since humidity is relaxed.

## Implementation

The runtime consumes the ERA5 targets exactly like the prescribed
aerosol/ozone inputs: through a `TimeVaryingInput` that regrids a global
`lon`-`lat`-`z`-`time` NetCDF file onto the model grid and interpolates it in
time. Because the raw ERA5 data is a directory of *pressure-level* snapshots, a
one-time preprocessing step interpolates each snapshot spanning the window from
pressure levels to altitude levels (using the geopotential for heights, as in the
weather-model initial condition) and concatenates them along a `time` dimension.
The combined file is cached next to the raw data and generated on the root rank
only.

```
Preprocessing (once per window, root rank):
  era5_relaxation_data_path
    └─ generate_era5_relaxation_file          → combined lon-lat-z-time NetCDF (t, q, u, v)

Cache build (era5_relaxation_cache):
  TimeVaryingInput per variable (regrid + time-interp) → ᶜT_era5, ᶜq_era5, ᶜu_era5, ᶜv_era5

Every step (era5_relaxation_tendency!, from additional_tendency!):
  α(t) taper; skip once α = 0
  -α (T - ᶜT_era5)/τ_T, -α (q_tot - ᶜq_era5)/τ_q  → ρe_tot, ρq_tot via apply_Tq_forcing!
  -α (uₕ - uₕ_era5)/τ_uv                            → Y.c.uₕ via nudge_uv!
```
