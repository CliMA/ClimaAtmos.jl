# Running with Radiation

Radiation is off by default. This page covers what to switch on for each kind of
run, how to choose the solve cadence, and which diagnostics tell you whether the
result is sensible. For what the modes actually compute, see
[Radiation](radiation.md).

## Choosing a mode

```yaml
rad: allskywithclear   # ~ (default, none), gray, clearsky, allsky,
                       # allskywithclear, held_suarez, DYCOMS, TRMM_LBA, ISDAC
insolation: idealized
dt_rad: "6hours"
```

Which mode fits depends on what the run is for.

| Purpose                                                             | Mode                          |
|:------------------------------------------------------------------- |:----------------------------- |
| A radiative sink with no radiative detail                           | `gray`                        |
| Dry benchmark of the circulation against a known target             | `held_suarez`                 |
| Isolating the circulation from cloud radiative feedback             | `clearsky`                    |
| Climate and weather runs                                            | `allskywithclear`             |
| Stratocumulus, Arctic stratocumulus, deep-convection single columns | `DYCOMS`, `ISDAC`, `TRMM_LBA` |

The production runs use `allskywithclear`, which adds the
clear-sky fluxes to the diagnostics. Prefer `allskywithclear`
unless the extra work for the clear-sky fluxes is a problem.

`held_suarez` replaces radiation by Newtonian relaxation of temperatures and
ignores `dt_rad`; the single-column modes are prescribed profiles evaluated
every stage. Both are described on the [Radiation](radiation.md) page.

## Choosing the cadence

`dt_rad` sets how often RRTMGP runs. Between calls the flux is frozen, so the
cadence has to be short compared with the timescale on which the heating rate
changes.

  - With `insolation: idealized` there is no diurnal cycle, so the only
    constraint is how fast the clouds and the thermal structure evolve. The
    6-hour default is comfortable, and longer is often fine for dry or
    clear-sky runs.
  - With `insolation: timevarying` the solar zenith angle changes through the
    day, and a 6-hour cadence resolves it with four samples. For boundary-layer
    or diurnal-cycle work, shorten it to an hour or less. Note that a cadence
    that does not divide the day evenly will alias the diurnal cycle.
  - In single-column cases the solve is one column, so a short cadence is
    unremarkable. In global all-sky runs, the radiation solve is a noticeable
    part of the wall-clock time, and halving `dt_rad` roughly doubles that
    part.

If `dt_rad` is shorter than or equal to `dt`, the callback fires once per step.

## Insolation

```yaml
insolation: timevarying   # idealized, timevarying, rcemipii, gcmdriven,
                          # externaldriventv, larcform1
```

`idealized` is the conventional choice for idealized aquaplanet climate runs:
annual-mean, latitude-dependent, no diurnal cycle. Use `timevarying` when the
diurnal or seasonal cycle matters, which includes essentially all AMIP-style and
site-comparison work. From a script, `TimeVaryingInsolation` also takes an
explicit site:

```julia
insolation = ClimaAtmos.TimeVaryingInsolation(; latitude = 36.6, longitude = -97.5)
```

which pins the insolation to the ARM Southern Great Plains site. The override
exists for single-column setups whose coordinates carry no latitude or
longitude, such as ARM VARANAL; without it, a flat-space column falls back to
the equator.

## Clouds, aerosols, and gases

All of these apply only to the RRTMGP modes.

```yaml
rad: allskywithclear
prescribe_clouds_in_radiation: false   # ERA5 climatology instead of model clouds
idealized_clouds: false                # two fixed cloud layers
idealized_h2o: false                   # prescribed RH instead of model vapor
aerosol_radiation: false               # aerosols in the shortwave optics
prescribed_aerosols: []                # required when aerosol_radiation is true
time_varying_trace_gases: []           # e.g. ["CO2", "O3"]
add_isothermal_boundary_layer: true
radiation_reset_rng_seed: false
```

Three of these break a feedback on purpose, so be deliberate about which one you
want:

  - `prescribe_clouds_in_radiation: true` lets the model form clouds that do
    not radiate, and radiates an ERA5 climatology instead. Use it to attribute
    a circulation change to something other than the cloud radiative feedback.
  - `idealized_clouds: true` freezes two cloud layers at construction. Use it
    for a controlled radiative forcing, not as an approximation to a cloud
    field.
  - `idealized_h2o: true` decouples the radiation from the model's moisture.

`aerosol_radiation: true` requires at least one species in
`prescribed_aerosols`, from `DST01`–`DST05`, `SSLT01`–`SSLT05`, `SO4`, `CB1`,
`CB2`, `OC1`, `OC2`. Enabling it with an empty list raises an error at model
construction rather than running without aerosols. Note that the prescribed
aerosols also feed the cloud droplet number closure behind the liquid effective
radius, so switching them on changes cloud optics even in the shortwave-only
sense.

For time-varying ozone and carbon dioxide, including which artifact supplies the
ozone data, see [Trace Gases](trace_gases.md).

Turn off `add_isothermal_boundary_layer` only if you want top-of-atmosphere
fluxes reported at the model top rather than at the top of the real atmosphere.
With it off, downwelling longwave is assumed to be zero at the model top (rather
than TOA), and the TOA longwave will be biased against observations.

## Surface

Radiation needs a surface temperature and an albedo. The temperature comes from
the surface model (see [Surface Conditions](surface_conditions.md)); the albedo
from `albedo_model`:

```yaml
albedo_model: RegressionFunctionAlbedo   # ConstantAlbedo (default),
                                         # RegressionFunctionAlbedo, CouplerAlbedo
```

`RegressionFunctionAlbedo` makes the ocean albedo depend on the solar zenith
angle and the near-surface wind, which matters for the shortwave budget at high
latitudes and low sun angles. See [Ocean Surface Albedo](surface_albedo.md).

## Diagnostics

Request the flux families you need; they are all off unless asked for.

```yaml
diagnostics:
  - short_name: [rsdt, rsut, rlut, rsds, rsus, rlds, rlus]
    reduction_time: average
    period: 1days
```

That set is the energy budget at the two boundaries: incoming shortwave at the
top, reflected shortwave and outgoing longwave at the top, and the four surface
components. In the `allskywithclear` mode, add the clear-sky counterparts to get
the cloud radiative effect:

```yaml
  - short_name: [rsutcs, rlutcs]
    reduction_time: average
    period: 1days
```

The shortwave and longwave cloud radiative effects at the top of the atmosphere
are then `rsutcs - rsut` and `rlutcs - rlut`.

For diagnosing the cloud optics rather than the fluxes, `reffclw` and `reffcli`
give the effective radii the radiation used, and `clt` and `cltl` the cloud
cover the shortwave and longwave McICA sampling produced. Requesting a
diagnostic that the active mode does not compute raises an error naming the
variable and the mode.

## Worked examples

The tested configurations are the best starting point.
[`amip_target.yml`](https://github.com/CliMA/ClimaAtmos.jl/blob/main/config/longrun_configs/amip_target.yml)
is the most complete radiation setup in the repository, and a good one to read
in full: hourly `allskywithclear` with time-varying insolation, time-varying CO₂
and ozone, and all fifteen prescribed aerosol species.

```yaml
rad: "allskywithclear"
dt_rad: "1hours"
insolation: "timevarying"
time_varying_trace_gases: ["CO2", "O3"]
aerosol_radiation: true
prescribed_aerosols: ["CB1", "CB2", "DST01", "DST02", "DST03", "DST04", "DST05",
                      "OC1", "OC2", "SO4", "SSLT01", "SSLT02", "SSLT03",
                      "SSLT04", "SSLT05"]
```

At the other end,
[`single_column_radiative_equilibrium_gray.yml`](https://github.com/CliMA/ClimaAtmos.jl/blob/main/config/model_configs/single_column_radiative_equilibrium_gray.yml)
is a single column relaxing to gray radiative equilibrium.

See [Running Global Simulations](global_simulations.md) for the surrounding
configuration, and [Creating Custom Configurations](configuration.md) for
combining configuration files.

## Troubleshooting

**The top-of-atmosphere budget does not close.** Check
`add_isothermal_boundary_layer` first, then whether the run has reached
equilibrium; a spinning-up model has a real imbalance. Compare `rsdt - rsut`
with `rlut` in a long time average.

**All-sky fluxes are noisy.** That is the McICA sampling, not a bug. It averages
out over time and area. If you need bitwise reproducibility for a restart test,
set `radiation_reset_rng_seed: true`, but leave it off otherwise, because it
correlates the noise in time.

**Shortwave fluxes are zero everywhere.** Check the insolation: `larcform1` is
perpetual polar night by construction, and `idealized` at high latitude in a
short run can be close to zero.

**A radiation diagnostic errors out.** The clear-sky (`*cs`) diagnostics need
`rad: allskywithclear`; the cloud diagnostics need one of the all-sky modes;
none of them exist under `held_suarez` or the idealized single-column modes. The
error names both the variable and the active mode.

**Radiation dominates the run time.** Lengthen `dt_rad`, or drop to `clearsky`
if cloud radiative effects are not part of the question. Dropping from
`allskywithclear` to `allsky` removes the clear-sky fluxes and the work that
produces them, but then the cloud radiative effect is no longer available.
