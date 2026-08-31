# Trace Gases

`ClimaAtmos` implements two modes each for ozone and carbon dioxide: one time varying and one time invariant. These are only relevant for radiative transfer, and only when RRTMGP is used. All other atmospheric gases are held fixed with default values from RRTMGP that can be changed in the toml file. See [Radiation](radiation.md) for how the gas concentrations reach the solver, and [Running with Radiation](radiation_howto.md) for the configuration keys.

## Time Invariant Ozone Profile

The time invariant type of ozone uses the `idealized_ozone` function to
compute an idealized ozone profile based on the work of [Wing2018](@cite).
This option is the default.

The `idealized_ozone` function returns the ozone concentration in volume mixing
ratio (VMR) at a given altitude `z`.

```@docs
ClimaAtmos.idealized_ozone
```

This function looks like

```@example
using CairoMakie
import ClimaAtmos

z = range(0, 60000, length = 100)
ozone = ClimaAtmos.idealized_ozone.(z)

fig = Figure()
ax = Axis(fig[1, 1]; xlabel = "Ozone (VMR)", ylabel = "Altitude (m)")
lines!(ax, ozone, z)
save("idealized_ozone.png", fig);
nothing # hide
```

![Idealized ozone profile](idealized_ozone.png)

## Time Varying Ozone Profile

The time varying ozone profile uses CMIP6 forcing data to prescribe ozone
as read from files. A high-resolution, multi-year file is available in the
`ozone_concentrations` artifact. This file is not small, so you have to obtain
it independently. Please refer to `ClimaArtifacts` for more information. If the
file is not found, a low-resolution, single-year version is used. This is not
advised for production simulations. This option is enabled by adding `"O3"`
to the `time_varying_trace_gases` config argument list, i.e.:
`time_varying_trace_gases: ["O3"]`.

We interpolate the file data in time whenever radiation is called. The
interpolation used is `LinearInterpolation` from `ClimaUtilities` (linear in
time between file snapshots).

## Time Invariant CO2 Profile

By default, CO2 concentrations are set to 397.547 ppm. This value can be changed
with the `CO2_fixed_value` parameter in the toml file.

## Time Varying CO2 Profile

`ClimaAtmos` can prescribe CO2 concentration using data
from [Mauna Loa CO2 measurements](https://gml.noaa.gov/ccgg/trends/data.html).
This option is enabled by adding `"CO2"` to the `time_varying_trace_gases`
config argument list, i.e.: `time_varying_trace_gases: ["CO2"]`.
