# Aerosols

# Trace Gases
`ClimaAtmos` implements two modes for each ozone and carbon dioxide: one time varying and one time invariant. These are only relevant for the radiation transfer, and only when RRTMGP is used. All other atmospheric gases are held fixed with default values from RRTMPG that can be changed in the toml file.

### Time Invariant Ozone Profile

The time invariant type of ozone uses the `idealized_ozone` function to
compute an idealized ozone profile based on the work of `Wing2018`.
This option is default.

The `idealized_ozone` function returns the ozone concentration in volume mixing
ratio (VMR) at a given altitude `z`.
```@docs
ClimaAtmos.idealized_ozone
```

This function looks like
```@example
using CairoMakie
import ClimaAtmos

z = range(0, 60000, length=100)
ozone = ClimaAtmos.idealized_ozone.(z)

lines(ozone, z)
```

### Time Varying Ozone Profile

The time varying ozone profile uses CMIP6 forcing data to prescribe ozone
as read from files. A high-resolution, multi-year file is available in the
`ozone_concentrations` artifact. This file is not small, so you have to obtain
independently. Please, refer to `ClimaArtifacts` for more information. If the
file is not found, a low-resolution, single-year version is used. This is not
advised for production simulations. This option is enabled with by adding `"O3"`
to the `time_varying_gases` config argument list, ie: `time_varying_gases: ["O3"]`.

We interpolate the data from file in time every time radiation is called. The
interpolation used is the `LinerPeriodFilling` from `ClimaUtilities`. This is a
linear period-aware interpolation that preserves the annual cycle.

### Time Invariant CO2

By default, CO2 concentrations are set to 397.547 ppm. The number can be altered
by changing the `CO2_fixed_value` parameter in the toml file.

### Time Varying CO2

`ClimaAtmos` can prescribe CO2 concentration using data
from [Mauna Loa CO2 measurements](https://gml.noaa.gov/ccgg/trends/data.html).
This option is enabled with by adding `"CO2"` to the `time_varying_gases`
config argument list, ie: `time_varying_gases: ["CO2"]`.
