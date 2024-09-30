# Aerosols

`ClimaAtmos` implements two modes for ozone profiles. These are only relevant
for the radiation transfer, and only when RRTMGP is used.

### Idealized Ozone Profile

The `IdealizedOzone` type of ozone uses the `idealized_ozone` function to
compute an idealized ozone profile based on the work of `Wing2018`. This
profile is not time-varying.

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

### Prescribed Ozone Profile

The `PrescribedOzone` type of ozone uses CMIP6 forcing data to prescribe ozone
as read from files. A high-resolution, multi-year file is available in the
`ozone_concentrations` artifact. This file is not small, so you have to obtain
independently. Please, refer to `ClimaArtifacts` for more information. If the
file is not found, a low-resolution, single-year version is used. This is not
advised for production simulations.

We interpolate the data from file in time every time radiation is called. The
interpolation used is the `LinerPeriodFilling` from `ClimaUtilities`. This is a
linear period-aware interpolation that preserves the annual cycle.

### More docstrings

```@docs
ClimaAtmos.AbstractOzone
ClimaAtmos.IdealizedOzone
ClimaAtmos.PrescribedOzone
```
