# Radiation

ClimaAtmos computes radiative fluxes with
[RRTMGP.jl](https://clima.github.io/RRTMGP.jl/stable/), a GPU-capable Julia
implementation of the RRTMGP correlated k-distribution model, wrapped in
`src/parameterized_tendencies/radiation/RRTMGPInterface.jl`. The radiative flux
divergence enters the total energy equation as an explicit tendency (see the
[governing equations](equations.md)), updated by a radiation callback at the
cadence set by `dt_rad` (default `6hours`).

!!! note "Under construction"

    A full explanation page (the RRTMGP coupling, cloud and aerosol optics,
    and the insolation models) is planned. The overview below is current.

## Radiation modes

The `rad` configuration argument selects the mode:

  - `gray`: gray radiation, an idealized one-band model.
  - `clearsky`: RRTMGP with gases but no cloud optics.
  - `allsky`: RRTMGP with interactive cloud optics.
  - `allskywithclear`: as `allsky`, plus clear-sky diagnostic fluxes.
  - `held_suarez`: the Held–Suarez temperature relaxation (no radiative
    transfer; applied every stage), plus idealized single-column forcings
    (`DYCOMS`, `TRMM_LBA`, `ISDAC`).

Clouds enter the all-sky modes either interactively from the model state or
prescribed from ERA5 data (`prescribe_clouds_in_radiation`). Trace-gas
concentrations are described on the [Trace Gases](trace_gases.md) page, and the
insolation at the top of the atmosphere is computed by
[Insolation.jl](https://clima.github.io/Insolation.jl/stable/) (see
[The CliMA Ecosystem](ecosystem.md)). The ocean surface albedo models are
described on the [Ocean Surface Albedo](surface_albedo.md) page.
