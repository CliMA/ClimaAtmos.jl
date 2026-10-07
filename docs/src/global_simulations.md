# Running Global Simulations

A global run in ClimaAtmos is a cubed-sphere simulation: an aquaplanet, a
dry or moist baroclinic wave, a Held–Suarez integration, or an AMIP-style
configuration with Earth topography and prescribed sea surface temperatures.
This guide covers the choices a global run requires — resolution, timestep,
output, and running at scale — and points to the pages that treat each in
detail.

## Starting from a script

The default `AtmosSimulation` is already global: a cubed-sphere grid with a
decaying temperature profile (see [Your First Simulation](first_simulation.md)).
The quickest physically interesting global run is the aquaplanet preset, which
pairs a moist model with an aquaplanet surface:

```julia
import ClimaAtmos as CA

simulation = CA.Presets.aquaplanet(Float32; t_end = "1days")
CA.solve_atmos!(simulation)
```

Customize it through the [grid](grids.md) (`SphereGrid`), the
[setup](setups.md), and the `AtmosModel` components.

## Choosing a horizontal resolution

Horizontal resolution is set by `h_elem`, the number of spectral elements along
each edge of the cubed sphere, together with the polynomial degree `nh_poly`
(3 by default, giving four Gauss–Lobatto–Legendre nodes per element per
direction). Because one panel edge of the cubed sphere spans a quarter of a
great circle (an arc length of ``\pi a / 2`` for a planet of radius ``a``), the
average spacing between nodes is roughly

```math
\Delta x \approx \frac{\pi a}{2 \, (h_{elem} \times n_{h,poly})} .
```

For Earth's radius (``a \approx 6{,}371 \, \mathrm{km}``, so that
``\pi a / 2 \approx 10{,}000 \, \mathrm{km}``), this becomes

```math
\Delta x \approx \frac{10{,}000 \, \mathrm{km}}{h_{elem} \times n_{h,poly}} ,
```

giving about 550 km at `h_elem: 6`, 206 km at 16, 110 km at 30, and 55 km
at 60.

Vertical resolution is set by `z_elem` layers between the surface and a model
top at `z_max`. With `z_stretch: true`, the default, layers are stretched with a
hyperbolic tangent profile anchored on `dz_bottom`, the thickness of the lowest
layer, so that the boundary layer has high vertical resolution that becomes
coarser aloft.

Start from one of the numerical configurations in `config/common_configs/`,
which pair a resolution with a timestep and a matching set of numerics:

| Common configuration           | ``\Delta x`` | Levels | Model top | Sponges              |
|:------------------------------ |:------------ |:------ |:--------- |:-------------------- |
| `numerics_sphere_he6ze10.yml`  | 550 km       | 10     | 30 km     | none                 |
| `numerics_sphere_he6ze31.yml`  | 550 km       | 31     | 60 km     | Rayleigh and viscous |
| `numerics_sphere_he16ze63.yml` | 206 km       | 63     | 60 km     | Rayleigh and viscous |
| `numerics_sphere_he30ze43.yml` | 110 km       | 43     | 30 km     | none                 |
| `numerics_sphere_he30ze63.yml` | 110 km       | 63     | 60 km     | Rayleigh and viscous |

Model top and sponges are chosen together. A 60 km top puts the rigid lid in
the mesosphere, where upward-propagating waves have to be absorbed before they
reflect, so those configurations enable both sponges and treat vertical
diffusion implicitly; see [Model Top and Sponge Layer](sponge.md). See
[Creating custom configurations](configuration.md) for how to combine a common
configuration with a model configuration.

## Choosing a timestep

The tested timesteps scale roughly linearly with horizontal resolution: 400 s at
550 km, 120 s at 206 km, 90 s at 110 km, and 30 s at 55 km. Acoustic and
gravity-wave propagation in the vertical is handled implicitly, so the vertical
grid spacing does not enter the stability limit. What limits the timestep is
the horizontal acoustic Courant–Friedrichs–Lewy (CFL) bound, since the
horizontal terms are explicit. See [Implicit Solver](implicit_solver.md) for
the split and [Discretization and Operators](discretization.md) for which terms
fall on each side of it.

Two related keys matter at these resolutions:

  - `ode_algo` selects the implicit–explicit scheme, `ARS343` by default. The
    PROPHET production runs use `ARS222`.
  - `dt_rad` sets how often radiation is called, 6 hours by default and 1 hour
    in the production configurations. Radiation takes more run time than any
    other parameterization in a global run, so this interval needs to be set
    carefully; see [Running with Radiation](radiation_howto.md).

An explicit parameterization with its own stability limit can also limit the
timestep before the dynamics do. Explicit horizontal diffusion is the usual
example; see [PROPHET Horizontal Diffusion](prophet_horizontal_diffusion.md).

## Physics choices specific to global runs

Several parameterizations are either only meaningful, or only tested, in the
global setting:

  - [Radiation](radiation.md), with prescribed or time-varying trace gases,
    aerosols, and insolation.
  - [PROPHET](prophet.md), which supplies the subgrid-scale turbulence,
    convection, and clouds that a coarse grid cannot resolve. See
    [Configuring and Tuning PROPHET](prophet_howto.md).
  - [Non-orographic](non_orographic_gravity_wave.md) and
    [orographic](orographic_gravity_wave.md) gravity-wave drag, which the
    middle-atmosphere circulation needs once the model top is in the
    mesosphere.
  - [Topography](topography.md), through `topography: "Earth"`.
  - The surface: `surface_setup` selects the flux scheme, and
    `prognostic_surface` chooses between prescribed sea surface temperatures and
    a slab ocean.

## Tested production configurations

The configurations in `config/longrun_configs/` are the ones exercised on GPUs
by the long-run pipeline, each on a single GPU with a 12- or 24-hour walltime.
They are the best starting points for a new global run.

| Configuration                                        | Common numerics                      | Length   | Physics                                                                                         |
|:---------------------------------------------------- |:------------------------------------ |:-------- |:----------------------------------------------------------------------------------------------- |
| `longrun_hydrostatic_balance.yml`                    | he30ze63                             | 360 days | Dynamics only, 55 km top                                                                        |
| `longrun_dry_baroclinic_wave_he60.yml`               | he30ze43, overridden to `h_elem: 60` | 30 days  | Dry dynamics at 55 km resolution                                                                |
| `longrun_moist_baroclinic_wave_he60.yml`             | he30ze43, overridden to `h_elem: 60` | 30 days  | 0-moment microphysics                                                                           |
| `longrun_dry_held_suarez.yml`                        | he30ze63                             | 360 days | Held–Suarez forcing                                                                             |
| `longrun_moist_held_suarez.yml`                      | he30ze63                             | 360 days | Held–Suarez, moist, prescribed vertical diffusion                                               |
| `longrun_aquaplanet_allsky_1M.yml`                   | he16ze63                             | 120 days | Eddy-diffusivity-only EDMFX, 1-moment microphysics, all-sky radiation                           |
| `longrun_aquaplanet_allsky_progedmf_0M.yml`          | he16ze63                             | 360 days | PROPHET, 0-moment microphysics                                                                  |
| `longrun_aquaplanet_allsky_progedmf_1M.yml`          | he16ze63                             | 120 days | PROPHET, 1-moment microphysics                                                                  |
| `longrun_aquaplanet_allsky_tvinsol_0M_slabocean.yml` | he16ze63                             | 360 days | Time-varying insolation, slab ocean                                                             |
| `amip_target.yml`                                    | he16ze63                             | 60 days  | Earth topography, PROPHET, 1-moment microphysics, prescribed aerosols, time-varying trace gases |

Run one of them by combining it with its numerical configuration:

```julia
import ClimaAtmos as CA

config = CA.AtmosConfig(
    [
        "config/common_configs/numerics_sphere_he16ze63.yml",
        "config/longrun_configs/longrun_aquaplanet_allsky_progedmf_1M.yml",
    ];
    job_id = "my_aquaplanet",
)
simulation = CA.AtmosSimulation(config)
CA.solve_atmos!(simulation)
```

Later files override earlier ones key by key, so a model configuration that
pins its own `h_elem` or `dt` wins over the common configuration it is combined
with.

`amip_target.yml` is the atmosphere-only counterpart of the coupled AMIP setup:
it prescribes sea surface temperatures. Coupled
simulations with interactive land, ocean, and sea ice are run from
[ClimaCoupler.jl](https://clima.github.io/ClimaCoupler.jl/stable/), which drives
ClimaAtmos as one component.

## Output and analysis

Output lands in a numbered subdirectory of `output/<job_id>`, with
`output/<job_id>/output_active` pointing at the most recent run.

With `output_default_diagnostics: true`, the default, the model writes the
default diagnostics for whichever components are active. Add a `diagnostics:`
block to request specific variables and periods; see
[Computing and saving diagnostics](diagnostics.md) for the format and
[Available Diagnostics](available_diagnostics.md) for the catalog.

Two keys control how the fields are written to NetCDF:

  - `netcdf_output_at_levels`, `true` by default, interpolates horizontally onto
    a longitude–latitude grid but leaves the vertical on model levels. Set it to
    `false` to interpolate onto height levels as well; those levels are spaced
    exponentially to approximate constant-pressure surfaces, but the coordinate
    stays height. For true pressure coordinates, set `pressure_coordinates: true`
    on the diagnostic.
  - `netcdf_interpolation_num_points` overrides the size of the target grid, as
    a list, for example `[360, 180, 2]` for the high-resolution baroclinic-wave
    runs. The third entry, the vertical, is ignored while
    `netcdf_output_at_levels` is `true`, since the output lives on the model
    levels instead.

Reading and plotting the output is covered in
[Loading and Visualizing Output](visualizing_output.md).

For runs long enough to outlast a queue slot, set `dt_save_state_to_disk` — the
production configurations use 10 or 30 days — and restart from the checkpoint;
see [Restarting and Checkpointing](restarts.md). It defaults to `Inf`, which
writes no checkpoint at all.

## Running at scale

We run global simulations at 200 km and finer resolution on GPUs, which are
more energy- and cost-efficient than CPUs for this work. The long-run pipeline
uses one GPU per job, enough for the resolutions above; MPI across several
devices is available for larger problems. Selecting a device, running
under MPI, and combining the two are covered in
[Running on GPUs and MPI](gpu_and_mpi.md).

## Practical notes

  - Sponge and model-top settings should follow the model top, not be carried
    over from a shallower configuration. The Rayleigh and viscous sponge onset
    heights are absolute altitudes, so a configuration tuned for a 60 km top
    needs to be adjusted for a 30 km domain; see
    [Model Top and Sponge Layer](sponge.md).
  - Earth topography is smoothed relative to the grid rather than by a fixed
    length, so the same `topography_damping_factor` removes a comparable number
    of grid scales at any resolution and a larger physical scale as the grid
    coarsens; see [Topography](topography.md).
  - Halving `dt_rad` doubles the number of radiation calls, which is the single
    largest addition to run time in these configurations.
  - A run that becomes unstable shortly after start is more often a timestep or
    sponge mismatch than a physics problem. Compare against the tested
    configuration closest to your resolution before changing parameterizations.
