# Tracers

ClimaAtmos provides automatic treatment of conserved scalar tracers at two
levels: **grid-scale** (resolved) and **sub-grid scale** (SGS, inside
prognostic EDMF updrafts). Both levels use an auto-discovery mechanism: any
field that follows the naming convention is automatically picked up for
transport, diffusion, hyperdiffusion, and other generic operations — no
additional code changes are required.

## Grid-Scale Tracers

Grid-scale tracers are density-weighted scalars ``\rho \chi`` stored at cell
centers in the prognostic state `Y.c`.

### Naming convention

A grid-scale tracer is identified by a name that starts with `ρ` followed
by the scalar name, e.g. `ρq_tot`, `ρq_lcl`, `ρn_rai`. The utility function
`gs_tracer_names(Y)` discovers all such tracers automatically by inspecting
`Y.c` and excluding non-tracer fields (`ρ`, `ρe_tot`, `uₕ`, `ρtke`,
`sgsʲs`).

### Automatically handled operations

| Operation | Description |
|---|---|
| Horizontal advection | Flux-form divergence of ``\rho \chi \boldsymbol{u}_h`` |
| Vertical advection | Upwinded vertical transport |
| Vertical diffusion | Eddy-diffusivity-based mixing |
| Hyperdiffusion | 4th-order ``\nabla^4`` stabilization with DSS |

Sedimentation is **not** auto-discovered at the grid scale because each
species requires its own terminal velocity field (e.g. `ᶜwₗ`, `ᶜwᵣ`).
Terminal velocity mappings must be wired explicitly.

The iteration utility `foreach_gs_tracer(f, Y...)` applies a function `f` to
each discovered tracer.

## SGS Tracers (Prognostic EDMF)

When prognostic EDMF is enabled, each updraft carries its own set of scalar
fields inside `Y.c.sgsʲs.:(j)`. The utility function `sgs_tracer_names(Y)`
discovers all scalars in the first updraft (`Y.c.sgsʲs.:(1)`) and excludes
the core EDMF variables `ρa`, `mse`, and `q_tot`, which receive
physics-specific treatment.

### Naming convention

An SGS tracer `χ` in `Y.c.sgsʲs.:(j)` maps to a grid-scale
density-weighted counterpart `ρχ` in `Y.c`. For example:

| SGS field (in `sgsʲs.:(j)`) | Grid-scale field (in `Y.c`) |
|---|---|
| `q_lcl` | `ρq_lcl` |
| `q_rai` | `ρq_rai` |
| `n_rai` | `ρn_rai` |
| `A` (user-defined) | `ρA` |

This pairing is enforced by `get_ρχ_name(χ_name)` which constructs
`ρχ` from `χ`.

### Automatically handled operations

The following operations are auto-discovered for all SGS tracers. No code
changes are needed when adding a new tracer:

| Operation | File | Pattern |
|---|---|---|
| Horizontal advection | `advection.jl` | `for χ_name in sgs_tracer_names(Y)` |
| Vertical advection (advective form) | `advection.jl` | `for χ_name in sgs_tracer_names(Y)` |
| Entrainment/detrainment mixing | `edmfx_entr_detr.jl` | `for χ_name in sgs_tracer_names(Y)` |
| SGS mass flux (draft + environment → grid mean) | `edmfx_sgs_flux.jl` | `for χ_name in sgs_tracer_names(Y)` |
| SGS diffusive flux (grid mean) | `edmfx_sgs_flux.jl` | `for χ_name in sgs_tracer_names(Y)` |
| Updraft vertical diffusion | `mass_flux_closures.jl` | `for χ_name in sgs_tracer_names(Y)` |
| Rayleigh sponge | `remaining_tendency.jl` | `for χ_name in sgs_tracer_names(Y)` |
| Filter/physical constraints | `mass_flux_closures.jl` | `for χ_name in sgs_tracer_names(Y)` |
| Hyperdiffusion | `hyperdiffusion.jl` | Sequential prep→DSS→apply |
| Manual sparse Jacobian | `manual_sparse_jacobian.jl` | Advection, entrainment, diffusion, mass flux |

Precipitating species (`q_rai`, `q_sno`, `n_rai`) automatically receive
reduced diffusion coefficients via `is_precip_sgs_tracer(χ_name)`.

### Operations that remain explicitly wired

Some operations couple to thermodynamics or specific physical processes and
cannot be auto-discovered:

| Operation | Why explicit |
|---|---|
| `mse` transport | Couples to enthalpy, buoyancy, kinetic energy |
| `q_tot` transport | Couples to density (`ρa`), thermodynamics, boundary conditions |
| `ρa`–`q_tot` coupling in hyperdiffusion | Physics-specific density correction |
| Sedimentation (terminal velocities) | Per-species velocity fields required |
| Microphysics source/sink tendencies | Physics-specific |

## Adding a New Passive Tracer

### Grid-scale only

To add a new passive tracer `A` that is advected, diffused, and
hyperdiffused at the grid scale:

1. Add `ρA = ρ * A_init` to the initial condition in
   `prognostic_variables.jl` (or your setup's initialization).

That's it — `gs_tracer_names(Y)` will discover `ρA` automatically.

### Grid-scale + prognostic EDMF

To also transport `A` inside EDMF updrafts:

1. Add `A = A_init` to the SGS physical state (initial condition).
2. Add `A = physical_state.A` to the `sgsʲs` `NamedTuple` in
   `prognostic_variables.jl`.
3. Add `ρA = ρ * physical_state.A` to the grid-scale center variables.

All EDMF operations (advection, entrainment mixing, SGS mass/diffusive
flux, Rayleigh sponge, filter/constraints, hyperdiffusion, and the
corresponding manual Jacobian entries) are picked up automatically by
`sgs_tracer_names(Y)`.

## Utility Functions

The following internal functions power the auto-discovery mechanism:

| Function | Purpose |
|---|---|
| `sgs_tracer_names(Y)` | Returns a `Tuple` of `FieldName`s for all SGS tracers in `Y` |
| `foreach_sgs_tracer(f, Y...)` | Applies `f` to each SGS tracer (uses `unrolled_foreach`) |
| `is_precip_sgs_tracer(χ_name)` | Returns `true` for precipitating species (`q_rai`, `q_sno`, `n_rai`) |
| `get_ρχ_name(χ_name)` | Maps `@name(χ)` → `@name(ρχ)` (SGS to grid-scale name) |
| `get_sgsʲ_name(χ_name)` | Maps `@name(χ)` → `@name(c.sgsʲs.:(1).χ)` (for Jacobian matrix keys) |
| `get_c_ρχ_name(χ_name)` | Maps `@name(χ)` → `@name(c.ρχ)` (for Jacobian matrix keys) |
| `gs_tracer_names(Y)` | Returns a `Tuple` of `FieldName`s for all grid-scale tracers in `Y` |
| `foreach_gs_tracer(f, Y...)` | Applies `f` to each grid-scale tracer |

---

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

z = range(0, 60000, length = 100)
ozone = ClimaAtmos.idealized_ozone.(z)

fig = Figure()
ax = Axis(fig[1, 1]; xlabel = "Ozone (VMR)", ylabel = "Altitude (m)")
lines!(ax, ozone, z)
save("idealized_ozone.png", fig); nothing # hide
```
![Idealized ozone profile](idealized_ozone.png)

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

### Time Invariant CO2 Profile

By default, CO2 concentrations are set to 397.547 ppm. The number can be altered
by changing the `CO2_fixed_value` parameter in the toml file.

### Time Varying CO2 Profile

`ClimaAtmos` can prescribe CO2 concentration using data
from [Mauna Loa CO2 measurements](https://gml.noaa.gov/ccgg/trends/data.html).
This option is enabled with by adding `"CO2"` to the `time_varying_gases`
config argument list, ie: `time_varying_gases: ["CO2"]`.
