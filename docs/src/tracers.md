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
| Updraft constraint enforcement | `mass_flux_closures.jl` | `for χ_name in sgs_tracer_names(Y)` |
| Rayleigh sponge damping | `remaining_tendency.jl` | `for χ_name in sgs_tracer_names(Y)` |

### Precipitating vs non-precipitating SGS tracers

Some operations apply different coefficients to precipitating species
(e.g. reduced vertical diffusion). The `@generated` function
`is_precip_sgs_tracer(χ_name)` identifies `q_rai`, `q_sno`, and `n_rai`
as precipitating. All other SGS tracers (including user-defined ones)
receive the default non-precipitating coefficient.

## Adding a New Passive Tracer

To add a new passive tracer `A` that is transported through the full
grid-scale + SGS system, only **two** changes are needed:

### Step 1: Add `ρA` to the grid-scale prognostic state

In `prognostic_variables.jl`, add `ρA` to the center variables:

```julia
ρA = ρ * physical_state.A
```

This gives automatic grid-scale advection, diffusion, hyperdiffusion,
and surface flux — all handled by `foreach_gs_tracer`.

### Step 2: Add `A` to the SGS updraft state

In `prognostic_variables.jl`, add `A` to the SGS struct:

```julia
sgsʲs = uniform_subdomains((; ρa, mse, q_tot, A = physical_state.A), turbconv_model)
```

This gives automatic SGS entrainment, mass flux, diffusive flux,
vertical diffusion, updraft constraints, advection, and sponge damping —
all handled by `sgs_tracer_names`.

### Step 3: Initial condition

Set the initial value of `A` in the setup file (e.g. `Bomex.jl`):

```julia
A = FT(1.0)  # constant initial concentration
```

That's it — no tendency code changes needed.

### Step 4 (if using implicit solver): Update the Jacobian

The implicit solver's Jacobian (`manual_sparse_jacobian.jl`) uses hardcoded
tracer lists for performance reasons (`unrolled_foreach` with closures inside
`@.` broadcasts allocates; see the `TODO` comment at the top of
`update_jacobian!`). All locations are marked with `# TRACER-JACOBIAN:`
comments — search for that string to find every block.

For a **passive tracer** (doesn't affect pressure or buoyancy, has no
sedimentation velocity), update these 4 locations:

#### 4a. Sparsity pattern (`jacobian_cache`)

Add the new tracer to the name lists that define which matrix blocks exist:

- `condensate_mass_names` / `condensate_names`: add `@name(c.ρA)`
- `sgs_condensate_mass_names` / `sgs_condensate_names`: add `@name(c.sgsʲs.:(1).A)`

#### 4b. SGS vertical diffusion block

Search for `TRACER-JACOBIAN: SGS vertical diffusion`. Add:
```julia
(@name(c.sgsʲs.:(1).A), FT(1))          # non-precipitating
```

#### 4c. SGS entrainment/detrainment block

Search for `TRACER-JACOBIAN: SGS entrainment`. Add:
```julia
(@name(c.sgsʲs.:(1).A))
```

#### 4d. Grid-mean + SGS mass flux block

Search for `TRACER-JACOBIAN: grid-mean + SGS mass flux`. Add:
```julia
(@name(c.ρA), @name(c.sgsʲs.:(1).A), @name(A))
```

For a **moisture species** (affects pressure, buoyancy, and/or sediments),
also update these physics-specific blocks:

#### 4e. Grid-mean pressure gradient block

Search for `TRACER-JACOBIAN: grid-mean pressure gradient`. Add:
```julia
(@name(c.ρA), e_int_A0, Δcv_A)   # thermodynamic coefficients
```

#### 4f. Grid-mean sedimentation block

Search for `TRACER-JACOBIAN: grid-mean sedimentation`. Add:
```julia
(@name(c.ρA), @name(ᶜwA), α)     # sedimentation velocity + diffusion coeff
```

#### 4g. SGS pressure/buoyancy block

Search for `TRACER-JACOBIAN: SGS pressure/buoyancy`. Add:
```julia
(@name(c.sgsʲs.:(1).A), LH_A, ∂cp∂A, ∂Rm∂A)
```

#### 4h. SGS sedimentation block

Search for `TRACER-JACOBIAN: SGS sedimentation`. Add:
```julia
(@name(c.sgsʲs.:(1).A), @name(ᶜwAʲs.:(1)))
```

### Operations that remain manual

| Operation | Reason |
|---|---|
| Initial / boundary conditions | Problem-specific |
| Source / sink terms | Physics-specific |
| Jacobian blocks (implicit solver) | See Step 4 above |
| Diagnostics output | User must define short names |

## Implementation details

The auto-discovery relies on two key patterns:

1. **`@generated` predicates** — `_is_sgs_tracer_name` and
   `is_ρ_weighted_name` return compile-time `Bool` constants, enabling
   `unrolled_filter` to resolve the tracer list at compile time with
   zero runtime cost.

2. **`MatrixFields.get_field` + `FieldName`** — tracer fields are
   accessed via `MatrixFields.get_field(Y.c.sgsʲs.:(1), χ_name)` using
   the discovered `FieldName`. This is equivalent to direct property
   access (e.g. `Y.c.sgsʲs.:(1).q_lcl`) and compiles to the same code.

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
