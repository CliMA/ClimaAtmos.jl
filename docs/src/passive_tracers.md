# Tracers

ClimaAtmos provides automatic treatment of conserved scalar tracers at two
levels: **grid-scale** (resolved) and **sub-grid scale** (SGS, inside
prognostic EDMF updrafts). Both levels use an auto-discovery mechanism: any
field that follows the naming convention is automatically picked up for
transport, diffusion, and other generic operations — no
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

| Operation            | Description                                            |
|:-------------------- |:------------------------------------------------------ |
| Horizontal advection | Flux-form divergence of ``\rho \chi \boldsymbol{u}_h`` |
| Vertical advection   | Upwinded vertical transport                            |
| Vertical diffusion   | Eddy-diffusivity-based mixing                          |
| Hyperdiffusion       | 4th-order ``\nabla^4`` stabilization with DSS          |

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
| `q_gas_A` (user-defined) | `ρq_gas_A` |

This pairing is enforced by `get_ρχ_name(χ_name)` which constructs
`ρχ` from `χ`.

### Automatically handled operations

The following operations are auto-discovered for all SGS tracers. No code
changes are needed when adding a new tracer:

| Operation                                       | File                    | Pattern                             |
|:----------------------------------------------- |:----------------------- |:----------------------------------- |
| Horizontal advection                            | `advection.jl`          | `for χ_name in sgs_tracer_names(Y)` |
| Vertical advection (advective form)             | `advection.jl`          | `for χ_name in sgs_tracer_names(Y)` |
| Entrainment/detrainment mixing                  | `edmfx_entr_detr.jl`    | `for χ_name in sgs_tracer_names(Y)` |
| SGS mass flux (draft + environment → grid mean) | `edmfx_sgs_flux.jl`     | `for χ_name in sgs_tracer_names(Y)` |
| SGS diffusive flux (grid mean)                  | `edmfx_sgs_flux.jl`     | `for χ_name in sgs_tracer_names(Y)` |
| Updraft vertical diffusion                      | `mass_flux_closures.jl` | `for χ_name in sgs_tracer_names(Y)` |
| Updraft constraint enforcement                  | `mass_flux_closures.jl` | `for χ_name in sgs_tracer_names(Y)` |
| Rayleigh sponge damping                         | `remaining_tendency.jl` | `for χ_name in sgs_tracer_names(Y)` |

All SGS tracers (cloud species and precipitation alike) receive the same
reduced vertical diffusion coefficient (`α_vert_diff_microphysics`).

## Adding a New Passive Tracer

To add a new passive tracer `q_gas_A` that is transported through the full
grid-scale + SGS system, the only changes needed are:

### Step 1: Add `ρq_gas_A` to the grid-scale prognostic state

In `prognostic_variables.jl`, add `ρq_gas_A` to the center variables:

```julia
ρq_gas_A = ρ * physical_state.q_gas_A
```

This gives automatic grid-scale advection, diffusion, hyperdiffusion,
and surface flux — all handled by `foreach_gs_tracer`.

### Step 2: Add `q_gas_A` to the SGS updraft state

In `prognostic_variables.jl`, add `q_gas_A` to the SGS struct:

```julia
sgsʲs = uniform_subdomains((; ρa, mse, q_tot, q_gas_A = physical_state.q_gas_A), turbconv_model)
```

This gives automatic SGS entrainment, mass flux, diffusive flux,
vertical diffusion, updraft constraints, advection, and sponge damping —
all handled by `sgs_tracer_names`.

### Step 3: Initial condition

Set the initial value of `q_gas_A` in the setup file (e.g. `Bomex.jl`):

```julia
q_gas_A = FT(1.0)  # constant initial concentration
```

That's it — no tendency code changes needed.

### Step 4 (if using implicit solver): Update the Jacobian

The implicit solver's Jacobian (`manual_sparse_jacobian.jl`) uses hardcoded
tracer lists for performance reasons (`unrolled_foreach` with compile-time
tuples). Adding a new tracer to the Jacobian requires manually editing
several locations in `jacobian_cache` (sparsity pattern) and
`update_jacobian!` (numeric updates). Search for existing microphysics
tracer names (e.g. `q_lcl`, `q_rai`) to find each block and add the new
tracer alongside them.

Key locations to update:

| Location | What to add |
|---|---|
| `condensate_names` / `condensate_mass_names` in `jacobian_cache` | `@name(c.ρq_gas_A)` |
| `sgs_condensate_names` / `sgs_condensate_mass_names` in `jacobian_cache` | `@name(c.sgsʲs.:(1).q_gas_A)` |
| SGS vertical diffusion block | Append to `sgs_microphysics_tracers` tuple |
| SGS entrainment block | Append to `sgs_microphysics_tracers` tuple |
| Grid-mean + SGS mass flux block | Append to `microphysics_tracers` tuple |

For **moisture species** that affect pressure, buoyancy, or have a
sedimentation velocity, additional blocks need updating (pressure
gradient, sedimentation, SGS pressure/buoyancy). Search for existing
species like `q_rai` to locate each block.

!!! note

    A passive tracer that doesn't affect thermodynamics and has no
    sedimentation can often run without Jacobian entries. The implicit
    solver will still converge, just more slowly for that variable.

### Operations that remain manual

| Operation                         | Reason                       |
|:--------------------------------- |:---------------------------- |
| Initial / boundary conditions     | Problem-specific             |
| Source / sink terms               | Physics-specific             |
| Jacobian blocks (implicit solver) | See Step 4 above             |
| Diagnostics output                | User must define short names |

## Implementation details

The auto-discovery relies on two key patterns:

 1. **Field-name predicates** — `_is_sgs_tracer_name` and
    `is_ρ_weighted_name` filter the top-level field names at the type
    level, enabling `unrolled_filter` to resolve the tracer list with
    zero runtime cost.

 2. **`MatrixFields.get_field` + `FieldName`** — tracer fields are
    accessed via `MatrixFields.get_field(Y.c.sgsʲs.:(1), χ_name)` using
    the discovered `FieldName`. This is equivalent to direct property
    access (e.g. `Y.c.sgsʲs.:(1).q_lcl`) and compiles to the same code.
