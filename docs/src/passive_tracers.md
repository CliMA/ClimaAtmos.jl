# Passive Tracers

ClimaAtmos provides automatic treatment of conserved scalar tracers at two
levels: **grid-scale** (resolved) and **sub-grid scale** (SGS, inside
PROPHET updrafts). Both levels use an auto-discovery mechanism: any
field that follows the naming convention is automatically picked up for
transport, diffusion, and other generic operations; no
additional code changes are required.

## Grid-Scale Tracers

Grid-scale tracers are density-weighted scalars ``\rho \chi`` stored at cell
centers in the prognostic state `Y.c`.

### Naming convention

A grid-scale tracer is identified by a name that starts with `ρ` followed
by the scalar name, e.g. `ρq_tot`, `ρq_lcl`, `ρn_rai`. The utility function
`gs_tracer_names(Y)` discovers all such tracers automatically by keeping only
top-level `ρ`-prefixed names in `Y.c` (the `is_ρ_weighted_name` predicate,
which already excludes `uₕ` and `sgsʲs`) and then excluding `ρ`, `ρe_tot`,
and `ρtke`.

### Automatically handled operations

| Operation            | Description                                            |
|:-------------------- |:------------------------------------------------------ |
| Horizontal advection | Flux-form divergence of ``\rho \chi \boldsymbol{u}_h`` |
| Vertical advection   | Upwinded vertical transport                            |
| Vertical diffusion   | Eddy-diffusivity-based mixing                          |
| Hyperdiffusion       | 4th-order ``\nabla^4`` stabilization with DSS          |

The iteration utility `foreach_gs_tracer(f, Y...)` applies a function `f` to
each discovered tracer.

## SGS Tracers (PROPHET)

When PROPHET is enabled, each updraft carries its own set of scalar
fields inside `Y.c.sgsʲs.:(j)`. The utility function `sgs_tracer_names(Y)`
discovers all scalars in the first updraft (`Y.c.sgsʲs.:(1)`) and excludes
the core PROPHET variables `ρa`, `mse`, and `q_tot`, which receive
physics-specific treatment.

### Naming convention

An SGS tracer `χ` in `Y.c.sgsʲs.:(j)` maps to a grid-scale
density-weighted counterpart `ρχ` in `Y.c`. For example:

| SGS field (in `sgsʲs.:(j)`) | Grid-scale field (in `Y.c`) |
|:--------------------------- |:--------------------------- |
| `q_lcl`                     | `ρq_lcl`                    |
| `q_rai`                     | `ρq_rai`                    |
| `n_rai`                     | `ρn_rai`                    |
| `A` (user-defined)          | `ρA`                        |

This pairing is enforced by `get_ρχ_name(χ_name)`, which constructs
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
| SGS diffusive flux (grid mean)                  | `edmfx_sgs_flux.jl`     | `foreach_gs_tracer(Yₜ, Y)`          |
| SGS hyperdiffusion                              | `hyperdiffusion.jl`     | `for χ_name in sgs_tracer_names(Y)` |
| Updraft constraint enforcement                  | `mass_flux_closures.jl` | `for χ_name in sgs_tracer_names(Y)` |
| Rayleigh sponge damping                         | `remaining_tendency.jl` | `for χ_name in sgs_tracer_names(Y)` |

Sedimenting microphysics species are diffused with the reduced coefficient
`α_vert_diff_microphysics * K_h`; passive tracers use the unscaled `K_h`.

## Adding a new passive tracer

Adding a tracer is developer territory; see
[Adding a Passive Tracer](extending_tracers.md) in the Developer Guide.
