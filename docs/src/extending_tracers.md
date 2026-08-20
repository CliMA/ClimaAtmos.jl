# Adding a Passive Tracer

How to add a new passive tracer that is transported through the full
grid-scale + SGS system. For the tracer naming conventions and the operations
that are handled automatically, see [Passive Tracers](passive_tracers.md).

The model includes a working, config-selectable passive tracer:
setting `chemistry_model: "passive"` enables the tracer `q_gas_A`, threaded
through [`physical_state`](@ref ClimaAtmos.Setups.physical_state) and the
prognostic-variable construction, with `q_gas_A`/`q_gas_Aup` diagnostics. It is
exercised by [`config/model_configs/prognostic_edmfx_bomex_tracerA_column.yml`](https://github.com/CliMA/ClimaAtmos.jl/blob/main/config/model_configs/prognostic_edmfx_bomex_tracerA_column.yml)
and is the best reference implementation for the steps below.

To add a new passive tracer `A` that is transported through the full
grid-scale + SGS system, the changes needed are:

## Step 1: Add `ρA` to the grid-scale prognostic state

In `prognostic_variables.jl`, add `ρA` to the center variables:

```julia
ρA = ρ * physical_state.A
```

This gives automatic grid-scale advection, diffusion, hyperdiffusion,
and surface flux (zero by default; only `ρq_tot` has a nonzero surface flux)
— all handled by `foreach_gs_tracer`.

## Step 2: Add `A` to the SGS updraft state

In `prognostic_variables.jl`, add `A` to the SGS struct:

```julia
sgsʲs = uniform_subdomains((; ρa, mse, q_tot, A = physical_state.A), turbconv_model)
```

This gives automatic SGS entrainment, mass flux, diffusive flux,
vertical diffusion, updraft constraints, advection, and sponge damping —
all handled by `sgs_tracer_names`.

## Step 3: Initial condition

`physical_state` has a closed keyword list, so first add the tracer to it
(`src/setups/common/physical_state.jl`, as done for `q_gas_A`), then set its
initial value in the setup file (e.g. `Bomex.jl`):

```julia
A = FT(1.0)  # constant initial concentration
```

No tendency code changes are needed.

## Step 4 (if using implicit solver): Jacobian

No Jacobian changes are needed for a passive tracer. The implicit solver's
Jacobian (`manual_sparse_jacobian.jl`) discovers tracers through the process
classification in `src/utils/tracer_processes.jl`; a passive tracer is matched
by `passive_gs_tracer_names`/`passive_sgs_tracer_names` automatically.

Only a tracer that **sediments** requires edits: add it to
`gs_sedimenting_tracer_candidates`/`sgs_sedimenting_tracer_candidates` and to
`sedimentation_velocity_name`/`sgs_sedimentation_velocity_name` in
`tracer_processes.jl`, so the sedimentation Jacobian blocks pick it up.

## Operations that remain manual

| Operation                      | Reason                                      |
|:------------------------------ |:------------------------------------------- |
| Initial / boundary conditions  | Problem-specific                            |
| Source / sink terms            | Physics-specific                            |
| Sedimentation Jacobian entries | See Step 4 above (sedimenting tracers only) |
| Diagnostics output             | User must define short names                |

# Implementation details

The auto-discovery relies on two patterns:

 1. **Field-name predicates**: `_is_sgs_tracer_name` and
    `is_ρ_weighted_name` filter the top-level field names at the type
    level, enabling `unrolled_filter` to resolve the tracer list with
    zero runtime cost.

 2. **`MatrixFields.get_field` + `FieldName`**: tracer fields are
    accessed via `MatrixFields.get_field(Y.c.sgsʲs.:(1), χ_name)` using
    the discovered `FieldName`. This is equivalent to direct property
    access (e.g. `Y.c.sgsʲs.:(1).q_lcl`) and compiles to the same code.
