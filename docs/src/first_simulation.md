# Your First Simulation

This page walks through building, running, and inspecting one simulation. To
configure each component in turn, see
[Scripting Simulations](scripting_simulations.md); to run from a YAML file
instead, see [Creating Custom Configurations](configuration.md), and for how
the two interfaces relate, [Script vs Config Interface](interfaces.md).

## Minimal example

The simplest ClimaAtmos simulation uses all defaults. It solves the dry
compressible equations (with hyperdiffusion and surface fluxes) on a global
cubed-sphere grid, starting from a hydrostatically balanced, slightly perturbed
state with a vertically decaying temperature profile:

```@example first_sim
using Logging # hide
Logging.disable_logging(Logging.Info) # hide
import ClimaAtmos as CA

simulation = CA.AtmosSimulation{Float32}(; t_end = "1days")
nothing # hide
```

`t_end` accepts a number of seconds or a duration string (`secs`, `mins`,
`hours`, `days`, `weeks`), as does the timestep `dt`. Every other aspect of the
simulation has a keyword argument too; omitted ones take their defaults.

The first construction and solve in a session compile a large amount of code
and can take several minutes; later calls are fast.

## Inspecting the state

Constructing an `AtmosSimulation` sets everything up but does not advance it in
time. Even before running, the initial state is available through the
integrator:

!!! note

    `Y` is the state vector: `Y.c` holds the cell-center variables (such as
    the density `ρ` and the total energy `ρe_tot`) and `Y.f` the cell-face
    variables (such as the vertical velocity `u₃`). The `integrator` is the
    ODE integrator, from ClimaTimeSteppers, that advances the state in time.
    See the [Glossary](@ref) for these and other recurring names.

```@example first_sim
Y = simulation.integrator.u

# Center (cell-center) variables
propertynames(Y.c)  # (:ρ, :uₕ, :ρe_tot) for this dry default

# Face (cell-interface) variables
propertynames(Y.f)  # e.g., (:u₃,)
```

## Running a case end to end

The default simulation is plain, and a global run is slow to
integrate. Presets bundle a grid, a setup, and matching physics into one call,
which is the quickest way to a running case, here a column with the BOMEX
shallow-cumulus initial state and moist physics. `solve_atmos!` integrates it
forward to `t_end`, and the integrator time confirms where the run stopped:

```@example first_sim
simulation = CA.Presets.bomex(Float32; t_end = "10mins", output_dir = mktempdir())
CA.solve_atmos!(simulation)
simulation.integrator.t
```

(This page runs during the documentation build, so it writes to a temporary
directory; drop `output_dir` to get the default location described below.)

Presets matter beyond brevity: the `setup` argument sets the initial state,
while the physics comes from the model, so the two have to be chosen together.
BOMEX with the default dry model would have no moisture to convect. Each preset
pairs a setup with a matching grid and model. The pairing is minimal rather
than complete: `Presets.bomex` enables moist physics but no
turbulence-convection scheme or case forcings; pass
`model = CA.Presets.prognostic_edmf(Float32)` to add convective transport, or
run the corresponding YAML case config for the full published setup. See the
[Presets](api.md#Presets) section of the API for the full list.

## Where output goes

Output is written to `simulation.output_dir`. The base directory defaults to
`output/<job_id>` under the directory Julia was started in (just `<job_id>`
when the `CI` environment variable is set); with the default `job_id` of
`atmos_sim`, that is `output/atmos_sim`. Each run writes to a numbered
subdirectory of the base directory — `simulation.output_dir` is that
subdirectory, such as `output/atmos_sim/output_0000` — and
`output/atmos_sim/output_active` links to the most recent one. Two
formats appear there, each with a distinct role:

  - **NetCDF** (`.nc`) files hold the **diagnostics** -- derived (and often interpolated)
    output variables such as temperature or precipitation. See
    [Computing and saving diagnostics](@ref) for how to configure them.
  - **HDF5** (`.hdf5`) files hold full-resolution **model-state checkpoints**, written when
    `checkpoint_frequency` is set. These are the files a simulation reads to
    [restart](@ref "Restarting and Checkpointing").

[Loading and Visualizing Output](visualizing_output.md) covers reading the
NetCDF files with ClimaAnalysis.

## Next steps

  - [Scripting Simulations](@ref) -- configure the grid, model, setup, and
    diagnostics from a script, and step the integrator interactively
  - [Script vs Config Interface](@ref) -- the same runs from YAML files
  - [Running Single-Column Cases](@ref) -- BOMEX, DYCOMS, RICO, and more
  - [Computing and saving diagnostics](@ref) -- configure output variables and formats
  - [Glossary](@ref) -- the state vector `Y`, the cache `p`, and other recurring symbols
