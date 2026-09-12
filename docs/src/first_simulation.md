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

model = CA.AtmosModel(CA.SphereGrid(Float32))
simulation = CA.AtmosSimulation(model; t_end = "1days")
nothing # hide
```

Construction has two steps: `AtmosModel(grid; ...)` defines the physical
system -- it holds the grid, the parameters, and the case setup -- and
`AtmosSimulation(model; ...)` defines how to run that system: the timestep, the
duration, the callbacks, and the output.

`t_end` accepts a number of seconds or a duration string (`secs`, `mins`,
`hours`, `days`, `weeks`), as does the timestep `dt`. See
[`AtmosModel`](@ref ClimaAtmos.AtmosModel) and
[`AtmosSimulation`](@ref ClimaAtmos.AtmosSimulation) for the full list of
keyword arguments and their defaults.

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

The default simulation is plain, and a global run is slow to integrate. A
[setup](setups.md) supplies a case: its initial state, its boundary conditions,
and its forcings. Here that is BOMEX, a shallow-cumulus column, paired with
moist physics from a model preset. `solve_atmos!` integrates the simulation
forward to `t_end`, and the integrator time confirms where the run stopped:

```@example first_sim
grid = CA.ColumnGrid(Float32; z_elem = 60, z_max = 3000.0, z_stretch = false)
params = CA.ClimaAtmosParameters(Float32)
model = CA.AtmosModel(
    grid;
    params,
    setup = CA.Setups.Bomex(; thermo_params = params.thermodynamics_params),
    defaults = CA.Presets.equil_moist_0m(),
)
simulation = CA.AtmosSimulation(
    model; dt = 10, t_end = "10mins", output_dir = mktempdir(),
)
CA.solve_atmos!(simulation)
simulation.integrator.t
```

(This page runs during the documentation build, so it writes to a temporary
directory; drop `output_dir` to get the default location described below.)

The setup and the physics are not independent: the setup sets the initial state
and the case forcings, while the parameterizations come from the rest of the
model, so the two have to be chosen together. A setup can require prognostic
variables that only some physics provide. BOMEX with the default dry model
would have no moisture to convect, which is why `equil_moist_0m` appears above.
Model presets provide reasonable physics defaults for simulations scientists
commonly run. The setup and your own keyword arguments override them. The one
above enables moist physics but no turbulence-convection scheme. Pass
`defaults = CA.Presets.prognostic_edmf(Float32)` instead to add convective
transport, or run the corresponding YAML case config for the full published
setup. See the [Presets](api.md#Presets) section of the API for the full list.

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
