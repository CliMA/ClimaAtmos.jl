# ClimaAtmos.jl

ClimaAtmos is the atmosphere model of the Climate Modeling Alliance (CliMA)
Earth system model. CliMA is a consortium, led by Caltech and MIT, that has
built a new Earth system model from scratch, in Julia, designed to learn from
data. ClimaAtmos is its nonhydrostatic atmosphere component, spanning large-eddy
simulation in Cartesian domains to global weather and climate simulation on
the sphere [Yatunin2026](@cite).

The model can be run from the Julia REPL, with no namelists or batch scripts
required; a first global simulation takes three lines of code (see
[Your First Simulation](first_simulation.md)).

## What the model does

  - **One model across scales.** ClimaAtmos solves the fully compressible
    equations of motion for a deep atmosphere in a coordinate-independent
    formulation. The same equation set runs in Cartesian geometries, for
    cloud-resolving and large-eddy simulations, and on the sphere, for global
    weather and climate simulations.
  - **Conservative by construction.** Energy, air mass, and water are conserved
    to floating-point precision, without ad hoc fixers. Using the specific
    total energy of moist air as a prognostic variable, together with an
    internally consistent thermodynamic formulation, guarantees closed budgets
    even in moist atmospheres and in the presence of subgrid-scale
    parameterizations.
  - **Performance portable.** The model runs on CPUs and GPUs from the same
    code base and scales from a laptop to cloud supercomputers, with strong and
    weak scaling that make kilometer-scale global simulations achievable.
  - **Resolution-adaptive physics.** The parameterization suite, built around
    the PROPHET scheme (an extended, prognostic eddy-diffusivity mass-flux
    scheme), avoids assumptions of scale separation that become inadequate as
    resolved scales approach the scales of parameterized processes such as
    atmospheric turbulence and convection. As resolution increases, the
    parameterized transport diminishes and hands over to the resolved flow.
  - **Built for calibration with data.** All model parameters live in a
    central repository, [ClimaParams.jl](https://github.com/CliMA/ClimaParams.jl),
    and CliMA's calibration tools tune parameters against data, whether those
    are output from high-resolution simulations or Earth observations from space and from the ground.

To learn more, the dynamical core (concepts, numerics, and scaling) is
described in [Yatunin2026](@cite), with companion papers to follow.

## ClimaAtmos in the CliMA ecosystem

ClimaAtmos composes packages from the CliMA ecosystem into a full model;
installing ClimaAtmos brings every dependency along.

  - [ClimaCore.jl](https://clima.github.io/ClimaCore.jl/stable/) provides the
    numerical methods for the dynamical core: the
    spectral-element/finite-difference discretization, fields, and discrete
    operators on which the [governing equations](equations.md) are built.
  - [ClimaTimeSteppers.jl](https://clima.github.io/ClimaTimeSteppers.jl/stable/)
    provides the implicit–explicit (IMEX) time steppers designed for Earth
    system models, used together with the [implicit solver](implicit_solver.md).
  - [Thermodynamics.jl](https://clima.github.io/Thermodynamics.jl/stable/) and
    [ClimaParams.jl](https://clima.github.io/ClimaParams.jl/stable/) are the
    shared foundation: one thermodynamic formulation for all CliMA components
    (the basis for energetic consistency) and a single source of truth for all
    model parameters.
  - Physics libraries supply the parameterized processes:
    [Insolation.jl](https://clima.github.io/Insolation.jl/stable/),
    [RRTMGP.jl](https://clima.github.io/RRTMGP.jl/stable/),
    [SurfaceFluxes.jl](https://clima.github.io/SurfaceFluxes.jl/stable/), and
    [CloudMicrophysics.jl](https://clima.github.io/CloudMicrophysics.jl/stable/).
  - [ClimaCoupler.jl](https://clima.github.io/ClimaCoupler.jl/stable/) couples
    ClimaAtmos to the CliMA land, ocean, and sea-ice models, and
    [ClimaCalibrate.jl](https://clima.github.io/ClimaCalibrate.jl/stable/)
    drives the calibration-with-data workflows.

See [The CliMA Ecosystem](ecosystem.md) for the full architectural overview,
including where each package enters the ClimaAtmos source code.

## Finding your way around

Each section of these docs answers a different need; head for the one that
matches yours:

  - **Getting Started**: new to ClimaAtmos? Start here.

      + [Installation](installation.md) -- install the package, or run from a clone
      + [Your First Simulation](first_simulation.md) -- build, run, and inspect one simulation
      + [Script vs Config Interface](interfaces.md) -- the two ways to configure a run

  - **How-to Guides**: task recipes for running and configuring simulations.

      + Running simulations: [single-column cases](single_column.md), [global simulations](global_simulations.md), [restarts](restarts.md), [running on GPUs and MPI](gpu_and_mpi.md)
      + Configuration: [scripting simulations](scripting_simulations.md), [custom configurations](configuration.md)
      + [Computing and saving diagnostics](diagnostics.md)
      + [Loading and visualizing output](visualizing_output.md)

  - **Explanation**: the science and numerics behind the model.

      + [The CliMA ecosystem](ecosystem.md) -- how ClimaAtmos composes the CliMA packages
      + Dynamics & numerics: [governing equations](equations.md), [implicit solver](implicit_solver.md), [integer time (ITime)](itime.md)
      + Physics & parameterizations: [PROPHET](edmf_equations.md), [microphysics](microphysics.md), [radiation](radiation.md), [non-orographic gravity-wave drag](non_orographic_gravity_wave.md), [orographic gravity-wave drag](orographic_gravity_wave.md), [ocean surface albedo](surface_albedo.md), [topography](topography.md)

  - **Reference**: look-up material.

      + [API](api.md), [Configuration options](configuration_options.md), [Setups](setups.md), [Column Datasets](column_datasets_reference.md), [Grids](grids.md), [Surface conditions](surface_conditions.md), [Passive tracers](passive_tracers.md), [Trace gases](trace_gases.md), [Available diagnostics](available_diagnostics.md), [Notation](notation.md), [Glossary](glossary.md), [Bibliography](references.md)

  - **Developer Guide**: [contributing](contributor_guide.md) and extending the model ([setups](extending_setups.md), [diagnostics](extending_diagnostics.md), [tracers](extending_tracers.md), [column datasets](extending_column_datasets.md), [surface internals](surface_conditions_internals.md))

New here? Start with [Installation](@ref) and [Your First Simulation](@ref), then pick
the workflow that suits you in [Script vs Config Interface](@ref). For column experiments,
see [Running Single-Column Cases](@ref).

ClimaAtmos is open source under the Apache 2.0 license. Questions and bug
reports are welcome on the
[GitHub issue tracker](https://github.com/CliMA/ClimaAtmos.jl/issues).
