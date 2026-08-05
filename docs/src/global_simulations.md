# Running Global Simulations

How to run global simulations (aquaplanets and AMIP-style configurations)
on the cubed sphere.

!!! note "Under construction"

    This page is a starting point; a full guide (resolution and timestep
    choices, output and analysis, GPU and MPI runs at scale) is planned. The
    pointers below are current.

## From a script

The default `AtmosSimulation` is already global: a cubed-sphere grid with a
decaying temperature profile (see [Your First Simulation](first_simulation.md)).
The quickest physically interesting global run is the aquaplanet preset,
which pairs a moist model with an aquaplanet surface:

```julia
import ClimaAtmos as CA

simulation = CA.Presets.aquaplanet(Float32; t_end = "1days")
CA.solve_atmos!(simulation)
```

Customize it through the [grid](grids.md) (`SphereGrid`), the
[setup](setups.md), and the `AtmosModel` components. Horizontal resolution is
set by `h_elem`, the number of spectral elements per cube edge: `h_elem = 6`
corresponds to about 550 km, 16 to about 206 km, and 30 to about 110 km.

## From a configuration file

The aquaplanet configurations in `config/model_configs/` (e.g.,
`prognostic_edmfx_aquaplanet.yml`, `aquaplanet_equil_allsky_gw_raw.yml`) and
the production-oriented configurations in `config/longrun_configs/` (e.g.,
`longrun_aquaplanet_allsky_progedmf_1M.yml`, `amip_target.yml`) are the best
starting points:

```julia
import ClimaAtmos as CA

config = CA.AtmosConfig(
    "config/model_configs/prognostic_edmfx_aquaplanet.yml";
    job_id = "my_aquaplanet",
)
simulation = CA.AtmosSimulation(config)
CA.solve_atmos!(simulation)
```

Many model configurations already pin their grid and timestep (the file above
sets a reduced vertical resolution so it runs quickly). Combining a numerics
file from `config/common_configs/` with a model configuration, as described
in [Creating custom configurations](configuration.md), applies only to model
configurations that do not fix those keys themselves; later files override
earlier ones key by key.

Output lands in `output/<job_id>` (see
[Loading and Visualizing Output](visualizing_output.md)). The tested global
production configurations can be found in `config/longrun_configs/`. For GPU runs, see
[Running on GPUs and MPI](gpu_and_mpi.md).
