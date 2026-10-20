# Example: Moist Baroclinic Wave

This example runs a complete global simulation in a few minutes on a laptop:
the moist baroclinic wave of [Ullrich2014](@cite), with the moisture
profile of the 2016 Dynamical Core Model Intercomparison Project. The initial
state holds a midlatitude jet in each hemisphere, in balance with the pressure
and temperature fields, plus a small wind perturbation centered at 20°E, 40°N.
Over about ten days the perturbation grows into a chain of low- and
high-pressure systems, and water vapor condenses and rains out along the fronts
between them.

The page builds the model, runs it for ten days, and plots the result. It runs
during the documentation build, so the figure at the end comes from the code
shown here. To run it yourself, add the packages it uses to your
environment (`pkg> add ClimaAtmos ClimaAnalysis CairoMakie`) and run the blocks
in order, in the REPL or as a script.

## Building the model

```@example moist_bw
using Logging # hide
Logging.disable_logging(Logging.Info) # hide
import ClimaAtmos as CA

FT = Float32
grid = CA.SphereGrid(FT; h_elem = 6, z_elem = 10)
model = CA.AtmosModel(
    grid;
    setup = CA.Setups.MoistBaroclinicWave(; deep_atmosphere = true),
    microphysics_model = CA.EquilibriumMicrophysics0M(),
    hyperdiff = CA.cam_se_hyperdiffusion(FT),
    disable_surface_flux_tendency = true,
)
nothing # hide
```

Each piece sets one part of the case:

  - `SphereGrid(FT; h_elem = 6, z_elem = 10)` is a cubed sphere with 6
    elements along each panel edge, about 550 km between grid points, and 10
    levels up to 30 km. [Running Global Simulations](global_simulations.md)
    explains how `h_elem` sets the resolution.
  - [`MoistBaroclinicWave`](@ref ClimaAtmos.Setups.MoistBaroclinicWave) sets
    the initial state. `deep_atmosphere = true` builds it for the
    deep-atmosphere equations, which `SphereGrid` uses by default.
  - [`EquilibriumMicrophysics0M`](@ref ClimaAtmos.EquilibriumMicrophysics0M)
    adds moisture with the simplest microphysics: cloud condensate is
    diagnosed from saturation, and precipitation leaves the domain as soon as
    it forms. See [Microphysics](microphysics.md).
  - `cam_se_hyperdiffusion` sets the numerical smoothing to the coefficients
    the tested YAML configurations use. See [Hyperdiffusion](hyperdiffusion.md).
  - `disable_surface_flux_tendency = true` stops the surface from exchanging
    heat, moisture, and momentum with the air, so the wave grows from the
    initial state alone.

## Running the simulation

The simulation writes two diagnostics once per simulated day: the pressure
`pfull` and the specific humidity `hus`. `default = false` skips the built-in
diagnostic set, which keeps the run fast and the output small.
`interpolation_num_points` sets the longitude-latitude grid the output is
written on, here 2° spacing. The vertical entry is ignored, since the output
stays on the model levels:

```@example moist_bw
diagnostics = CA.DiagnosticsConfig(;
    default = false,
    additional = (
        "pfull" => (; period = "1days", reduction = "inst"),
        "hus" => (; period = "1days", reduction = "inst"),
    ),
    interpolation_num_points = (180, 90, 10),
)
simulation = CA.AtmosSimulation(
    model;
    dt = "1200secs",
    t_end = "10days",
    update_cache_every = "step",
    diagnostics,
    job_id = "moist_baroclinic_wave",
    output_dir = mktempdir(), # hide
)
CA.solve_atmos!(simulation)
nothing # hide
```

The 1200 s timestep is longer than the 450 s that the CI configuration uses at
this resolution, which keeps the run to a few minutes. If a change to the case
makes the run unstable, return to 450 s. `update_cache_every = "step"`
recomputes derived quantities such as pressure and temperature once per
timestep instead of at every stage of the time integrator, as the tested YAML
configurations do. The first run in a session spends a few minutes compiling
before it starts stepping, and the log reports progress as it runs. The output
goes to `output/moist_baroclinic_wave/output_active`; see
[Where output goes](@ref) for the layout.

## Plotting the wave

[ClimaAnalysis.jl](https://clima.github.io/ClimaAnalysis.jl/stable/) reads the
NetCDF output. `viz.plot!` cuts each variable at the nearest available time and
height and draws a longitude-latitude map:

```@example moist_bw
import ClimaAnalysis
import ClimaAnalysis.Visualize as viz
import CairoMakie

simdir = ClimaAnalysis.SimDir(simulation.output_dir)
pfull = get(simdir; short_name = "pfull", reduction = "inst")
hus = get(simdir; short_name = "hus", reduction = "inst")

day = 86400 # seconds
fig = CairoMakie.Figure(size = (900, 750))
viz.plot!(fig, pfull; time = 10 * day, z = 0, p_loc = (1, 1))
viz.plot!(fig, hus; time = 10 * day, z = 1800, p_loc = (2, 1))
fig
```

Both panels show day 10. The top panel is the pressure on the lowest model
level, about 250 m above the surface, where the lows and highs of the wave
sit along the northern jet, downstream (east) of the initial perturbation. The
bottom panel is the specific humidity about 1.8 km up, where the wave draws moist
subtropical air poleward. The southern hemisphere starts with no perturbation,
so its jet stays close to zonal.

## Next steps

  - Raise `h_elem` to 16 with `dt = "120secs"` to resolve the fronts. This
    costs about 70 times as much, so run it on a GPU; see
    [Running on GPUs and MPI](gpu_and_mpi.md), and
    [Running Global Simulations](global_simulations.md) for tested pairs of
    resolution and timestep.
  - Swap the setup for
    [`DryBaroclinicWave`](@ref ClimaAtmos.Setups.DryBaroclinicWave) and drop
    `microphysics_model` to compare against the dry wave.
  - [Loading and Visualizing Output](visualizing_output.md): averages, slices,
    and other plots of the output.
  - [Available Diagnostics](available_diagnostics.md): the variables a
    simulation can write.
