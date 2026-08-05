# Loading and Visualizing Output

A simulation that [saves diagnostics](diagnostics.md) writes NetCDF (`.nc`)
files to its output directory. The recommended tool for reading them is
[ClimaAnalysis.jl](https://clima.github.io/ClimaAnalysis.jl/stable/), which
loads variables, reduces them (averages, slices, windows), and plots them
through Makie.

ClimaAnalysis and a Makie backend are separate packages; add them to the
environment you work in (`pkg> add ClimaAnalysis CairoMakie`).

## Loading the output

Point `SimDir` at the output directory. With the default directory style,
the latest run is behind the `output_active` link:

```julia
using ClimaAnalysis

simdir = SimDir("output/my_script_run/output_active")
```

`SimDir` catalogs every variable the simulation wrote; `get` retrieves one as
an `OutputVar` by its short name:

```julia
ta = get(simdir, "ta")   # air temperature
```

## Reducing dimensions

`OutputVar`s reduce with functions such as `average_time`, `average_lon`,
`average_lat`, and `slice`:

```julia
# Time average over the simulation, then zonal mean
ta_avg = average_lon(average_time(ta))

# Vertical profile of the zonal mean at the equator
ta_profile = slice(ta_avg, lat = 0.0)
```

Each reduction returns a new `OutputVar` with the reduced dimension removed
and the attributes updated, so results stay self-describing.

## Plotting

The `ClimaAnalysis.Visualize` module plots `OutputVar`s through a Makie
backend such as `CairoMakie` or `GLMakie`:

```julia
import ClimaAnalysis.Visualize as viz
import CairoMakie

fig = CairoMakie.Figure()
viz.plot!(fig, ta_avg)   # latitude-height heatmap of the zonal-mean temperature
CairoMakie.save("zonal_mean_temperature.png", fig)
```

`viz.plot!` picks a heatmap or a line plot from the number of remaining
dimensions; `viz.heatmap2D!`, `viz.sliced_heatmap!`, and
`viz.line_plot1D!` give explicit control. The
[ClimaAnalysis.jl documentation](https://clima.github.io/ClimaAnalysis.jl/stable/)
covers the full feature set, including windowing, units, and comparison
against observations.

## Where the files come from

[ClimaDiagnostics.jl](https://clima.github.io/ClimaDiagnostics.jl/stable/)
computes the diagnostics during the simulation and writes the NetCDF files;
ClimaAnalysis reads and interprets them. To add diagnostic outputs or change
how variables are interpolated to the output grid, see
[Computing and Saving Diagnostics](diagnostics.md).
