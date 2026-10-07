# COSP Satellite Simulator

COSP turns the model's cloud and precipitation fields into what a satellite
instrument would have measured, so that a simulation can be compared with the
satellite record on the instrument's own terms rather than through
model-native cloud fraction. ClimaAtmos implements the CloudSat branch: a
94 GHz cloud radar, whose reflectivity depends on hydrometeor size and
concentration and is attenuated by gases and by the hydrometeors themselves
[Bodas-Salcedo2011](@cite).

The simulator is off by default. It is switched on by giving the `dt_subcol`
configuration key a finite value, which sets how often it runs, and it is
configured through two further keys:

| Key                 | Default            | Meaning                                                            |
|:------------------- |:------------------ |:------------------------------------------------------------------ |
| `dt_subcol`         | `"Inf"` (off)      | Interval between simulator calls                                   |
| `cosp_n_subcolumns` | `100`              | Number of subcolumns sampled per grid column                       |
| `cosp_overlap`      | `"maximum_random"` | Cloud overlap assumption: `maximum`, `random`, or `maximum_random` |

The CloudSat outputs need the hydrometeor masses and number concentrations of
the one-moment scheme, so they require `microphysics_model: "1M"`. With `"2M"`
the subcolumns are still generated but no reflectivity is computed, and the
outputs are left undefined; any other microphysics model is rejected when the
simulator runs.

## Why subcolumns

A grid column carries one cloud fraction per level and says nothing about how
the cloudy parts of neighboring levels line up. A radar beam, by contrast,
passes through one realization of that overlap. The simulator therefore
samples each grid column into ``N`` subcolumns, each of which is either fully
cloudy or clear at every level, such that the average over subcolumns
reproduces the grid-mean cloud fraction and the vertical alignment follows the
chosen overlap assumption. The sampler is the SCOPS generator of COSP: a
threshold recurrence that proceeds from the model top toward the surface,
placing cloud where a per-level random draw falls below the cloud fraction.
Under `maximum` overlap, cloud in adjacent levels is stacked; under `random`
it is independent; `maximum_random` stacks contiguous cloudy layers and
decorrelates across clear gaps.

Precipitation is sampled the same way, from selectors shared with the cloud
draw so that rain and snow fall from the subcolumns that hold cloud. The
random seed is fixed, so the subcolumns are reproducible across calls and
across restarts.

Subcolumns are generated and used one at a time. The reflectivity of each
is computed and folded into the running statistics before the next is drawn,
so the memory footprint is that of one column, not of ``N`` columns.

## From hydrometeors to reflectivity

For each subcolumn the simulator assigns the cloud liquid, cloud ice, rain, and
snow contents of the grid mean to the levels that the sampler marked cloudy or
precipitating, then diagnoses a characteristic particle size for each species
from the grid-mean state: a monodisperse radius for cloud liquid, from the
prescribed droplet number, and the inverse Marshall–Palmer slope for cloud ice,
rain, and snow, from the same size-distribution parameters the microphysics
uses.

Radar optics follow Quickbeam, the COSP radar module. The one-way gas
attenuation at 94 GHz is computed once per grid column from temperature,
pressure, and water vapor, and integrated from the model top to give the
two-way path attenuation at each level. The hydrometeor backscatter and
attenuation are then evaluated per subcolumn, the path attenuation accumulated
downward from the top, and the result expressed as the attenuated equivalent
reflectivity ``Z_e`` in dBZ.

## Statistics

Three quantities are accumulated over the subcolumns of each grid column:

| Quantity        | Definition                                                                                                         |
|:--------------- |:------------------------------------------------------------------------------------------------------------------ |
| `cloudsat_tcc`  | Percentage of subcolumns with reflectivity in the inclusive ``[-30, 10]`` dBZ window at any level                  |
| `cloudsat_tcc2` | The same, ignoring the lowest 1 km above the surface, which CloudSat cannot see through ground clutter             |
| `cfadDbze94`    | The reflectivity histogram (CFAD) on the model vertical grid, normalized by the number of subcolumns at each level |

The detection window is CloudSat's: ``-30`` dBZ is the minimum detectable
signal, and ``10`` dBZ the level above which the retrieval saturates. The CFAD
bins follow COSP's `hist1D` convention, with lower edges inclusive and upper
edges exclusive.

`cloudsat_tcc` and `cloudsat_tcc2` are available as output diagnostics;
requesting either in a run without COSP or without one-moment microphysics
raises an error naming what is missing. `cfadDbze94` is accumulated in the
cache but not yet exposed as a diagnostic.

## Where this is implemented

| Concept                                       | Source                                                                                                                                                                                                                       |
|:--------------------------------------------- |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Subcolumn generation (SCOPS)                  | [cosp/subcol.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/cosp/subcol.jl)                                                                                                                                        |
| Precipitation and hydrometeor subcolumns      | [cosp/prec_subcol.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/cosp/prec_subcol.jl), [cosp/hydrometeor_subcol.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/cosp/hydrometeor_subcol.jl)               |
| Particle sizes, gas and hydrometeor optics    | [cosp/cloudsat_optics.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/cosp/cloudsat_optics.jl)                                                                                                                      |
| Attenuated reflectivity                       | [cosp/cloudsat_reflectivity.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/cosp/cloudsat_reflectivity.jl)                                                                                                          |
| Cloud cover and CFAD accumulation             | [cosp/cloudsat_cloud_fraction.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/cosp/cloudsat_cloud_fraction.jl), [cosp/cloudsat_cfad.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/cosp/cloudsat_cfad.jl) |
| Driver, streaming loop, microphysics dispatch | [cosp/cloudsat.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/cosp/cloudsat.jl)                                                                                                                                    |
| Callback at `dt_subcol`                       | [cosp/callbacks.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/cosp/callbacks.jl)                                                                                                                                  |
| Diagnostics                                   | [diagnostics/cosp_diagnostics.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/diagnostics/cosp_diagnostics.jl)                                                                                                      |
| Model type                                    | [`ClimaAtmos.COSPModel`](@ref)                                                                                                                                                                                               |

See [Configuration Options](configuration_options.md) for the keys named here
and [Computing and saving diagnostics](diagnostics.md) for requesting the
outputs.
