ClimaAtmos.jl Release Notes
============================

Main
-------

v0.26.2
-------
- Limit temperature input to RRTMGP within the lookup table range.
  PR [#3124](https://github.com/CliMA/ClimaAtmos.jl/pull/3124)
  
v0.26.1
-------
- Updated RRTMGP compat from 0.15 to 0.16
  PR [#3114](https://github.com/CliMA/ClimaAtmos.jl/pull/3114)
- ![][badge-🔥behavioralΔ] Removed the filter for shortwave radiative fluxes. 
  PR [#3099](https://github.com/CliMA/ClimaAtmos.jl/pull/3099).

v0.26.0
-------
- ![][badge-💥breaking] Add precipitation fluxes to 1M microphysics output.
  Rename col_integrated_rain (and snow) to surface_rain_flux (and snow)
  PR [#3084](https://github.com/CliMA/ClimaAtmos.jl/pull/3084).

v0.25.0
-------
- ![][badge-💥breaking] Remove reference state from the dycore and the
  relevant config. PR [#3074](https://github.com/CliMA/ClimaAtmos.jl/pull/3074).
- Make prognostic and diagnostic EDMF work with 1-moment microphysics on GPU
  PR [#3070](https://github.com/CliMA/ClimaAtmos.jl/pull/3070)
- Add precipitation heating terms for 1-moment microphysics
  PR [#3050](https://github.com/CliMA/ClimaAtmos.jl/pull/3050)

v0.24.2
-------
- ![][badge-🔥behavioralΔ] Fixed incorrect surface fluxes for uh. PR [#3064]
  (https://github.com/CliMA/ClimaAtmos.jl/pull/3064).

v0.24.1
-------

v0.24.0
-------
- ![][badge-💥breaking]. CPU/GPU runs can now share the same yaml files. The driver now calls `AtmosConfig` via `(; config_file, job_id) = ClimaAtmos.commandline_kwargs(); config = ClimaAtmos.AtmosConfig(config_file; job_id)`, which recovers the original behavior. PR [#2994](https://github.com/CliMA/ClimaAtmos.jl/pull/2994), issue [#2651](https://github.com/CliMA/ClimaAtmos.jl/issues/2651).
- Move config files for gpu jobs on ci to config/model_configs/.
  PR [#2948](https://github.com/CliMA/ClimaAtmos.jl/pull/2948).

v0.23.0
-------
- ![][badge-✨feature/enhancement]![][badge-💥breaking]. Use
  [ClimaUtilities](https://github.com/CliMA/ClimaUtilities.jl) for
  `TimeVaryingInputs` to read in prescribed aerosol mass concentrations. This PR
  is considered breaking because it changes `AtmosCache` adding a new field,
  `tracers`. PR [#2815](https://github.com/CliMA/ClimaAtmos.jl/pull/2815).

- ![][badge-✨feature/enhancement]![][badge-💥breaking]. Use
    [ClimaUtilities](https://github.com/CliMA/ClimaUtilities.jl) for
    `OutputPathGenerator` to handle where the output of a simulation should be
    saved. Previously, the output was saved to a folder named `$job_id`. Now, it
    is saved to `$job_id/output-active`, where `output-active` is a link that
    points to `$job_id/output-XXXX`, with `XXXX` a counter that increases ever
    time a simulation is run with this output directory. PR
    [#2606](https://github.com/CliMA/ClimaAtmos.jl/pull/2606).

v0.22.1
-------
- ![][badge-🚀performance] Reduced the number of allocations in the NetCDF
  writer. PRs [#2772](https://github.com/CliMA/ClimaAtmos.jl/pull/2772),
  [#2773](https://github.com/CliMA/ClimaAtmos.jl/pull/2773).
- Added a new script, `perf/benchmark_netcdf_io.jl` to test IO performance for
  the NetCDF writer. PR [#2773](https://github.com/CliMA/ClimaAtmos.jl/pull/2773).

<!--

Contributors are welcome to begin the description of changelog items with badge(s) below. Here is a brief description of when to use badges for a particular pull request / set of changes:

 - 🔥behavioralΔ - behavioral changes. For example: a new model is used, yielding more accurate results.
 - 🤖precisionΔ - machine-precision changes. For example, swapping the order of summed arguments can result in machine-precision changes.
 - 💥breaking - breaking changes. For example: removing deprecated functions/types, removing support for functionality, API changes.
 - 🚀performance - performance improvements. For example: improving type inference, reducing allocations, or code hoisting.
 - ✨feature - new feature added. For example: adding support for a cubed-sphere grid
 - 🐛bugfix - bugfix. For example: fixing incorrect logic, resulting in incorrect results, or fixing code that otherwise might give a `MethodError`.

-->

[badge-🔥behavioralΔ]: https://img.shields.io/badge/🔥behavioralΔ-orange.svg
[badge-🤖precisionΔ]: https://img.shields.io/badge/🤖precisionΔ-black.svg
[badge-💥breaking]: https://img.shields.io/badge/💥BREAKING-red.svg
[badge-🚀performance]: https://img.shields.io/badge/🚀performance-green.svg
[badge-✨feature/enhancement]: https://img.shields.io/badge/feature/enhancement-blue.svg
[badge-🐛bugfix]: https://img.shields.io/badge/🐛bugfix-purple.svg
