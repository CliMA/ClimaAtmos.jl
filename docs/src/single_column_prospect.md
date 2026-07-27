# Single Column Models

## Idealized cases

`ClimaAtmos.jl` supports several canonical test cases that are run in a single column model designed to verify how well PROSPCT (EDMF) is able to reproduce each of the convective schemes. These cases include variants of `bomex`, `dycoms`, `rico`, `soares`, `gabls`, `gate`,and `trmm` and can be found in the `config/model_configs` directory. The purpose of each simulation is summarized in the following table:

| Abbreviation | Long Name                                            | Cloud Regime                  | Reference                                                                                                                   |
|:------------ |:---------------------------------------------------- |:----------------------------- |:--------------------------------------------------------------------------------------------------------------------------- |
| BOMEX        | Barbados Oceanographic and Meteorological Experiment | Marine Cumulus                | [Siebesma et al. (2003)](https://doi.org/10.1175/1520-0469(2003)60%3C1201:ALESIS%3E2.0.CO;2)                                |
| DYCOMS       | Dynamics and Chemistry of Marine Stratocumulus       | Marine Stratocumulus          | [Stevens et al. (2005)](https://doi.org/10.1175/MWR2930.1), [Ackerman et al. (2009)](https://doi.org/10.1175/2008MWR2582.1) |
| RICO         | Rain in Cumulus over the Ocean                       | Rainy Cumulus                 | [Rauber et al. (2007)](https://doi.org/10.1175/BAMS-88-12-1912)                                                             |
| SOARES       | Shallow Cumulus Convection                           | Shallow Cumulus               | [Soares et al. (2004)](https://doi.org/10.1256/qj.03.223)                                                                   |
| GABLS        | GEWEX Atmospheric Boundary Layer Study               | Dry Convective Boundary Layer | [Kosović & Curry (2000)](https://doi.org/10.1175/1520-0469(2000)057%3C1052:ALESSO%3E2.0.CO;2)                               |
| TRMM         | Tropical Rainfall Measuring Mission                  | Deep Convection               | [Grabowski et al. (2006)](https://doi.org/10.1256/qj.04.147)                                                                |

For example, to run the BOMEX test case with the
[configuration driver](@ref "Running from a cloned repository"), execute the following:

```bash
julia --project=.buildkite .buildkite/ci_driver.jl \
    --config_file config/model_configs/prognostic_edmfx_bomex_column.yml \
    --job_id bomex
```

It may also be helpful to run in interactive mode to be able to examine the simulation
object, debug, and develop the code further. To enter debug mode run
`julia --project=.buildkite` and then in the REPL run:

```julia
using Revise # if you are developing ClimaAtmos
import ClimaAtmos as CA

# get the configuration arguments
config = CA.AtmosConfig("config/model_configs/prognostic_edmfx_bomex_column.yml")
simulation = CA.get_simulation(config)
sol_res = CA.solve_atmos!(simulation) # run the simulation
```

## Externally-Driven Single Column Models

Currently several externally driven single column setups are supported in `ClimaAtmos.jl`: `GCM` driven, `ReanalysisTimeVarying`, `ReanalysisMonthlyAveragedDiurnal`, and ARM VARANAL. Externally-driven means that the model is initialized and forced with data coming from a different simulation or analysis product. This differs from setups like, for example, BOMEX or SOARES which have steady forcing and low domain tops (~4km) or functional forcing, respectively. They have been developed specifically for the purpose of model calibration by recreating statistics that are close to either LES, for the `GCM` driven case only, or to observations.

### GCM-Driven Case

For the `GCM` driven case we can run the experiment using the config file `config/model_configs/prognostic_edmfx_gcmdriven_column.yml` by running:

```bash
julia --project=.buildkite .buildkite/ci_driver.jl \
    --config_file config/model_configs/prognostic_edmfx_gcmdriven_column.yml \
    --job_id gcm_driven_scm
```

In the config the following settings are particularly important:

```YAML
initial_condition: "GCM"
external_forcing: "GCM"
external_forcing_file: artifact"cfsite_gcm_forcing"/HadGEM2-A_amip.2004-2008.07.nc
cfsite_number : "site23"
surface_setup: "GCM"
```

Here we must set all of `initial_condition`, `external_forcing` and `surface_setup` to be `GCM` as each component requires information from the external file. The `external_forcing_file` and `cfsite_number` together determine the temperature, specific humidity, and wind as well as horizontal and vertical advection profiles that drive the simulation, and can be set to a local file path as opposed to using the artifact. Radiation and surface temperature are also specified. Here the forcing file, an example of which is stored in the artifact, contains groups for each `cfsite` to drive the simulation. See [Shen et al. 2022](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021MS002631) for more information.

### ARM VARANAL Case (SGP)

The ARM VARANAL setup drives a single column at the SGP Central Facility with
time-varying profiles and tendencies from the ARM Variational Analysis product
(`sgp60varanarucC1.c1`). Forcing includes horizontal advection, large-scale
subsidence (from `omega`), nudging toward observed T/q/u/v, prescribed surface
fluxes (LH/SH), and time-varying skin temperature. Monthly files are available
from the [ARM Data Center](https://adc.arm.gov/discovery/).

```bash
julia --project=.buildkite .buildkite/ci_driver.jl \
    --config_file config/model_configs/prognostic_edmfx_armvaranal_column.yml \
    --job_id scm_varanal
```

Key config entries (edit `external_forcing_file`, `start_date`, and `t_end` to
pick a sub-period within the monthly file):

```YAML
initial_condition: "ARMVARANAL"
external_forcing_file: artifact"arm_sgp_varanal_forcing"/sgp60varanarucC1.c1.20100901.000000.cdf
start_date: "20100918"
t_end: "4days"
```

The `ARMVARANAL` case is converted to the ClimaColumn schema
(`ColumnDatasets.VaranalFiles.to_climacolumn`) and then run through the generic
`ForcingFromFile` path; the forcing, prescribed surface fluxes, and insolation
are supplied by the setup, so `external_forcing` is left unset.

Suggested run periods at SGP (interesting regimes):

| Period          | File month | `start_date` | `t_end` | Notes                                                                                         |
|:--------------- |:---------- |:------------ |:------- |:--------------------------------------------------------------------------------------------- |
| Sep 18–22, 2010 | `20100901` | `20100918`   | `4days` | Default. Clear → convective transition; cold-front passage days 2–3. Good diurnal-cycle test. |
| Aug 8–10, 2010  | `20100801` | `20100808`   | `3days` | Deep convection, single cell/pulse.                                                           |

Forcing and obs for CI auto-download (`arm_sgp_varanal_forcing`, `arm_sgp_varanal_obs`).
Obs comparisons use three ARM products at SGP:

| Product            | ARM name                    | Used for                             |
|:------------------ |:--------------------------- |:------------------------------------ |
| Interpolated Sonde | `sgpinterpolatedsondeC1.c1` | θ, RH, q, u, v profiles (~3–6 hr)    |
| BEATM              | `sgparmbeatmC1.c1`          | Surface T, precip, SH/LH             |
| CLDRAD             | `sgparmbecldradC1.c1`       | Cloud fraction, SW/LW radiation, LWP |

`arm_sgp_varanal_obs_full` (~43 GB, HPC only) is registered but not used by
ClimaAtmos and will not download automatically.

#### Running outside the CI period

CI covers September 2010 forcing and Sep 18–22 obs. For other periods:

 1. **Forcing** — set `external_forcing_file` to the monthly VARANAL `.cdf` for
    that month.
 2. **Observations** — in `~/.julia/artifacts/Overrides.toml`, point
    `arm_sgp_varanal_obs` at the full obs directory on disk.

### Reanalysis-Driven Case

#### Matched ERA5 Trajectory

The `ReanalysisTimeVarying` case extends the `GCM` driven case by providing support for single-column simulations which resolve the diurnal cycle, can be run at any site globally, and uses reanalysis to drive the simulation, allowing for calibration of EDMF to earth-system observations in the single-column setting. This feature was found to be needed to address biases in calibration arising from correlation between time-of-day and cloud liquid water path over the tropical Pacific. For this simulation we again highlight similar arguments in the config file:

```YAML
initial_condition: "ReanalysisTimeVarying"
start_date: "20070701"
site_latitude: 17.0
site_longitude: -149.0
```

The ReanalysisTimeVarying initial condition generates a column forcing file for
the requested site and dates (regridded from the global ERA5 archive, schema
below) and hands it to the generic `ForcingFromFile` setup, which takes the
initial condition, external forcing, surface skin temperature, and insolation
from that one file (surface fluxes are computed interactively by Monin–Obukhov).
Setting `external_forcing: "ReanalysisTimeVarying"` as well is still accepted but
no longer needed. You give the site and dates directly rather than a file path
because the file is generated on demand from the version-pinned ERA5 archive
(stored through `ClimaArtifacts` for reproducibility): start_date is YYYYMMDD,
site_latitude in degrees (-90...90) and site_longitude in (-180...180). Note that
artifact-backed ERA5 data is currently available only for the tropical Pacific
in the first 5 days of July 2007.

!!! note

    Depending on the amount of smoothing and data resolution, points near the boundaries will throw index errors. With default settings, users should stay at least 5 points away from the poles (1° for ERA5 data) for smoothing (4 points) and gradients (one extra point).

The data is generated by downloading from ECMWF and further documentation for ERA5 data download can be found either directly on the ECMWF page and `ClimaArtifacts`. Note that the profiles, surface temperature, and surface fluxes cannot be obtained from a single request and so together we need 3 files for all the data. We include a script at `src/config/era5_observations_to_forcing_file.jl` which extracts the profiles and computes the tendencies needed for the simulation from the raw ERA5 reanalysis files. We store the observations directly into an artifact `era5_hourly_atmos_processed` to eliminate the need to reprocess specific sites and locations. This setup means that users are free to choose sites globally at any time at which ERA5 data is available. Unfortunately, global hourly renanalysis is too large to store in an artifact and so we have currently only provided support for the first 5 days of July 2007 in the tropical Pacific, stored in `era5_hourly_atmos_raw`, only available on the `clima` and Caltech `HPC` servers. The test case can be run using:

```bash
julia --project=.buildkite .buildkite/ci_driver.jl \
    --config_file config/model_configs/prognostic_edmfx_tv_era5driven_column.yml \
    --job_id era5driven
```

#### Monthly Averaged Forcing

As the matched ERA5 trajectory is data intensive, requiring downloads for each day, we have also implemented an external forcing dispatch to repeat a specific day of data indefinitely. This setup is ideal for use with monthly averaged ERA5 data by hour of day and can be used to calibrate to monthly statistics. The setup is similar, except we change the flag for `external_forcing` to indicate that we want to repeat data:

```YAML
initial_condition: "ReanalysisTimeVarying"
external_forcing: "ReanalysisMonthlyAveragedDiurnal"
surface_setup: "ReanalysisTimeVarying"
surface_temperature: "ReanalysisTimeVarying"
start_date: "20070701"
site_latitude: 17.0
site_longitude: -149.0
```

```bash
julia --project=.buildkite .buildkite/ci_driver.jl \
    --config_file config/model_configs/prognostic_edmfx_diurnal_scm_imp.yml \
    --job_id bomex
```

#### Running the Reanalysis-driven cases at different times and locations

You need 3 separate files with specific variables and naming convention for the data processing script to work.

 1. Hourly profiles with variables, following ERA5 naming convention, including `t`, `q`, `u`, `v`, `w`, `z`, `clwc`, `ciwc`. This file should be stored in the appropriate artifacts directory, named `"forcing_and_cloud_hourly_profiles_$(start_date).nc"` for `ReanalysisTimeVarying` and `monthly_diurnal_profiles_$start_date).nc` for `ReanalysisMonthlyAveragedDiurnal` where `start_date` should specify the date data starts on formatted YYYYMMDD. We require `clwc` and `ciwc` profiles because these are typical targets for calibration but are not needed to run the simulation directly.
 2. Instantaneous variables, including surface temperature `ts` which should be stored in `"hourly_inst_$(start_date).nc"` for `ReanalysisTimeVarying` and `monthly_diurnal_inst` for `ReanalysisMonthlyAveragedDiurnal`.
 3. Accumulated variables, including surface sensible and latent heat fluxes, `hfls` and `hfss`, which should be stored in `"hourly_accum_$(start_date).nc"` for `ReanalysisTimeVarying` and `monthly_diurnal_accum` for `ReanalysisMonthlyAveragedDiurnal`. These need to be divided by the appropriate time resolution, which for hourly data is 3600 and for daily and monthly data is 86400 (not a typo see [here](https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#ERA5:datadocumentation-Monthlymeans)).

##### On HPC/Clima

To run locations already in the artifact, e.g., sites in the tropical Pacific in the first 5 days of July 2007 the config file will work out of the box. To run other locations or times please follow the steps for local.

##### On local

To run the simulation on a local machine you will need to first download the reanalysis data from ECMWF, ensuring that you have all the required variables. This will be stored in 3 separate files which should be all placed in the same directory. The user edit `.julia/artifacts/Overrides.toml` to point the `era5_hourly_atmos_raw` artifact point to the folder where the data is stored. For the raw data and location for processed files you'll need to specify the path where the data is stored and where to store the files as follows:

```bash
8234def2ead82e385a330a48ed2f0c030e434065 = "/some/random/path/raw_data_dir" # for raw data
a1a465e8d237d78bef1e6d346054da395787a9f9 = "/some/random/path/processed_files" # for storing
```

### Column forcing datasets

Adding a new externally-driven column case in a supported format requires no
source code: point the config at the file and the `ForcingFromFile` setup
builds the case (initial condition, external forcing, surface temperature, and
insolation) from it.

```YAML
initial_condition: "ForcingFromFile"
external_forcing_file: /path/to/my_case_forcing.nc
start_date: "20200101"
config: "column"
```

To use a forcing file with a different (analytic) initial condition, set
`external_forcing: "ForcingFromFile"` instead and keep your `initial_condition`.

The reader consumes one format: the native `ClimaColumn` schema (below),
written by the ERA5 generator and the target for hand-made case files. A file
that is not a conforming ClimaColumn file is a loud error at construction. A
stale cached file (e.g. an ERA5 forcing file written by an older version in a
different on-disk layout) is regenerated on demand from the source rather than read.

The forcing is composed from explicit per-process terms
(`HorizontalAdvection`, `VerticalFluctuation`, `Nudging`, `Subsidence`). The
default composition is all four. A runscript can narrow or reshape it without
any YAML option:

```julia
forcing = ClimaAtmos.ExternalDrivenTVForcing(
    forcing_file;
    forcing = (ClimaAtmos.HorizontalAdvection(),),   # advection only
)
model = ClimaAtmos.AtmosModel(; external_forcing = forcing)
simulation = ClimaAtmos.AtmosSimulation{Float64}(; model, setup, grid)
```

When the same file also supplies the initial condition, pass the terms to the
setup's `forcing` slot: `ForcingFromFile(...; forcing = (...,))`.

Per-variable relaxation timescales and height-dependent masks compose as
multiple `Nudging` terms (`Nudging(:ta; timescale, mask = z -> ...)`).

Surface-temperature and insolation inputs are required only when the model uses
them (`ExternalTemperature` needs `ts`; `ExternalTVInsolation` needs
`coszen`/`rsdt`), so runscripts need not track those separately.

The built-in file-driven cases wire these defaults (a runscript can override any
slot):

| Case                                                                          | Large-scale forcing (default)                                                                  | Surface / insolation (default)                                                                                                                |
|:----------------------------------------------------------------------------- |:---------------------------------------------------------------------------------------------- |:--------------------------------------------------------------------------------------------------------------------------------------------- |
| `ForcingFromFile`, `ReanalysisTimeVarying` (ERA5 time-varying)                | `default_forcing_terms()`: HAdv + VertFluc + Nudge(`ta`,`hus`) + Nudge(`ua`,`va`) + Subsidence | MO (`z0 = 1e-4`); `ExternalTemperature` (file `ts`); `ExternalTVInsolation` (file `coszen`/`rsdt`)                                            |
| `ReanalysisMonthlyAveragedDiurnal` (ERA5 monthly, set via `external_forcing`) | same terms, but periodic time interpolation (repeats the one-day file)                         | MO (`z0 = 1e-4`); `ExternalTemperature`; `ExternalTVInsolation`                                                                               |
| `ARMVARANAL`                                                                  | HAdv + Nudge(`ta`,`hus`) + Nudge(`ua`,`va`) + Subsidence (no VertFluc)                         | MO (`z0 = 0.05`, `ustar = 0.28`) + `FileHeatFluxes` when `hfls`/`hfss` present; `ExternalTemperature`; `TimeVaryingInsolation` (site lat/lon) |

```@docs
ClimaAtmos.ExternalDrivenTVForcing
ClimaAtmos.AbstractForcingTerm
ClimaAtmos.HorizontalAdvection
ClimaAtmos.VerticalFluctuation
ClimaAtmos.Subsidence
ClimaAtmos.Nudging
```

#### Nonstandard forcing behavior from a runscript

The composed terms cover the standard file-driven processes. They are not
intended to encode every forcing experiment. For a wholly new tendency term,
an in-memory data source, or state-dependent behavior, define a small forcing
type in the runscript and extend the forcing interface. This keeps the
experiment visible next to the simulation construction and does not require a
new YAML option or a change under `src/`.

```julia
import ClimaAtmos as CA
import ClimaAtmos: external_forcing_cache, external_forcing_tendency!

struct MyCaseForcing{D, M}
    data::D
    mask::M
end

function external_forcing_cache(Y, forcing::MyCaseForcing, params, start_date)
    # Allocate model-grid fields and prepare any interpolation objects here.
    # `forcing.data` may be a ColumnDataset, arrays, or another runscript-owned
    # object. The returned value is available as `p.external_forcing` below.
    return (; mask = forcing.mask)
end

function external_forcing_tendency!(Yₜ, Y, p, t, ::MyCaseForcing)
    cache = p.external_forcing
    # Evaluate the data at `t` and add this case's tendencies to Yₜ here.
    # The implementation may use Y, p.precomputed, p.params, and cache.
    return nothing
end

forcing = MyCaseForcing(case_data, case_mask)
model = CA.AtmosModel(; external_forcing = forcing)
simulation = CA.AtmosSimulation{Float64}(; model, setup, grid)
```

The two methods are the complete interface when the tendency can evaluate its
data at the current model time. A custom forcing can reuse the standard file
reader by storing `CA.ColumnDatasets.ColumnDataset(path)` and calling the
`ColumnDatasets.column_timevaryinginputs` or `ColumnDatasets.surface_timevaryinginputs` utilities in
its cache method. It can instead store arrays or callables for a fully
in-memory experiment.

If forcing data must be refreshed by a scheduled callback, also extend
`CA.default_model_callbacks(::MyCaseForcing; kwargs...)` and return the
callback tuple for that type. It will be composed with the model's other
default callbacks. Supplying `callbacks` directly to `AtmosSimulation` requires
`default_callbacks = false` and replaces the complete default callback set, so
the component extension is normally the safer hook.

The surface uses the file's surface temperature `ts`
(`SurfaceConditions.ExternalTemperature`) with interactive Monin-Obukhov
fluxes. The initial condition additionally requires the `ta`, `ua`, `va`,
`hus`, and `rho` profiles.

#### The ClimaColumn schema

A ClimaColumn file is self-describing, so the reader needs no per-file
exceptions.

  - Global attributes: `site_latitude` / `site_longitude` in degrees.
  - Dimensions: column variables are pure 1D `(z, time)` and surface variables
    are `(time,)`. `z` is height in meters, strictly ascending, with at least
    two levels. `time` is a CF time coordinate (units plus calendar).
  - Variables use CMIP short names with SI `units` attributes. Column:
    `ta` [K], `hus` [kg kg⁻¹], `ua`/`va`/`wa` [m s⁻¹], `rho` [kg m⁻³],
    `tntha`/`tntva` [K s⁻¹], `tnhusha`/`tnhusva` [kg kg⁻¹ s⁻¹]. Surface:
    `ts` [K], `hfls`/`hfss` [W m⁻², upward positive], `coszen` [1],
    `rsdt` [W m⁻²].

Constructing a `ColumnDataset` validates a native file against this schema,
including exact canonical SI unit strings, and reports all violations.
`ColumnDatasets.validate(ColumnDatasets.ClimaColumnFile(), path)` performs
the same check explicitly;
`ClimaColumnFiles.write_column_forcing_file` is the one producer
implementation, used by the ERA5 generator.

#### Adding a format module

A new format is one self-contained module under `src/column_datasets/`:
define a singleton subtype of `ColumnDatasets.AbstractColumnFormat`, extend
the three required methods (`format_name`, `format_variable_name`,
`height_profile`) plus whichever overrides the format needs (`preprocess`
for unit conversions and fill values, `dates` for nonstandard time axes,
`read_profile`/`read_series` for layout quirks or derived variables,
`open_dataset` for grouped files), and pass the singleton via the `format`
keyword of `ColumnDatasets.ColumnDataset`. No changes to the forcing, setup, or
config machinery are needed.
