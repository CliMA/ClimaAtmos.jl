# Running Single-Column Cases

## Idealized cases

`ClimaAtmos.jl` supports several canonical test cases that are run in a single column model designed to verify how well PROPHET (an extended, prognostic EDMF scheme) reproduces each convective regime. These cases include variants of `bomex`, `dycoms`, `rico`, `soares`, `gabls`, and `trmm` and can be found in the `config/model_configs` directory. The purpose of each simulation is summarized in the following table:

| Abbreviation | Long Name                                            | Cloud Regime          | Reference                                                                                                                   |
|:------------ |:---------------------------------------------------- |:--------------------- |:--------------------------------------------------------------------------------------------------------------------------- |
| BOMEX        | Barbados Oceanographic and Meteorological Experiment | Marine Cumulus        | [Siebesma et al. (2003)](https://doi.org/10.1175/1520-0469(2003)60%3C1201:ALESIS%3E2.0.CO%3B2)                              |
| DYCOMS       | Dynamics and Chemistry of Marine Stratocumulus       | Marine Stratocumulus  | [Stevens et al. (2005)](https://doi.org/10.1175/MWR2930.1), [Ackerman et al. (2009)](https://doi.org/10.1175/2008MWR2582.1) |
| RICO         | Rain in Cumulus over the Ocean                       | Rainy Cumulus         | [Rauber et al. (2007)](https://doi.org/10.1175/BAMS-88-12-1912)                                                             |
| SOARES       | Shallow Cumulus Convection                           | Shallow Cumulus       | [Soares et al. (2004)](https://doi.org/10.1256/qj.03.223)                                                                   |
| GABLS        | GEWEX Atmospheric Boundary Layer Study               | Stable Boundary Layer | [Beare et al. (2006)](https://doi.org/10.1007/s10546-004-2820-6)                                                            |
| TRMM         | Tropical Rainfall Measuring Mission                  | Deep Convection       | [Grabowski et al. (2006)](https://doi.org/10.1256/qj.04.147)                                                                |

These are the canonical intercomparison cases; the [Setups](setups.md)
reference page lists all available setups, including further idealized
columns.

To run the BOMEX test case from the configuration file, start Julia in the
project root (`julia --project`) and execute the following:

```julia
import ClimaAtmos as CA

# get the configuration arguments
config = CA.AtmosConfig(
    "config/model_configs/prognostic_edmfx_bomex_column.yml";
    job_id = "bomex",
)
simulation = CA.AtmosSimulation(config)
sol_res = CA.solve_atmos!(simulation) # run the simulation
```

The same three lines run every case on this page; only the configuration
file changes.

## Externally-Driven Single Column Models

`ClimaAtmos.jl` currently supports several externally driven single column setups: `GCM` driven, `ReanalysisTimeVarying`, `ReanalysisMonthlyAveragedDiurnal`, and ARM VARANAL. Externally-driven means that the model is initialized and forced with data from a different simulation or analysis product. This differs from setups such as BOMEX or SOARES, which have steady forcing or functional forcing, respectively. These setups have been developed for model calibration and testing by recreating statistics that are close to either LES (for the `GCM` driven case only) or observations.

### GCM-Driven Case

For the `GCM` driven case, run the configuration file
`config/model_configs/prognostic_edmfx_gcmdriven_column.yml`. In the config,
the following settings are important:

```YAML
initial_condition: "GCM"
external_forcing_file: artifact"cfsite_gcm_forcing"/HadGEM2-A_amip.2004-2008.07.nc
cfsite_number : "site23"
```

Setting `initial_condition` to `GCM` selects the GCM-driven setup, which supplies the external forcing, surface treatment, and insolation from the external file. The `external_forcing_file` and `cfsite_number` together determine the temperature, specific humidity, and wind as well as horizontal and vertical advection profiles that drive the simulation, and can be set to a local file path instead of the artifact. Radiation and surface temperature are also specified. Here the forcing file, an example of which is stored in the artifact, contains groups for each `cfsite` to drive the simulation. See [Shen et al. 2022](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021MS002631) for more information.

### ARM VARANAL Case (SGP)

The ARM VARANAL setup drives a single column at the SGP Central Facility with
time-varying profiles and tendencies from the ARM Variational Analysis product
(`sgp60varanarucC1.c1`). Forcing includes horizontal advection, large-scale
subsidence (from `omega`), nudging toward observed T/q/u/v, prescribed surface
fluxes (LH/SH), and time-varying skin temperature. Monthly files are available
from the [ARM Data Center](https://adc.arm.gov/discovery/). Run the
configuration file `config/model_configs/prognostic_edmfx_armvaranal_column.yml`.

Key config entries (edit `external_forcing_file`, `start_date`, and `t_end` to
pick a sub-period within the monthly file):

```YAML
initial_condition: "ARMVARANAL"
external_forcing_file: artifact"arm_sgp_varanal_forcing"/sgp60varanarucC1.c1.20100901.000000.cdf
start_date: "20100918"
t_end: "4days"
```

The VARANAL file is converted to the ClimaColumn schema and run through the
generic `ForcingFromFile` path (see [Column Datasets](column_datasets_reference.md));
the forcing, prescribed surface fluxes, and insolation come from the setup,
so `external_forcing` is left unset.

The default period (Sep 18–22, 2010) spans a clear-to-convective transition
with a cold-front passage, a good diurnal-cycle test. To run another period,
set `external_forcing_file` to that month's VARANAL `.cdf` file.

### Reanalysis-Driven Case

#### Matched ERA5 Trajectory

The `ReanalysisTimeVarying` case extends the `GCM` driven case to single-column simulations that resolve the diurnal cycle, can be run at any site globally, and are driven by reanalysis, allowing calibration of PROPHET to earth-system observations in the single-column setting. For example, a set of config file arguments can be:

```YAML
initial_condition: "ReanalysisTimeVarying"
start_date: "20070701"
site_latitude: 17.0
site_longitude: -149.0
```

The case runs the configuration file
`config/model_configs/prognostic_edmfx_tv_era5driven_column.yml`. The
`ReanalysisTimeVarying` initial condition generates a column forcing file for
the requested site and dates (regridded from the global ERA5 archive, stored
through `ClimaArtifacts` for reproducibility) and hands it to the generic
`ForcingFromFile` setup, which takes the
initial condition, external forcing, surface skin temperature, and insolation
from that one file (surface fluxes are computed interactively by Monin–Obukhov similarity theory). Setting `external_forcing: "ReanalysisTimeVarying"` as well
is still accepted but no longer needed. You give the site and dates directly
rather than a file path because the file is generated on demand:
`start_date` is YYYYMMDD, `site_latitude` in degrees (-90...90), and
`site_longitude` in (-180...180). Artifact-backed ERA5 data is currently
available only for the tropical Pacific in the first 5 days of July 2007,
and only on the `clima` and Caltech HPC servers.

!!! note

    Depending on the amount of smoothing and data resolution, points near the boundaries can throw index errors. With default settings, users should stay at least 5 points away from the poles (1° for ERA5 data) for smoothing (4 points) and gradients (one extra point).

#### Monthly Averaged Forcing

As the matched ERA5 trajectory is data intensive, requiring downloads for each day, we have also implemented an external forcing dispatch to repeat a specific day of data indefinitely. This setup is ideal for monthly averaged ERA5 data by hour of day and can be used to calibrate to monthly statistics. The setup is similar, except we change the flag for `external_forcing` to indicate that we want to repeat data:

```YAML
initial_condition: "ReanalysisTimeVarying"
external_forcing: "ReanalysisMonthlyAveragedDiurnal"
start_date: "20070701"
site_latitude: 17.0
site_longitude: -149.0
```

The corresponding configuration file is
`config/model_configs/prognostic_edmfx_diurnal_scm_imp.yml`.

Running the reanalysis-driven cases at other times and locations requires
downloading and naming the raw ERA5 files for the processing script; see
[Generating ERA5 forcing data](@ref) in the Developer Guide.

### Column forcing datasets

To drive a case from a custom forcing file, see the
[Column Datasets](column_datasets_reference.md) reference page. To define
nonstandard forcing in a runscript, generate ERA5 forcing files, or write
your own datasets, see
[Adding a Column Dataset](extending_column_datasets.md) in the Developer
Guide.
