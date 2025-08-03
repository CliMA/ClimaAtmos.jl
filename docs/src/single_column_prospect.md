# Single Column Models
`ClimaAtmos.jl` supports several canonical test cases that are run in a single column model designed to verify how well PROSPCT (EDMF) is able to reproduce each of the convective schemes. These cases include variants of `bomex`, `dycoms`, `rico`, `soares`, `gabls`, `gate`,and `trmm` and can be found in the `config/model_configs` directory. The purpose of each simulation is summarized in the following table:

| Abbreviation | Long Name | Cloud Regime | Reference |
|--------------|-----------|--------------|-----------|
| BOMEX | Barbados Oceanographic and Meteorological Experiment | Marine Cumulus | [Siebesma et al. (2003)](https://doi.org/10.1175/1520-0469(2003)60<1201:ALESIS>2.0.CO;2) |
| DYCOMS | Dynamics and Chemistry of Marine Stratocumulus | Marine Stratocumulus | [Stevens et al. (2005)](https://doi.org/10.1175/MWR2930.1), [Ackerman et al. (2009)](https://doi.org/10.1175/2008MWR2582.1) |
| RICO | Rain in Cumulus over the Ocean | Rainy Cumulus | [Rauber et al. (2007)](https://doi.org/10.1175/BAMS-88-12-1912) |
| SOARES | Shallow Cumulus Convection | Shallow Cumulus | [Soares et al. (2004)](https://doi.org/10.1256/qj.03.223) |
| GABLS | GEWEX Atmospheric Boundary Layer Study | Dry Convective Boundary Layer | [Kosović & Curry (2000)](https://doi.org/10.1175/1520-0469(2000)057<1052:ALESSO>2.0.CO;2) |
| TRMM | Tropical Rainfall Measuring Mission | Deep Convection | [Grabowski et al. (2006)](https://doi.org/10.1256/qj.04.147) |
| ARM_SGP | Atmospheric Radiation Measurement Southern Great Plains | Continental Shallow Cumulus | [Brown et al. (2002)](https://doi.org/10.1256/qj.01.202) |
| LifeCycle | Life Cycle of Shallow Cumulus | Shallow Cumulus Life Cycle | [Tan et al. (2018)](https://doi.org/10.1002/2017MS001162) |
| GATE_III | GARP Atlantic Tropical Experiment | Tropical Deep Convection | [Khairoutdinov et al. (2009)](https://doi.org/10.3894/JAMES.2009.1.15) |

For example, to run the BOMEX test case execute the following:
```bash
julia --project=.buildkite .buildkite/ci_driver.jl --config_file config/model_configs/prognostic_edmfx_bomex_column.yml --job_id bomex
```
It may also be helpful to run in interactive mode to be able to examine the simulation object, debug, and develop the code further. To enter debug mode run `julia --project=.buildkite` and then in the REPL run:
```julia
using Revise # if you are developing ClimaAtmos
import ClimaAtmos as CA

# get the configuration arguments
simulation = CA.AtmosSimulation("config/model_configs/prognostic_edmfx_bomex_column.yml")
sol_res = CA.solve_atmos!(simulation) # run the simulation
```

## Externally-Driven Single Column Models
Currently three versions of the externally driven single column model, `GCM` driven, `ReanalysisTimeVarying` driven, and `ReanalysisMonthlyAveragedDiurnal` driven are supported in `ClimaAtmos.jl`. Externally-driven means that the model is initialized and forced with data coming from a different simulation. This differs from setups like, for example, BOMEX or SOARES which have steady forcing and low domain tops (~4km) or functional forcing, respectively. They have been developed specifically for the purpose of model calibration by recreating statistics that are close to either LES, for the `GCM` driven case only, or to observations.

### GCM-Driven Case
For the `GCM` driven case we can run the experiment using the config file `config/model_configs/prognostic_edmfx_gcmdriven_column.yml` by running:
```bash
julia --project=.buildkite .buildkite/ci_driver.jl --config_file config/model_configs/prognostic_edmfx_gcmdriven_column.yml --job_id gcm_driven_scm
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

### Reanalysis-Driven Case
#### Matched ERA5 Trajectory
The `ReanalysisTimeVarying` case extends the `GCM` driven case by providing support for single-column simulations which resolve the diurnal cycle, can be run at any site globally, and uses reanalysis to drive the simulation, allowing for calibration of EDMF to earth-system observations in the single-column setting. This feature was found to be needed to address biases in calibration arising from correlation between time-of-day and cloud liquid water path over the tropical Pacific. For this simulation we again highlight similar arguments in the config file:
```YAML
initial_condition: "ReanalysisTimeVarying"
external_forcing: "ReanalysisTimeVarying"
surface_setup: "ReanalysisTimeVarying"
surface_temperature: "ReanalysisTimeVarying"
start_date: "20070701"
site_latitude: 17.0
site_longitude: -149.0
```
By this point, the first 4 entries are intuitive. We need to dispatch over each of these methods to setup the forcing for each component of the model. To obtain the observations, now note that instead of directly specifying a file we must specify a `start_date`, `site_latitude`, and `site_longitude`. This is because we now use `ClimaArtifacts.jl` to store data to ensure reproducibility of our simulation and results. `start_date` should be in in YYYMMDD format, `site_latitude` should be in degrees between -90 and 90, and `site_longitude` should be between -180 and 180. 

!!! note
    Depending on the amount of smoothing and data resolution, points near the boundaries will throw index errors. With default settings, users should stay at least 5 points away from the poles (1° for ERA5 data) for smoothing (4 points) and gradients (one extra point).

The data is generated by downloading from ECMWF and further documentation for ERA5 data download can be found either directly on the ECMWF page and `ClimaArtifacts.jl`. Note that the profiles, surface temperature, and surface fluxes cannot be obtained from a single request and so together we need 3 files for all the data. We include a script at `src/utils/era5_observations_to_forcing_file.jl` which extracts the profiles and computes the tendencies needed for the simulation from the raw ERA5 reanalysis files. We store the observations directly into an artifact `era5_hourly_atmos_processed` to eliminate the need to reprocess specific sites and locations. This setup means that users are free to choose sites globally at any time at which ERA5 data is available. Unfortunately, global hourly renanalysis is too large to store in an artifact and so we have currently only provided support for the first 5 days of July 2007 in the tropical Pacific, stored in `era5_hourly_atmos_raw`, only available on the `clima` and Caltech `HPC` servers. The test case can be run using: 
```bash
julia --project=.buildkite .buildkite/ci_driver.jl --config_file config/model_configs/prognostic_edmfx_tv_era5driven_column.yml --job_id era5driven
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
julia --project=.buildkite .buildkite/ci_driver.jl --config_file config/model_configs/prognostic_edmfx_diurnal_scm_imp.yml --job_id bomex
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
Good luck! :wink:
