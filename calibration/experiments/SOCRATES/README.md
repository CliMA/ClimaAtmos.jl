# Simple SOCRATES Calibration (ClimaAtmos + ClimaCalibrate)

This directory is a minimal calibration scaffold for SOCRATES field-campaign cases,
without the larger gcm_driven_scm case-library workflow.

The forward-model cases are now flight-driven and resolved through
SOCRATESSingleColumnForcings (Atlas inputs/outputs), not placeholder static paths.

## Files

- run_calibration.jl: main entrypoint
- model_interface.jl: defines CAL.forward_model(iteration, member)
- observation_map.jl: maps model output to the calibration vector g
- experiment_config.yml: high-level calibration settings and case list
- model_config.yml: base ClimaAtmos SCM setup
- prior.toml: calibration parameter priors

## Required Edits Before First Run

1. Ensure each case has:
   - flight_number
   - forcing_type (obs_data or ERA5_data)
2. Optional: set reference_file explicitly per case; if omitted, it is resolved via SOCRATESSingleColumnForcings.
3. If your reference variable names differ from LES defaults, adjust variable handling in ../gcm_driven_scm/helper_funcs.jl.
4. Adjust model_config.yml and prior.toml to your target microphysics/EDMF settings.

## Run

From ClimaAtmos.jl root:

julia --project=calibration/experiments/gcm_driven_scm calibration/experiments/SOCRATES/run_calibration.jl

Or submit:

sbatch calibration/experiments/SOCRATES/run_calibration.sbatch

## Tests

Run SOCRATES experiment unit tests from this directory:

julia --project=. test/runtests.jl

## Notes

- This setup uses helper utilities from ../gcm_driven_scm/helper_funcs.jl for reading and normalizing reference profiles.
- Default config is single-case (batch_size = 1) for fast iteration and debugging.
- To calibrate multiple flights, add more entries under cases and increase batch_size as desired.
- Legacy preset inspirations are documented in MIGRATION_GUIDE.md, including 10-12 hour windows and EKI defaults.
- The provided test prior calibrates two modern parameters: entr_inv_tau and mixing_length_eddy_viscosity_coefficient.
- Case forcing files and LES reference files are resolved at runtime using:
   - SSCF.open_atlas_les_input(flight_number, forcing_type)
   - SSCF.open_atlas_les_output(flight_number, forcing_type)
