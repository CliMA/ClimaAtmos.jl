# SOCRATES ClimaAtmos calibration

Modern ClimaAtmos column SCM + ClimaCalibrate 0.3 experiment for the SOCRATES
Atlas LES cases (Atlas et al., 2020 — PDF in `docs/`).

## Cases (11 = 5 Obs + 6 ERA)

| Case | Flight | Forcing |
|------|--------|---------|
| RF01_Obs / RF01_ERA | 1 | Obs / ERA5 |
| RF09_Obs / RF09_ERA | 9 | Obs / ERA5 |
| RF10_Obs / RF10_ERA | 10 | Obs / ERA5 |
| RF11_ERA | 11 | ERA5 only (no Obs Atlas artifact) |
| RF12_Obs / RF12_ERA | 12 | Obs / ERA5 |
| RF13_Obs / RF13_ERA | 13 | Obs / ERA5 |

## Physics / windows

- Forward model: `prognostic_edmfx`, 1M microphysics, quadrature cloud, `allskywithclear`
- Forcing: in-memory `SocratesSetup` / `SocratesForcing` from SSCF (no ClimaColumn round-trip)
- Domain: per-case `z_max = maximum(SSCF.default_new_z(flight))`, `rayleigh_sponge: false`
- Time: `start_date = 19700101`; SSCF/insolation use true LES wall clock
- Obs scoring window: hours **10–12** of 12 h Obs runs (Atlas 2020 §5)
- ERA scoring window: hours **11–13** of 14 h ERA runs (around reference ~hour 12)
- `z_bounds` and `N_CCN` in `model_interface/utils.jl`

## Setup

From this directory:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Smoke one case

```bash
julia --project=. run_single_case.jl RF01_Obs --short
julia --project=. run_single_case.jl RF01_Obs
```

## Optional debug ClimaColumn dump

```bash
julia --project=. forcing/generate_climacolumn.jl RF01_Obs
```

Not used by the driver. Outputs land in `forcing/climacolumn/` (gitignored).

## Calibrate

Edit `priors/prior.toml` and `configs/experiment_config.yml` (`y_var_names`, ensemble size, etc.) as needed, then:

```bash
julia --project=. run_calibration.jl
```

Default `y_var_names`: `clw, husra, cli, hussn, lwp, iwp, rwp, swp`.

EKI outputs go under `socrates_calibration/`.

## Layout

```
configs/           model + experiment YAML
forcing/           SSCF → in-memory arrays (+ optional NC writer)
model_interface/   ClimaCalibrate AbstractModelInterface + SocratesSetup
priors/            editable prior.toml
reference/         optional local LES copies (SSCF artifacts preferred)
docs/              Atlas 2020 PDF
run_single_case.jl
run_calibration.jl
```

The old `calibrate/` tree is legacy scaffolding and is not used by these entry points.
