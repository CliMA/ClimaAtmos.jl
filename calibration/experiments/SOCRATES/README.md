# SOCRATES

A ClimaAtmos single-column model of the SOCRATES Atlas LES cases (Atlas et al., 2020 — PDF in
`docs/`), and an EKI calibration of its microphysics timescales against that LES.

Running the model is the primary capability; calibrating is built on top of it. Everything is ordinary
Julia — functions with keyword arguments, driven from a REPL or a short script. There is no
configuration file on the path and no YAML.

## Setup

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The Atlas LES reference is read either from the reduced files in `reference/Atlas_LES/` (`source =
:processed`, the default) or straight from the SSCF artifact (`source = :sscf`). Both are supported and
must agree.

## Run a case

```julia
include("src/model/SocratesModel.jl")

case = SocratesModel.socrates_case("RF09_Obs")
SocratesModel.run_case(case; output_dir = "runs/rf09")
```

The model integrates on the Atlas LES's own vertical levels by default — 320 for flights 1/9/10/11, 192
for 12/13 — so profiles compare to the LES with no vertical interpolation.

```julia
# a calibrated (or any) parameter set
SocratesModel.run_case(case; params = "path/to/parameters.toml", output_dir = "runs/calibrated")

# inline overrides; later sources win, and paths and Dicts can be mixed
SocratesModel.run_case(case;
    params = Dict("rain_autoconversion_timescale" => Dict("value" => 500.0, "type" => "float")),
    output_dir = "runs/tweaked")

# a different vertical resolution: build a grid and pass it
coarse = SocratesModel.socrates_grid(Float64, case; dz_min = 200)   # merge LES cells
SocratesModel.run_case(case; grid = coarse, output_dir = "runs/coarse")

uniform = SocratesModel.socrates_grid(Float64, case; faces = collect(range(0, 6000; length = 61)))
SocratesModel.run_case(case; grid = uniform, output_dir = "runs/uniform")

# Float32, and a shorter run
SocratesModel.run_case(case; FT = Float32, t_end = 3600, output_dir = "runs/quick")

# all 11 cases, optionally across a worker pool
SocratesModel.run_cases(SocratesModel.socrates_cases(); output_dir = "runs/all")

# build a simulation without solving it, to inspect or add callbacks
SocratesModel.socrates_simulation(Float64, case; output_dir = "runs/inspect")
```

`scripts/run_cases.jl` is the same thing as an editable script.

## Score a run against the LES

```julia
include("src/scoring/SocratesScoring.jl")

comparison = SocratesScoring.compare_to_les("runs/rf09", case)
SocratesScoring.print_comparison(comparison)
```

Reports a normalized misfit per variable, so the numbers are comparable across variables of very
different magnitude. No EKP is loaded. See `scripts/compare_to_les.jl`, and `scripts/tune_z_bounds.jl`
for the LES cloud top and derived scored region per case.

## Calibrate

```julia
include("src/calibration/SocratesCalibration.jl")

interface = SocratesCalibration.SocratesInterface(;
    output_dir = "calibrations/run01",
    grid_kwargs = (; dz_min = 200),      # optional; also fixes the observation levels
)
prior = SocratesCalibration.default_prior()
ekp = SocratesCalibration.build_ekp(interface, prior; ensemble_size = 10, T_stops = [1.0, 10.0, 100.0])
```

then run it with a worker backend — see `scripts/run_calibration.jl`, which also starts the workers.

Calibrated parameters are written per iteration to
`calibrations/run01/iteration_XXX/member_YYY/parameters.toml` and can be fed straight back to
`run_case`.

## Cases

11 cases: 5 `Obs`-forced and 6 `ERA5`-forced. Flight 11 has no `Obs` artifact, so `RF11_Obs` is
rejected. `Obs` cases run 12 h and are scored over hours 10–12; `ERA5` cases run 14 h and are scored
over a per-flight window derived from the SOCRATES summary file.

## Physics

Prognostic EDMFX with one updraft, 1-moment non-equilibrium microphysics, quadrature cloud, all-sky
RRTMGP with clear-sky diagnostics, prescribed surface temperature, Monin–Obukhov surface fluxes, and no
Rayleigh sponge (the domain top is the LES top). Radiation is computed online rather than prescribed
from the LES `dTdt_rad`, which SSCF could also supply.

## Layout

```
src/model/         run SOCRATES: cases, grid, forcing, parameters, model, runner
src/scoring/       compare a run to the LES: reference, scored region, normalization
src/calibration/   EKI on top: observations, observation map, interface, driver
scripts/           editable examples of each of the above
test/              runtests.jl -- no forward model required
reference/         Atlas LES data (untracked)
docs/              Atlas 2020 PDF
```

The three layers depend one way only: `scripts/run_cases.jl` loads `src/model/` alone and never pays
for EKP or ClimaCalibrate.
