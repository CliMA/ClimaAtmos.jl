# Variance adjustments and SGS quadrature

Branch: **`jb/variance_adjustments`** — keep physics changes on this branch; `run_calibration.jl` warns if the ClimaAtmos repo is on another branch.

## Science goal

Test whether adding **subcell geometric variance** from vertical background gradients improves clouds when using **SGS quadrature** over **(T, q_tot)**. Cached variances used for **cloud fraction** and mixing-length closures are **unchanged**; only the variances passed into **microphysics quadrature** and **SGS saturation adjustment** are augmented.

YAML flag: **`sgs_quadrature_subcell_geometric_variance`** (`false` by default). Implementation: `subcell_geometric_variance_increment`, `subcell_geometric_covariance_Tq`, `materialize_sgs_quadrature_moments!`, and `sgs_quadrature_Tq_moments` in [`src/cache/microphysics_cache.jl`](../../../src/cache/microphysics_cache.jl).

**Correlation:** With the flag **off**, quadrature uses scalar **`correlation_Tq(params)`** from ClimaParams. With the flag **on**, each cell uses an effective correlation **ρ_Tq** from total covariance  
`Cov_tot = ρ_param·σ_q,turb·σ_T,turb + (1/12)Δz²(∂T/∂z)(∂q/∂z)`  
with **∂T/∂z ≈ (∂T/∂θ_li)(∂θ_li/∂z)**, divided by **σ_q,tot·σ_T,tot** (Cauchy–Schwarz floor via `ϵ_numerics`), then clamped to **[-1, 1]**.

The older draft in `variance_adjustments.jl` described **(q, θ_li)** moments; the model follows **ClimaAtmos** quadrature variables **(q_tot, T)**.

## Repository layout

| Path | Role |
|------|------|
| [`model_configs/trmm_column_varquad.yml`](model_configs/trmm_column_varquad.yml) | TRMM_LBA column, 0M, `cloud_model: quadrature` |
| [`model_configs/dycoms_rf01_column_varquad.yml`](model_configs/dycoms_rf01_column_varquad.yml) | DYCOMS RF01 column, same pattern |
| [`experiment_config.yml`](experiment_config.yml) | EKI slice: case, `quadrature_order`, varfix flag, paths |
| [`prior.toml`](prior.toml) | Small EDMF-related prior (extend as needed) |
| [`generate_observations_reference.jl`](generate_observations_reference.jl) | Build `observations_reference.jld2` (truth **y**) |
| [`run_calibration.jl`](run_calibration.jl) | EKI via `ClimaCalibrate` + `WorkerBackend` |
| [`scripts/sweep_forward_runs.jl`](scripts/sweep_forward_runs.jl) | Grid: TRMM + DYCOMS × `N_quad=1:10` × varfix on/off |
| [`analysis/`](analysis/) | Stub plotting for profiles, losses, parameters |
| [`cfsite_stub.yml`](cfsite_stub.yml) | Placeholder for cfSite / GCM-driven style cases |
| [`googleles/README.md`](googleles/README.md) | GoogleLES stub |

Outputs go under **`simulation_output/<CASE>/N_<n>/varfix_on|off/...`** (see sweep script and `experiment_config.yml`).

## Baseline vs quadrature order

True **grid-mean** microphysics (no SGS integral over an SGS PDF) is **`sgs_distribution: mean`** (`GridMeanSGS`). That is different from **`quadrature_order: 1`**, which still evaluates the microphysics closure at **one** Gauss–Hermite node of the configured SGS distribution (so it is not the same as “no quadrature”). The sweep uses **orders 1–10** with the default **lognormal** SGS distribution; document comparisons accordingly.

## Calibration modes (workflow)

1. **Calibrated, varfix off:** edit `experiment_config.yml` (`sgs_quadrature_subcell_geometric_variance: false`), run `run_calibration.jl`.
2. **Calibrated, varfix on:** set flag `true`, change `output_dir` to a new folder, run again.
3. **Naive:** calibrate only with varfix **off**, then run forward jobs with varfix **on** using the **same** `parameters.toml` (no second EKI) — use `scripts/sweep_forward_runs.jl` or a one-off `AtmosConfig` with the saved TOML.

## Commands

```bash
cd calibration/experiments/variance_adjustments
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Reference vector y (matches z_elem columns in the chosen case YAML)
julia --project=. generate_observations_reference.jl

# EKI (needs Distributed workers; tune VARIANCE_CALIB_WORKERS)
julia --project=. run_calibration.jl

# Forward-only grid (long-running)
julia --project=. scripts/sweep_forward_runs.jl
```

Slurm: [`run_calibration.sbatch`](run_calibration.sbatch) (adjust partitions/GPU flags for your site).

## GPU

ClimaAtmos runs on GPU when built with a GPU-backed `ClimaComms` context; this experiment uses the same `AtmosConfig` stack as other calibrations. For cluster jobs, set device env vars and Slurm GPU options as in your site’s ClimaAtmos docs.

## Revisiting variance in dynamics

Overwriting **prognostic** variance fields or changing **cloud-fraction** covariances is **out of scope** for now; only quadrature inputs are modified. Revisit once the quadrature-only signal is understood.
