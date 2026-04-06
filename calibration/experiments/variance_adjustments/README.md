# Variance adjustments and SGS quadrature

Branch: **`jb/variance_adjustments`** — **all ClimaAtmos source edits for this study belong on this branch.** `run_variance_calibration!()` (in [`eki_calibration.jl`](eki_calibration.jl)) warns if the repo is on another branch.

## Experiment specification

This section is the **full experiment checklist** (science intent, setups, and infrastructure). Operational commands are in **One command**, **Study tracks**, and **Commands** below.

### Science question

On branch **`jb/variance_adjustments`**, we ask whether **accounting for background gradients in SGS quadrature** improves cloud development. Given local (at some scale) **variance of `q_tot`**, **variance of a thermodynamic variable**, and **their covariance**, does **using the background gradient when constructing quadrature moments** improve cloud representation relative to the reference closure?

### Draft code and thermodynamic choice

- There is an initial draft in [`variance_adjustments.jl`](variance_adjustments.jl). **Treat it as derivation notes, not the source of truth** — validate against the live quadrature path in [`src/cache/microphysics_cache.jl`](../../../src/cache/microphysics_cache.jl) (`subcell_geometric_*`, `materialize_sgs_quadrature_moments!`, `sgs_quadrature_Tq_moments`).
- The draft uses **liquid–ice potential temperature** (\( \theta_{li} \))-style notation in places. **ClimaAtmos SGS quadrature for this closure is implemented in \((q_{tot}, T)\).** The implementation and any new theory should stay aligned with that pair.
- Assume **`ref_includes_gradient = false`** in the reference ClimaAtmos setup unless you deliberately change it.

### Repository discipline

- **Headquarters:** [`calibration/experiments/variance_adjustments/`](.) (this directory). Mirror patterns from other experiments under [`calibration/experiments/`](../).
- **Source changes** that support this study belong **only** on **`jb/variance_adjustments`**; the calibration driver checks the branch and **warns** if you are elsewhere.

### Scope: quadrature inputs only (revisit later)

- **Do not overwrite prognostic variance fields** for now. **Only** the **quadrature setup** (and related SGS saturation adjustment inputs) uses the gradient augmentation. Effects on dynamics and cached covariances elsewhere would multiply confounders; start here, then **revisit** tying dynamics to the same closure once the quadrature-only signal is clear (see also **Revisiting variance in dynamics**).

### Canonical setups, equilibrium, quadrature order, varfix

- Use **canonical column setups** already in the repo — **TRMM_LBA**, **DYCOMS_RF01**, etc. — via [`model_configs/`](model_configs/).
- Run in **equilibrium** configurations (e.g. **0M** column configs as provided); avoid transient suites unless you extend the experiment.
- Exercise **`quadrature_order` from 1 to 5** in forward sweeps (ClimaAtmos `gauss_hermite` only implements **N ≤ 5**). **`quadrature_order: 1`** is **not** the same as true grid-mean microphysics — it is still one node of the configured SGS PDF; see **Baseline vs quadrature order**.
- Run with **subcell geometric variance** (**varfix**) **`sgs_quadrature_subcell_geometric_variance`** **on** and **off**, with **separate output trees** per configuration.

### Vertical resolution and timestep (forward sweep)

- **Coarsening vs. `dt`:** the forward ladder **only coarsens** relative to each case’s baseline YAML. **`dt` is left at the YAML value** for every tier (you **do not** increase `dt` when coarsening). If you later add **refinement** tiers above the baseline, you may need a **smaller** `dt`; that is not implemented by default.
- **Generic ladder** ([`scripts/resolution_ladder.jl`](scripts/resolution_ladder.jl)): built from the **case model YAML** (`z_elem`, `z_max`, `z_stretch`, `dz_bottom`, `dt`) — no per-case Julia tables.
  - **`z_stretch: false`:** coarsen by dividing **`z_elem`** (CLI `--ladder-coarsen-ratio`, default `2`) down to **`--ladder-z-elem-min`**, for **`--ladder-n-tiers`** steps (default `4`).
  - **`z_stretch: true`:** coarsen by multiplying **`dz_bottom`** (`--ladder-min-dz-factor`, default `2`) while keeping **`z_elem`**, using the **same ClimaCore `DefaultZMesh`** ClimaAtmos uses; if the mesh constructor fails, **drop `z_elem`** (same ratio) and **reset `dz_bottom`** to the YAML baseline, then continue.
- **Cases:** listed in [`forward_sweep_cases.yml`](forward_sweep_cases.yml); override with **`--registry=REL.yml`** on [`scripts/sweep_forward_runs.jl`](scripts/sweep_forward_runs.jl). Add rows for new columns (cfSite, GoogleLES, …). Optional key **`forward_sweep_case_slug`** in a model YAML fixes the output folder if two configs share **`initial_condition`**.
- **Forward sweep:** [`run_full_study.jl`](run_full_study.jl) runs **`--resolution-ladder`** by default (subprocess flags, not env). Use **`--forward-baseline-only`** on `run_full_study.jl` for baseline YAML only (20 tasks with the default registry and `N_quad` 1–5), or run the sweep with **`--baseline-only`**. Output segments: uniform stretch → **`z82_dt150s/`**; stretched → **`z82_dt150s_dzb30/`** (includes effective `dz_bottom` when `z_stretch` is true).
- **EKI** slices still use **fixed** `z_elem`/`dt` from their **`model_config_path`** unless you add YAMLs per tier.

### Calibration: naive vs. calibrated; prior size

- **Naive:** calibrate with **varfix off** only; then run **forwards with varfix on** using the **same** drawn **`parameters.toml`** (not a second EKI).
- **Calibrated:** run **separate** EKIs for **varfix off** and **varfix on** so you can compare **optimal parameters** and performance.
- Use a **small** set of important parameters in [`prior.toml`](prior.toml) to stabilize the pipeline; **add more** once runs are routine.

### Outputs and analysis

- Keep **all** experiment outputs under this directory in a **consistent tree** (see **Layout** under **Study tracks**). **`simulation_output/...`** holds ClimaAtmos runs; **`analysis/figures/...`** holds plots.
- Provide **analysis** for **vertical profiles**, **EKI losses**, **parameter trajectories / comparisons**, etc. — [`analysis/plotting/`](analysis/plotting/) and [`run_post_analysis.jl`](analysis/plotting/run_post_analysis.jl).

### Parallel runs, ensemble on GPU, ClimaCalibrate

- It would be useful to run **many uncoupled SCM columns** (e.g. large ensembles) with **high throughput on GPU**. **Process-level** parallelism is what you have today: EKI **workers** (**`VARIANCE_CALIB_WORKERS`**), Slurm **array** jobs, and optional **per-process** **`CLIMACALIB_DEVICE`**. **Model-level** “batch many columns on one GPU” is **not** implemented here — see **Parallel runs and GPU**.

### GPU and Slurm

- **All** experiment driver code paths should remain **GPU-compatible** (`CLIMACALIB_DEVICE`, `get_comms_context`).
- Use the **Slurm** scripts in this directory (and under **`scripts/`**) for cluster submission — [`run_calibration.sbatch`](run_calibration.sbatch), [`run_full_study.sbatch`](run_full_study.sbatch), [`scripts/submit_forward_sweep_baseline.sh`](scripts/submit_forward_sweep_baseline.sh) / [`scripts/submit_forward_sweep_resolution.sh`](scripts/submit_forward_sweep_resolution.sh) (forward grid), [`scripts/calibration_sweep_array.sbatch`](scripts/calibration_sweep_array.sbatch).

### Future: cfSite and GoogleLES

- **cfSite / GCM-driven columns:** forcing patterns exist elsewhere (e.g. [`../gcm_driven_scm/`](../gcm_driven_scm/)); [`cfsite_stub.yml`](cfsite_stub.yml) is a placeholder. When you add a real column YAML, register it in [`forward_sweep_cases.yml`](forward_sweep_cases.yml) so the forward sweep picks it up without editing Julia lists.
- **GoogleLES:** stub only — [`googleles/README.md`](googleles/README.md); same registry hook when a config exists.

---

## Goals (short)

Same intent as **Science question** above: test whether **gradient-informed quadrature** improves clouds, using **canonical columns**, **N_quad 1–5**, **varfix on/off**, a **generic vertical-resolution ladder** in the forward sweep (**`dt` fixed** to each case YAML when coarsening), **naive vs calibrated** workflows (**naive** forwards are run by [`run_full_study.jl`](run_full_study.jl) from [`va_naive_varfix_off_source_configs()`](calibration_sweep_configs.jl)), **GPU-ready** code, **Slurm** drivers, and **analysis** under this folder.

## One command: full README workflow

From **`calibration/experiments/variance_adjustments`**:

```bash
julia --project=. run_full_study.jl
```

**REPL (kwargs, no env):** `using Pkg; Pkg.activate("."); include("run_full_study.jl"); run_full_study!()` — or `run_full_study!(; skip_forward = true)`, `run_full_study!(; forward_baseline_only = true)`, etc. See the header in [`run_full_study.jl`](run_full_study.jl).

This runs, in order:

1. **Forward grid** — every case in [`forward_sweep_cases.yml`](forward_sweep_cases.yml) × **`N_quad` 1–5** × **varfix on/off** × **resolution ladder** from [`scripts/resolution_ladder.jl`](scripts/resolution_ladder.jl) (**~80** runs with the default registry and ladder settings; long). Paths look like `simulation_output/<CASE>/z82_dt150s/N_3/varfix_on/forward_only/` (stretched grids add `_dzb…` in the segment).
2. **EKI sweep** — reference + calibration for every YAML in [`calibration_sweep_configs.jl`](calibration_sweep_configs.jl).
3. **Naive forwards** — for each varfix-off YAML in [`va_naive_varfix_off_source_configs()`](calibration_sweep_configs.jl), one varfix-**on** ClimaAtmos run at the **case YAML** resolution using the **merged member TOML** from the varfix-off EKI (default: latest iteration, member 1). Output: `simulation_output/<CASE>/N_<n>/varfix_on/naive_from_varfix_off/forward_only/`. Script: [`scripts/run_naive_varfix_on_forwards.jl`](scripts/run_naive_varfix_on_forwards.jl).
4. **Post-analysis** — [`va_run_post_analysis!`](analysis/plotting/run_post_analysis.jl) per calibration YAML, then an extra **naive overlay** pass (reference + naive `output_active`, losses/parameters still from varfix-off EKI) under `analysis/figures/<CASE>_N<n>_naive_varfix_on_from_<calibration_mode>/`.

**Fast / partial runs (CLI flags):**

```bash
julia --project=. run_full_study.jl --skip-forward                    # EKI + naive + figures (needs EKI for naive)
julia --project=. run_full_study.jl --forward-baseline-only           # 20 forwards + rest (no resolution ladder)
julia --project=. run_full_study.jl --skip-naive                      # no naive forwards / naive figure overlays
julia --project=. run_full_study.jl --skip-instantiate --skip-figures # examples
```

**Skips:** `--skip-calib`, `--skip-figures`, `--skip-instantiate`. Reuse finished cells: `--forward-skip-done`, `--naive-skip-done`. **Failures:** by default each sweep **logs and continues** (forward grid, EKI sweep, naive forwards); use **`--forward-fail-fast`**, **`--calib-fail-fast`**, or **`--naive-fail-fast`** on `run_full_study.jl` to stop the subprocess on the first error (same as `--fail-fast` on the underlying scripts). Slurm **one-task-per-job** array runs still fail that job if their single run errors. **`run_full_study.jl --help`** lists all flags.

**Naive member choice** (on [`scripts/run_naive_varfix_on_forwards.jl`](scripts/run_naive_varfix_on_forwards.jl)): `--eki-iteration=N`, `--eki-member=M` (defaults: latest iteration, member 1).

## Study tracks (pieces)

| Track | What it does | Entry |
|-------|----------------|--------|
| **Full driver** | Forwards (N×varfix×**resolution ladder** by default; registry [`forward_sweep_cases.yml`](forward_sweep_cases.yml)) + EKI sweep + **naive** varfix-on forwards + figures. | [`run_full_study.jl`](run_full_study.jl) |
| **A — Single EKI slice** | One YAML only. | [`run_e2e.jl`](run_e2e.jl) or `generate_observations_reference.jl` + `run_calibration.jl` |
| **B — Forward-only grid** | Registry cases × `N_quad` 1–5 × varfix × **`--resolution-ladder`** or **`--baseline-only`**. | [`scripts/sweep_forward_runs.jl`](scripts/sweep_forward_runs.jl) (`--help`), [`forward_sweep_cases.yml`](forward_sweep_cases.yml), Slurm: [`submit_forward_sweep_baseline.sh`](scripts/submit_forward_sweep_baseline.sh) / [`submit_forward_sweep_resolution.sh`](scripts/submit_forward_sweep_resolution.sh) plus the `.sbatch` bodies they submit |
| **C — Several EKI slices** | Same as full driver without figures/forward, or Slurm per task. | [`scripts/run_calibration_sweep.jl`](scripts/run_calibration_sweep.jl), [`calibration_sweep_configs.jl`](calibration_sweep_configs.jl) |

**Uniqueness (avoid clobbering):** Whenever **case**, **`quadrature_order`**, **`sgs_quadrature_subcell_geometric_variance`**, or **`observation_fields`** changes, use **distinct** `output_dir` and **`observations_path`** in that YAML (see checked-in templates). Reference ClimaAtmos output defaults to **`simulation_output/<CASE>/N_<n>/varfix_on/reference/`** or **`.../varfix_off/reference/`** unless you set **`reference_output_dir`**.

**Layout:**

- **Reference + observations file:** under **`simulation_output/<CASE>/N_<n>/varfix_on/`** or **`varfix_off/`**, directory **`reference/`** (truth run) and the path in **`observations_path`** (usually `.../reference/observations.jld2`).
- **EKI:** sibling **`eki/`** under the same case / N / varfix branch (or whatever you set in **`output_dir`**).
- **Forward sweep:** `simulation_output/<CASE>/<resSegment>/N_<n>/<varfix_on|off>/forward_only/` where `resSegment` is `z<z_elem>_dt<dtSlug>`, or (if `z_stretch: true` in the case YAML) `z<z_elem>_dt<dtSlug>_dzb<dz_bottom>`. With the ladder off, there is a **single** tier from the baseline YAML.
- **Naive (varfix on, frozen EKI TOML):** `simulation_output/<CASE>/N_<n>/varfix_on/naive_from_varfix_off/forward_only/` (same `z_elem`/`dt` as the varfix-off slice’s case YAML).
- **Figures:** `analysis/figures/<CASE>_N<n>_<varfix>_<calibration_mode>/` by default from [`va_run_post_analysis!`](analysis/plotting/run_post_analysis.jl) so slices do not overwrite PNGs; naive overlays use `analysis/figures/<CASE>_N<n>_naive_varfix_on_from_<calibration_mode>/`.

## Science goal

**Implementation:** Add **subcell geometric variance** from vertical **background gradients** for **SGS quadrature** in **(q_tot, T)** (ClimaAtmos’s quadrature variables). Cached variances for **cloud fraction** and **mixing length** are **unchanged**; only **microphysics quadrature** and **SGS saturation adjustment** are augmented. Use **`ref_includes_gradient = false`** in the reference setup unless you change it on purpose.

[`variance_adjustments.jl`](variance_adjustments.jl) holds **derivation notes** (e.g. **θ_li** notation); [`src/cache/microphysics_cache.jl`](../../../src/cache/microphysics_cache.jl) defines runtime behavior.

YAML: **`sgs_quadrature_subcell_geometric_variance`** (`false` default). Functions: `subcell_geometric_variance_increment`, `subcell_geometric_covariance_Tq`, `materialize_sgs_quadrature_moments!`, `sgs_quadrature_Tq_moments`.

**Correlation:** Flag **off** → scalar **`correlation_Tq(params)`** (ClimaParams). Flag **on** → **ρ_Tq** from  
`Cov_tot = ρ_param·σ_q,turb·σ_T,turb + (1/12)Δz²(∂T/∂z)(∂q/∂z)`  
with **∂T/∂z ≈ (∂T/∂θ_li)(∂θ_li/∂z)**, normalized by **σ_q,tot·σ_T,tot** (Cauchy–Schwarz floor `ϵ_numerics`), clamped to **[-1, 1]**.

## Where things live

| Topic | Location |
|-------|----------|
| `variance_adjustments.jl` vs implementation | Compare to [`microphysics_cache.jl`](../../../src/cache/microphysics_cache.jl). |
| No prognostic variance overwrite | **Revisiting variance in dynamics**. |
| TRMM / DYCOMS, `N_quad`, equilibrium | [`model_configs/`](model_configs/), [`scripts/sweep_forward_runs.jl`](scripts/sweep_forward_runs.jl), **Baseline vs quadrature order**. |
| Vertical res ladder | [`scripts/resolution_ladder.jl`](scripts/resolution_ladder.jl), [`forward_sweep_cases.yml`](forward_sweep_cases.yml); **`run_full_study.jl`** uses **`--resolution-ladder`** unless **`--forward-baseline-only`**. |
| Varfix on/off | YAML + separate **`output_dir`**. |
| Naive vs calibrated | **Calibration modes (workflow)**. |
| Prior size | [`prior.toml`](prior.toml). |
| Output paths | **`simulation_output/...`**. |
| Figures | [`analysis/plotting/`](analysis/plotting/); per-slice subfolders via [`va_run_post_analysis!`](analysis/plotting/run_post_analysis.jl). |
| Multi-slice EKI | [`scripts/run_calibration_sweep.jl`](scripts/run_calibration_sweep.jl), [`scripts/calibration_sweep_array.sbatch`](scripts/calibration_sweep_array.sbatch). |
| Slurm | [`run_calibration.sbatch`](run_calibration.sbatch), [`scripts/submit_forward_sweep_baseline.sh`](scripts/submit_forward_sweep_baseline.sh). |
| cfSite / GoogleLES | [`cfsite_stub.yml`](cfsite_stub.yml), [`../gcm_driven_scm/`](../gcm_driven_scm/), [`googleles/README.md`](googleles/README.md). |

## Parallel runs and GPU

A **nice-to-have** (not implemented here) is **model-level GPU batching**: **one** program advances **many** uncoupled SCM columns together—ensemble members with **different parameters**, no column-to-column coupling—using a **batch axis** in the state so kernels exploit **real on-device parallelism** (many columns per launch), instead of needing **one process or one dominant stream per member**. That is **not** the same as EKI **scheduling** members in rounds across **`VARIANCE_CALIB_WORKERS`** Julia processes (process parallelism, still **one** SCM domain per `forward_model`). It would need explicit support in ClimaAtmos and the calibration driver; this experiment does not provide it.

**What you have today:** each `forward_model` is **one** full SCM integration. **[`eki_calibration.jl`](eki_calibration.jl)** uses **`ClimaCalibrate.calibrate`** with **`CAL.WorkerBackend()`**; **`VARIANCE_CALIB_WORKERS`** caps how many of those integrations run **concurrently** (see **Environment variables**). Device selection uses **`CLIMACALIB_DEVICE`** and **`va_comms_ctx()`** in [`model_interface.jl`](model_interface.jl). For **embarrassingly parallel** forward grids, use **[`scripts/sweep_forward_runs.jl`](scripts/sweep_forward_runs.jl)** and **`sbatch --array=...`** (see **[`scripts/submit_forward_sweep_baseline.sh`](scripts/submit_forward_sweep_baseline.sh)**). This **`experiment_config.yml`** describes **one** case and **one** observation vector **y**; combining several cases in **one** EKI would need a different forward model or a **minibatch** pattern—see **[`../gcm_driven_scm/`](../gcm_driven_scm/)**.

## Repository layout

| Path | Role |
|------|------|
| [`experiment_common.jl`](experiment_common.jl) | Shared: observation field specs, noise matrix, `job_id`, comms context |
| [`model_configs/trmm_column_varquad.yml`](model_configs/trmm_column_varquad.yml) | TRMM_LBA column, 0M, `cloud_model: quadrature` |
| [`model_configs/dycoms_rf01_column_varquad.yml`](model_configs/dycoms_rf01_column_varquad.yml) | DYCOMS RF01 column, same pattern |
| [`experiment_config.yml`](experiment_config.yml) | Default EKI slice (TRMM, N=3, varfix off) |
| [`experiment_config_trmm_N3_varfix_on.yml`](experiment_config_trmm_N3_varfix_on.yml) | Same TRMM slice, varfix on (separate dirs) |
| [`experiment_config_dycoms_N3_varfix_off.yml`](experiment_config_dycoms_N3_varfix_off.yml) | DYCOMS RF01, N=3, varfix off |
| [`experiment_config_multifield_example.yml`](experiment_config_multifield_example.yml) | Comments for stacking `thetaa` / `hur` / `cl` in **y** |
| [`prior.toml`](prior.toml) | EDMF-related prior (names must exist in merged ClimaParams for the chosen SCM `toml`) |
| [`run_full_study.jl`](run_full_study.jl) | **Full spec workflow:** forward grid (resolution ladder by default) + EKI sweep + naive forwards + figures |
| [`calibration_sweep_configs.jl`](calibration_sweep_configs.jl) | `va_calibration_sweep_configs()` (EKI + figures) and `va_naive_varfix_off_source_configs()` (naive track) |
| [`scripts/run_naive_varfix_on_forwards.jl`](scripts/run_naive_varfix_on_forwards.jl) | Naive varfix-on forwards from varfix-off EKI member TOML |
| [`run_e2e.jl`](run_e2e.jl) | **Single YAML:** `Pkg.instantiate` + reference + EKI (one slice only) |
| [`reference_generation.jl`](reference_generation.jl) | Definitions: `generate_observations_reference!()` (include-only; no auto-run) |
| [`eki_calibration.jl`](eki_calibration.jl) | Definitions: `run_variance_calibration!()` (include-only; no auto-run) |
| [`generate_observations_reference.jl`](generate_observations_reference.jl) | Thin CLI: activate project + `generate_observations_reference!()` |
| [`run_calibration.jl`](run_calibration.jl) | Thin CLI: activate project + `run_variance_calibration!()` |
| [`scripts/sweep_forward_runs.jl`](scripts/sweep_forward_runs.jl) | Grid: cases × `N_quad` × varfix × optional res ladder |
| [`scripts/resolution_ladder.jl`](scripts/resolution_ladder.jl) | Generic `(z_elem, dt, dz_bottom?)` tiers from any column YAML |
| [`forward_sweep_cases.yml`](forward_sweep_cases.yml) | Which model configs + `scm_toml` enter the forward sweep |
| [`scripts/sweep_forward_array.sbatch`](scripts/sweep_forward_array.sbatch) | Slurm body only — **no** embedded `--array`; submit with [`submit_forward_sweep_baseline.sh`](scripts/submit_forward_sweep_baseline.sh) or `sbatch --array=0-$((N-1))` after `--print-task-count` |
| [`scripts/sweep_forward_array_resolution.sbatch`](scripts/sweep_forward_array_resolution.sbatch) | Same for the resolution ladder; [`submit_forward_sweep_resolution.sh`](scripts/submit_forward_sweep_resolution.sh) or manual `sbatch --array=...` |
| [`scripts/submit_forward_sweep_baseline.sh`](scripts/submit_forward_sweep_baseline.sh) | Runs `julia ... --print-task-count --baseline-only`, then `sbatch --array=0-(N-1)` (CLI overrides; no overwriting files) |
| [`scripts/submit_forward_sweep_resolution.sh`](scripts/submit_forward_sweep_resolution.sh) | Same for `--resolution-ladder` |
| [`scripts/run_calibration_sweep.jl`](scripts/run_calibration_sweep.jl) | Reference + EKI per YAML in [`calibration_sweep_configs.jl`](calibration_sweep_configs.jl) (subprocess per slice) |
| [`scripts/calibration_sweep_array.sbatch`](scripts/calibration_sweep_array.sbatch) | Slurm array over `va_calibration_sweep_configs()` (tune `--array` if you edit the list) |
| [`analysis/plotting/`](analysis/plotting/) | Makie scripts + [`run_post_analysis.jl`](analysis/plotting/run_post_analysis.jl) (all-in-one figures) |
| [`analysis/figures/`](analysis/figures/) | Default PNG output from plotting scripts (`.gitkeep` keeps the directory in git) |
| [`cfsite_stub.yml`](cfsite_stub.yml) | **Not implemented** — for GCM-driven / cfSite workflows see [`../gcm_driven_scm/`](../gcm_driven_scm/) |
| [`googleles/README.md`](googleles/README.md) | **Stub** — external columns as **y** or forcing (future) |

Outputs go under **`simulation_output/<CASE>/N_<n>/varfix_on|off/...`** (see sweep script and `experiment_config.yml`).

## Observation vector and noise

Default: one profile, **`thetaa`** (see `va_field_specs` in [`experiment_common.jl`](experiment_common.jl)).

To align with cloud-related diagnostics, add to **`experiment_config.yml`**:

```yaml
observation_fields:
  - short_name: thetaa
  - short_name: hur
  - short_name: cl
observation_noise_std: 0.05
# or per-field: [0.08, 0.15, 0.1]
```

Field names and `period` must match the case YAML **`diagnostics`** block. After any change, **regenerate** the file at **`observations_path`** (e.g. `.../reference/observations.jld2`) so its length matches **`z_elem × n_fields`**.

## Baseline vs quadrature order

True **grid-mean** microphysics (no SGS integral over an SGS PDF) is **`sgs_distribution: mean`** (`GridMeanSGS`). That is different from **`quadrature_order: 1`**, which still evaluates the microphysics closure at **one** Gauss–Hermite node of the configured SGS distribution (so it is not the same as “no quadrature”). The sweep uses **orders 1–5** with the default **lognormal** SGS distribution; document comparisons accordingly.

## Calibration modes (workflow)

`calibration_mode` in **`experiment_config.yml`** is **logged** at EKI start (for your own bookkeeping). Output paths are controlled by **`output_dir`**.

1. **Calibrated, varfix off:** `sgs_quadrature_subcell_geometric_variance: false`, run `run_calibration.jl`.
2. **Calibrated, varfix on:** set flag `true`, point **`output_dir`** to a new folder, run again.
3. **Naive:** calibrate with varfix **off**, then forward runs with varfix **on** and the same EKI **member** parameters — automated by [`scripts/run_naive_varfix_on_forwards.jl`](scripts/run_naive_varfix_on_forwards.jl) and [`run_full_study.jl`](run_full_study.jl) (edit [`va_naive_varfix_off_source_configs()`](calibration_sweep_configs.jl) to choose slices). Default member: latest iteration, member 1; override with **`--eki-iteration`** / **`--eki-member`** on the naive script.

## Commands

**Recommended (one process, one precompile):** from [`run_e2e.jl`](run_e2e.jl)

```bash
cd calibration/experiments/variance_adjustments
julia --project=. run_e2e.jl
```

Optional env: `VA_SKIP_INSTANTIATE=1`, `VA_SKIP_REFERENCE=1` (reuse existing **`observations_path`** from the active YAML), `VA_SKIP_CALIBRATION=1`. **`VA_EXPERIMENT_CONFIG=relative.yml`** selects a non-default experiment YAML for the whole session.

**Step-by-step** (separate processes if you prefer):

```bash
cd calibration/experiments/variance_adjustments
julia --project=. -e 'using Pkg; Pkg.instantiate()'

julia --project=. generate_observations_reference.jl
julia --project=. run_calibration.jl
```

Same session / REPL (after `Pkg.activate(".")`):

```julia
include("run_e2e.jl")
# or, without re-activating inside thin CLIs:
include("reference_generation.jl"); generate_observations_reference!()
include("eki_calibration.jl"); run_variance_calibration!()
```

**Forward-only sweep** (separate runs):

```bash
cd calibration/experiments/variance_adjustments
julia --project=. scripts/sweep_forward_runs.jl --skip-done
julia --project=. scripts/sweep_forward_runs.jl --task-id=0
```

**EKI sweep** (reference + calibration once per YAML in [`scripts/run_calibration_sweep.jl`](scripts/run_calibration_sweep.jl); edit [`calibration_sweep_configs.jl`](calibration_sweep_configs.jl) as needed):

```bash
cd calibration/experiments/variance_adjustments
julia --project=. scripts/run_calibration_sweep.jl
CALIB_SWEEP_TASK_ID=0 julia --project=. scripts/run_calibration_sweep.jl
```

### Analysis

Plotting code lives in **`analysis/plotting/`**. By default, figures go under **`analysis/figures/<CASE>_N<n>_<varfix>_<calibration_mode>/`** (including `profiles/` for case diagnostics) so different YAML slices do not clobber each other.

**All-in-one (recommended after a run):** from the REPL,

```julia
include("analysis/plotting/run_post_analysis.jl")
va_run_post_analysis!()
# Optional: another slice without changing the default experiment_config.yml file:
# va_run_post_analysis!(experiment_config = "experiment_config_trmm_N3_varfix_on.yml")
# Or fix the figure folder explicitly:
# va_run_post_analysis!(figure_root = "analysis/figures/my_run")
```

Optional: `va_run_post_analysis!(profile_paths = [path_to_ref, path_to_member])` to overlay multiple `output_active` dirs on the profile plots.

**What gets plotted**

- **`va_plot_all_case_diagnostic_profiles`** (on by default in `va_run_post_analysis!`): one PNG per variable under **`model_config_path` → `diagnostics`** → `figures/profiles/profile_<short_name>.png`. Legends use the parent folder of `output_active` (e.g. `member_001`, `reference`), not the literal `output_active`.
- **`va_plot_profiles`**: optional. Plots only **`observation_fields`** / default **`thetaa`** — the **EKI observation vector** **y** (saved as `profiles_eki_observation_stack.png` if you pass `do_observation_profiles = true`). If **y** is just `thetaa`, this duplicates `profiles/profile_thetaa.png`; use it when **y** stacks several fields or for a quick EKI-specific layout.

**À la carte** (same directory): `include` individual scripts and call `va_plot_losses()`, `va_plot_parameters()`, etc.

CLI (separate process):

```bash
julia --project=. analysis/plotting/run_post_analysis.jl \
  --experiment-config=experiment_config.yml \
  simulation_output/TRMM_LBA/N_3/varfix_off/reference/output_active

julia --project=. analysis/plotting/plot_profiles.jl path/to/output_active
julia --project=. analysis/plotting/plot_losses.jl path/to/eki_file.jld2
julia --project=. analysis/plotting/plot_parameters.jl path/to/eki_file.jld2 prior.toml
```

Use **`--figures-dir=DIR`** with `run_post_analysis.jl` to override the default per-slice figure root.

### Slurm

- Full pipeline: [`run_calibration.sbatch`](run_calibration.sbatch) runs [`run_e2e.jl`](run_e2e.jl). Set `VA_SKIP_REFERENCE=1` if the **`observations_path`** file for the active YAML already exists on the shared filesystem.
- Forward sweep: **`scripts/submit_forward_sweep_baseline.sh`** or **`scripts/submit_forward_sweep_resolution.sh`** (each queries `--print-task-count` then `sbatch --array=...`). The `.sbatch` files omit `--array` on purpose so the range is not hand-maintained. Custom registry/ladder flags: use the same Julia flags for counting and for the `julia` lines inside the `.sbatch`, then `sbatch --array=0-$((N-1)) ...`.
- **EKI multi-slice:** submit **`scripts/calibration_sweep_array.sbatch`**; edit **`va_calibration_sweep_configs()`** in [`calibration_sweep_configs.jl`](calibration_sweep_configs.jl) and match **`#SBATCH --array`** to that length minus one.
- **Full study (forwards + EKI + figures):** [`run_full_study.sbatch`](run_full_study.sbatch) (add flags to the `julia ... run_full_study.jl` line, e.g. `--skip-forward`).

## Environment variables

| Variable | Effect |
|----------|--------|
| `VA_EXPERIMENT_CONFIG` | Path to experiment YAML (relative to experiment dir unless absolute); default `experiment_config.yml` |
| `VA_SKIP_INSTANTIATE` | If `1`, `run_e2e.jl` skips `Pkg.instantiate()` |
| `VA_SKIP_REFERENCE` | If `1`, `run_e2e.jl` skips `generate_observations_reference!()` |
| `VA_SKIP_CALIBRATION` | If `1`, `run_e2e.jl` skips `run_variance_calibration!()` |
| `JOB_ID` | If set, prepended to ClimaAtmos `job_id` for forward runs |
| `VARIANCE_CALIB_WORKERS` | Worker count for EKI (default: min(4, CPUs−1)) |
| `SWEEP_TASK_ID` / `SLURM_ARRAY_TASK_ID` | Optional: single **forward** sweep task index if you do not pass `--task-id=` (job-array convenience) |
| `NAIVE_SWEEP_TASK_ID` / `SLURM_ARRAY_TASK_ID` | Optional: single naive slice index if you do not pass `--task-id=` on the naive script |
| `CALIB_SWEEP_TASK_ID` / `SLURM_ARRAY_TASK_ID` | Run a single **EKI** slice from `run_calibration_sweep.jl` (see `scripts/run_calibration_sweep.jl`) |
| *(prefer CLI)* | **`run_full_study.jl --help`**, **`scripts/sweep_forward_runs.jl --help`**, **`scripts/run_naive_varfix_on_forwards.jl --help`** — experiment driver options are flags or `run_full_study!(; kwargs...)` in the REPL, not `VA_*` env. |
| `CLIMACALIB_DEVICE` | Passed to `ClimaAtmos.get_comms_context` (`auto`, `CUDADevice`, …). Workers still set `CLIMACOMMS_CONTEXT=SINGLETON` in `run_variance_calibration!()`; adjust if you need a different layout. |

## GPU

Set **`CLIMACALIB_DEVICE=CUDADevice`** (or another device **`get_comms_context`** supports) when CUDA is available. SCM columns are often **CPU-bound**; profile before large sweeps. EKI concurrency is **worker-limited** (**`VARIANCE_CALIB_WORKERS`**); see **Parallel runs and GPU** for **model-level** multi-column GPU batching (not implemented here).

## Revisiting variance in dynamics

Overwriting **prognostic** variance fields or changing **cloud-fraction** covariances is **out of scope** for now; only quadrature inputs are modified. Revisit once the quadrature-only signal is understood.

## Tests

- `Pkg.test("ClimaAtmos")` includes **SGS quadrature** tests and [`test/calibration_experiment_variance_adjustments.jl`](../../../test/calibration_experiment_variance_adjustments.jl) for **`experiment_common.jl`** helpers.
