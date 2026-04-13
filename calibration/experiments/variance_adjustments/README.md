# Variance adjustments and SGS quadrature

Branch: **`jb/variance_adjustments`** — **all ClimaAtmos source edits for this study belong on this branch.** `run_variance_calibration!()` (in [`lib/eki_calibration.jl`](lib/eki_calibration.jl)) warns if the repo is on another branch.

## Experiment specification

This section is the **full experiment checklist** (science intent, setups, and infrastructure). Operational commands are in **One command**, **Study tracks**, and **Commands** below.

### Science question

On branch **`jb/variance_adjustments`**, we ask whether **accounting for background gradients in SGS quadrature** improves cloud development. Given local (at some scale) **variance of `q_tot`**, **variance of a thermodynamic variable**, and **their covariance**, does **using the background gradient when constructing quadrature moments** improve cloud representation relative to the reference closure?

### Draft code and thermodynamic choice

- There is an initial draft in [`variance_adjustments.jl`](lib/variance_adjustments.jl). **Treat it as derivation notes, not the source of truth** — validate against the live path: [`src/utils/variance_statistics.jl`](../../../src/utils/variance_statistics.jl) (pure kernels), [`sgs_quadrature_moments_from_gradients`](../../../src/parameterized_tendencies/microphysics/sgs_quadrature.jl) / microphysics wrappers that call [`integrate_over_sgs`](../../../src/parameterized_tendencies/microphysics/sgs_quadrature.jl) (no extra precomputed SGS-moment Fields).
- The draft uses **liquid–ice potential temperature** (\( \theta_{li} \))-style notation in places. **ClimaAtmos SGS quadrature for this closure is implemented in \((q_{tot}, T)\).** The implementation and any new theory should stay aligned with that pair.
- Assume **`ref_includes_gradient = false`** in the reference ClimaAtmos setup unless you deliberately change it.

### Repository discipline

- **Headquarters:** [`calibration/experiments/variance_adjustments/`](.) (this directory). Mirror patterns from other experiments under [`calibration/experiments/`](../).
- **Source changes** that support this study belong **only** on **`jb/variance_adjustments`**; the calibration driver checks the branch and **warns** if you are elsewhere.

### Scope: quadrature inputs only (revisit later)

- **Do not overwrite prognostic variance fields** for now. **Only** the **quadrature setup** (and related SGS saturation adjustment inputs) uses the gradient augmentation. Effects on dynamics and cached covariances elsewhere would multiply confounders; start here, then **revisit** tying dynamics to the same closure once the quadrature-only signal is clear (see also **Revisiting variance in dynamics**).

### Canonical setups, equilibrium, quadrature order, varfix

- Use **GCM-forced SCM column setups** for this workflow via [`model_configs/gcm_forced_column_varquad_hires.yml`](model_configs/gcm_forced_column_varquad_hires.yml) (**operational mesh:** `z_elem: 120`, `dz_bottom: 20` — LES truth is interpolated to this SCM grid) and per-case `les_truth` selectors. **Non-operational** archived coarse meshes live in [`model_configs/*_legacy_coarse.yml`](#legacy-coarse-column-meshes-non-operational) (not referenced by checked-in study YAMLs).
- Run with **1M** bulk microphysics in the provided column YAMLs (`microphysics_model: "1M"`); avoid transient suites unless you extend the experiment.
- Exercise **`quadrature_order` from 1 to 5** in forward sweeps (ClimaAtmos `gauss_hermite` only implements **N ≤ 5**). **`quadrature_order: 1`** is **not** the same as true grid-mean microphysics — it is still one node of the configured SGS PDF; see **Baseline vs quadrature order**.
- Run with **gridscale-corrected SGS quadrature** (**varfix**) **on** and **off**: set ClimaAtmos `sgs_distribution` to a base name (**`lognormal`**, **`gaussian`**, …) vs an explicit gridscale-corrected string (**`lognormal_gridscale_corrected`**, **`gaussian_gridscale_corrected`**, column-tensor / profile–Rosenblatt variants, …), with **separate output trees** per configuration. Effective distribution is **`expc["sgs_distribution"]`** when set, otherwise the merged case YAML’s `sgs_distribution`.

### Vertical resolution and timestep (forward sweep)

- **Coarsening vs. `dt`:** the forward ladder **only coarsens** relative to each case’s baseline YAML. **`dt` is left at the YAML value** for every tier (you **do not** increase `dt` when coarsening). If you later add **refinement** tiers above the baseline, you may need a **smaller** `dt`; that is not implemented by default.
- **Generic ladder** ([`scripts/resolution_ladder.jl`](scripts/resolution_ladder.jl)): built from the **case model YAML** (`z_elem`, `z_max`, `z_stretch`, `dz_bottom`, `dt`) — no per-case Julia tables.
  - **`z_stretch: false`:** coarsen by dividing **`z_elem`** (CLI `--ladder-coarsen-ratio`, default `2`) down to **`--ladder-z-elem-min`**, for **`--ladder-n-tiers`** steps (default `4`).
  - **`z_stretch: true`:** coarsen by multiplying **`dz_bottom`** (`--ladder-min-dz-factor`, default `2`) while keeping **`z_elem`**, using the **same ClimaCore `DefaultZMesh`** ClimaAtmos uses; if the mesh constructor fails, **drop `z_elem`** (same ratio) and **reset `dz_bottom`** to the YAML baseline, then continue.
- **Cases:** listed in [`forward_sweep_cases.yml`](registries/forward_sweep_cases.yml); override with **`--registry=REL.yml`** on [`scripts/sweep_forward_runs.jl`](scripts/sweep_forward_runs.jl). Add rows for new columns (cfSite, GoogleLES, …). Optional key **`forward_sweep_case_slug`** in a model YAML fixes the output folder if two configs share **`initial_condition`**.
- **Forward sweep:** [`scripts/run_full_study.jl`](scripts/run_full_study.jl) runs **`--resolution-ladder`** by default (subprocess flags, not env). Default parameter source is **merged EKI member TOML** → outputs under **`forward_eki/`** (requires calibration first). Use **`--forward-baseline-scm`** for registry **`scm_toml` only** → **`forward_only/`**. Use **`--forward-baseline-only`** for a **single vertical tier** per case (60 tasks with the default registry, `N_quad` 1–5, and six column cases), or run the sweep with **`--baseline-only`**. Output segments: uniform stretch → **`z82_dt150s/`**; stretched → **`z82_dt150s_dzb30/`** (includes effective `dz_bottom` when `z_stretch` is true).
- **EKI** slices still use **fixed** `z_elem`/`dt` from their **`model_config_path`** unless you add YAMLs per tier.

### Calibration: naive vs. calibrated; prior size

This is the **two-workflow** design called out in the study spec — **nothing here replaces** the detailed steps in [**Calibration modes (workflow)**](#calibration-modes-workflow) below.

| | **Naive (frozen varfix-off parameters)** | **Calibrated separately for each varfix** |
|--|--|--|
| **EKI** | **One** calibration with **base** `sgs_distribution` (e.g. **`lognormal`**, varfix **off**). | **Two** calibrations: one YAML varfix **off**, one with **`lognormal_gridscale_corrected`** (or another gridscale-corrected string) — distinct `output_dir` / `observations_path` per slice. Listed in [`va_calibration_sweep_configs()`](lib/calibration_sweep_configs.jl). |
| **Extra forward** | After that EKI, run ClimaAtmos with varfix **on** but the **same** merged member **`parameters.toml`** as a chosen varfix-off ensemble member — **no second EKI** for varfix on. Script: [`scripts/run_naive_varfix_on_forwards.jl`](scripts/run_naive_varfix_on_forwards.jl); sources: [`va_naive_varfix_off_source_configs()`](lib/calibration_sweep_configs.jl). | No “naive” forward: varfix-on performance uses the **varfix-on EKI**’s own optimized parameters. |
| **Where it lives** | Naive run: `simulation_output/<CASE>/N_<n>/varfix_on/naive_from_varfix_off/forward_only/`. Varfix-off calibrated forwards are the usual EKI **`member_*/output_active`** under the varfix-off `output_dir`. | `simulation_output/.../varfix_off/eki/...` and `.../varfix_on/eki/...` (per YAML). |
| **Figures** | Per-slice EKI plots under `analysis/figures/<CASE>_N<n>_varfix_off_<mode>/`; **naive overlay** (reference + naive varfix-on `output_active`) under `analysis/figures/<CASE>_N<n>_naive_varfix_on_from_<calibration_mode>/`. | **Separate** folders per slice, e.g. `..._varfix_off_...` and `..._varfix_on_...`, from [`va_run_post_analysis!`](analysis/plotting/run_post_analysis.jl) for each YAML in the sweep. |

- **Naive (summary):** calibrate with **varfix off** only; then run **forwards with varfix on** using the **same** drawn **`parameters.toml`** (not a second EKI). The **varfix-off** state at calibrated parameters is already in the EKI ensemble; the **naive** step adds the **what if we flip varfix on without re-calibrating?** forward.
- **Calibrated (summary):** run **separate** EKIs for **varfix off** and **varfix on** so you can compare **optimal parameters** and performance.
- Use a **small** set of important parameters in [`config/prior.toml`](config/prior.toml) to stabilize the pipeline; **add more** once runs are routine.

### Outputs and analysis

- Keep **all** experiment outputs under this directory in a **consistent tree** (see **Layout** under **Study tracks**). **`simulation_output/...`** holds ClimaAtmos runs; **`analysis/figures/...`** holds plots.
- Provide **analysis** for **vertical profiles**, **EKI losses**, **parameter trajectories / comparisons**, etc. — [`analysis/plotting/`](analysis/plotting/) and [`run_post_analysis.jl`](analysis/plotting/run_post_analysis.jl).
- **Forward sweep (science grid):** overlay **all completed** `N_quad` × **varfix** cells on one figure per diagnostic and resolution tier — [`va_plot_forward_sweep_comparisons!`](analysis/plotting/plot_forward_sweep_body.jl) (invoked from [`scripts/run_full_study.jl`](scripts/run_full_study.jl) when figures run, or standalone [`analysis/plotting/plot_forward_sweep.jl`](analysis/plotting/plot_forward_sweep.jl)). Default figure root: **`analysis/figures/forward_sweep_eki_calibrated/<CASE>/<resSegment>/profiles/`** (matches **`forward_eki/`** runs); SCM-only sweeps use **`.../forward_sweep_baseline_scm/...`**. This is the layer that answers the README’s **N_quad 1–5** / **varfix** questions; per-YAML EKI figures remain **one slice at a time**.

### Parallel runs, ensemble on GPU, ClimaCalibrate

- It would be useful to run **many uncoupled SCM columns** (e.g. large ensembles) with **high throughput on GPU**. **Process-level** parallelism is what you have today: EKI **workers** (**`VARIANCE_CALIB_WORKERS`**), Slurm **array** jobs, and optional **per-process** **`CLIMACALIB_DEVICE`**. **Model-level** “batch many columns on one GPU” is **not** implemented here — see **Parallel runs and GPU**.

### GPU and Slurm

- **All** experiment driver code paths should remain **GPU-compatible** (`CLIMACALIB_DEVICE`, `get_comms_context`).
- Use the **Slurm** scripts in this directory (and under **`scripts/`**) for cluster submission — [`run_calibration.sbatch`](run_calibration.sbatch), [`run_full_study.sbatch`](run_full_study.sbatch), [`scripts/submit_forward_sweep_baseline.sh`](scripts/submit_forward_sweep_baseline.sh) / [`scripts/submit_forward_sweep_resolution.sh`](scripts/submit_forward_sweep_resolution.sh) (forward grid), [`scripts/calibration_sweep_array.sbatch`](scripts/calibration_sweep_array.sbatch).

### GoogleLES / CloudBench (default EKI sweep)

- **Data:** public on Google Cloud Storage ([Swirl-LM CloudBench README](https://github.com/google-research/swirl-lm/blob/06aefc0f2f152c033d91a3cdbff31519afd995cd/swirl_lm/example/geo_flows/cloud_feedback/README.md)): **`cloudbench-simulation-output/<SITE_ID>/<MONTH>/<EXPERIMENT>/`** contains **`data.zarr`**, **`sounding.csv`**, **`parameters.json`**. **`va_calibration_sweep_configs()`** lists **`experiment_configs/experiment_config_googleles_01`…`_10`**; the deterministic case list is [`googleles_cases_seed42_10.yaml`](registries/googleles_cases_seed42_10.yaml) (regenerate with [`scripts/generate_googleles_registry.jl`](scripts/generate_googleles_registry.jl) if you change sampling).
- **Truth `y`:** [`googleles_truth_build.jl`](googleles_truth_build.jl) reads **`data.zarr`** (horizontal domain means; [`Thermodynamics.potential_temperature`](../../../src/diagnostics/core_diagnostics.jl) for **`thetaa`**; Swirl-style linear liquid fraction for **`q_c` → clw/cli**). Zarr axis order follows the MLCondensateDistributions note **`docs/googleles_zarr_layout.md`** (companion repo / dataset tooling).
- **Forcing:** [`googleles_forcing_build.jl`](googleles_forcing_build.jl) writes a **synthetic** Shen-format NetCDF ( **`tn*` / `wap` = 0** — exploratory; not a full GCM decomposition). **`sounding.csv`** is downloaded to **`simulation_output/_googleles_cache/`** once.
- **GCM-forced cfSite (Zhaoyi tree):** resolving **`les_truth.source: gcm_forced_cfsite`** via **`GCMFORCEDLES_ROOT`** requires **`VA_CALTECH_HPC=1`**. Elsewhere, use **GoogleLES** YAMLs or set **`LES_STATS_FILE`** / **`les_truth.stats_file`** to a **local** `Stats*.nc`.

### Idealized columns (TRMM, DYCOMS, BOMEX, GABLS)

- **Model YAMLs (operational = `*_hires.yml` vertical mesh for `y`):** [`trmm_column_varquad_hires.yml`](model_configs/trmm_column_varquad_hires.yml) (82→**164** levels, same `z_max`), [`dycoms_rf01_column_varquad_hires.yml`](model_configs/dycoms_rf01_column_varquad_hires.yml) / [`dycoms_rf02_column_varquad_hires.yml`](model_configs/dycoms_rf02_column_varquad_hires.yml) (30→**60**), [`bomex_column_varquad_hires.yml`](model_configs/bomex_column_varquad_hires.yml) (60→**120**), [`gabls_column_varquad_hires.yml`](model_configs/gabls_column_varquad_hires.yml) (8→**32**). Older coarse duplicates are [`*_legacy_coarse.yml`](#legacy-coarse-column-meshes-non-operational) only.
- **EKI slices:** `experiment_configs/experiment_config_trmm_N3_varfix_off.yml`, `experiment_configs/experiment_config_dycoms_rf01_N3_varfix_off.yml`, … — set **`LES_STATS_FILE`** or non-empty **`les_truth.stats_file`** to a PyCLES **`Stats*.nc`** before **`scripts/build_observations_from_les.jl`**. TRMM uses layered TOML via **`toml/master_calibrated_baseline_1M.toml`** + **`toml/cases/TRMM_LBA_varquad_overlay_1M.toml`**.
- **Sweep helper:** [`va_idealized_calibration_sweep_configs()`](lib/calibration_sweep_configs.jl) (use `vcat` with the GoogleLES list if you want one multi-slice sweep).

### Uncalibrated forward grid (no EKI, no LES truth)

- **Registry:** [`forward_sweep_cases_uncalibrated.yml`](registries/forward_sweep_cases_uncalibrated.yml): five idealized columns plus five **cfsite_gcm_forcing** artifact groups ([`gcm_uncalibrated_cases_seed43_5.yaml`](registries/gcm_uncalibrated_cases_seed43_5.yaml) lists cfSites **2, 5, 10, 23, 25**; see thin wrappers `model_configs/gcm_forced_column_varquad_site02.yml`, …).
- **Full driver (recommended):** one command runs **`Pkg.instantiate`** → **uncalibrated forward sweep** (SCM-only, same registry) → **forward-sweep figures** (no EKI / naive / per-slice post-analysis):

  ```bash
  julia --project=. scripts/run_full_study.jl --uncalibrated-study 2>&1 | tee logs/uncalibrated_full_study.log
  ```

  Equivalent to **`--skip-calib --skip-naive --skip-les-observations --forward-baseline-scm --forward-registry=registries/forward_sweep_cases_uncalibrated.yml`**. Override the registry with **`--forward-registry=REL.yml`** or env **`VA_FORWARD_SWEEP_REGISTRY`**. Add **`--forward-baseline-only`**, **`--skip-forward`**, **`--skip-figures`**, etc. as usual.

- **Forward sweep only:** `julia --project=. scripts/sweep_forward_runs.jl --registry=registries/forward_sweep_cases_uncalibrated.yml --baseline-scm-forward` (add `--baseline-only` or `--resolution-ladder` as usual). Merges **`prognostic_edmfx.toml`** + optional [`toml/uncalibrated_stability_overlay.toml`](toml/uncalibrated_stability_overlay.toml).
- **Plots:** there is **no calibration** and **no LES observation vector** — for resolution / quadrature comparisons, treat the **finest completed ladder tier** at **`N_quad=3`**, **`varfix_off`** as a **pragmatic stand-in “truth”** when overlaying coarser runs (same baseline parameters throughout).

---

## Goals (short)

Same intent as **Science question** above: test whether **gradient-informed quadrature** improves clouds, using **canonical columns**, **N_quad 1–5**, **varfix on/off**, a **generic vertical-resolution ladder** in the forward sweep (**`dt` fixed** to each case YAML when coarsening), **naive vs calibrated** workflows (**naive** forwards are run by [`scripts/run_full_study.jl`](scripts/run_full_study.jl) from [`va_naive_varfix_off_source_configs()`](lib/calibration_sweep_configs.jl)), **GPU-ready** code, **Slurm** drivers, and **analysis** under this folder.

### Intended science workflow (this is what the default pipeline implements)

**Principle:** you **calibrate** at the study resolution (fixed case YAML: **`z_elem`**, **`dt`**, **`quadrature_order`**, typically **N_quad = 3**) to obtain **meaningful θ**. **Two EKIs** when you need both stories: one with **base** **`sgs_distribution`** (varfix off) and one with an explicit gridscale-corrected string (e.g. **`lognormal_gridscale_corrected`**, **`gaussian_gridscale_corrected`**, profile–Rosenblatt or column-tensor variants), each with its own **`output_dir`** / **`observations_path`**. Everything downstream (reference truth for **`y`**, forwards, resolution ladder) should use **those calibrated parameters** — not ad hoc SCM defaults as if they were “truth.”

- **Observation vector `y`:** EKI **always** reads **`observations.jld2`** at **`observations_path`** before any iteration — that file must exist. **Default track:** **`les_truth.source: googleles_cloudbench`** (CloudBench **`data.zarr`**; see **GoogleLES / CloudBench** above). **Alternative:** build from **LES `Stats*.nc`** (same **`get_profile`** stack as [`calibration/experiments/gcm_driven_scm`](../gcm_driven_scm/run_calibration.jl)) via **[`scripts/build_observations_from_les.jl`](scripts/build_observations_from_les.jl)** and the **`les_truth`** block. **Truth path:** **`les_truth.stats_file`**, **`LES_STATS_FILE`**, **`les_truth.source: googleles_cloudbench`**, or **`gcm_forced_cfsite`** (cluster resolution requires **`GCMFORCEDLES_ROOT`** and **`VA_CALTECH_HPC=1`**). That gives **physical-profile truth** interpolated to the SCM vertical grid (same units as [`process_member_column`](lib/observation_map.jl)). After EKI, **`generate_observations_reference!()`** (optional) can rebuild **`y`** from **SCM + EKI `parameters.toml`** per **`reference_truth_from_eki`**. **`run_calibration_sweep.jl`** only runs **`run_calibration.jl`**. **`run_e2e.jl`** runs calibration first, then (unless **`VA_SKIP_REFERENCE`**) may regenerate **`y`** from EKI. Use **`member: k`** if **`member: best`** is not yet usable (see [`va_eki_best_member_by_obs_loss`](lib/observation_map.jl)).
- **No re-calibration per vertical tier:** the resolution ladder **reuses the same merged EKI member θ** at every coarser tier (**`forward_eki/`**). Parameters are **frozen** across the ladder.
- **Naive workflow:** after varfix-off EKI, **naive** varfix-on forwards use the **same** merged member θ as varfix-off (**no** second EKI for varfix on). See [`scripts/run_naive_varfix_on_forwards.jl`](scripts/run_naive_varfix_on_forwards.jl).
- **Separate varfix on/off calibrations:** when you run **two** EKIs (varfix off and varfix on), the forward grid uses **`eki_varfix_off_config`** for varfix-off cells and **`eki_varfix_on_config`** for varfix-on cells ([`forward_sweep_cases.yml`](registries/forward_sweep_cases.yml)). If **`eki_varfix_on_config`** is omitted, varfix-on ladder cells **reuse the varfix-off merged θ** (frozen-θ / “naive-on-grid” at coarser resolutions), same as the README table in **Calibration modes**.
- **Which θ from EKI (not an ensemble mean):** forwards and naive runs use **one ensemble member’s** `parameters.toml`, combined into **`_atmos_merged_parameters.toml`** with the SCM baseline (“merged” means **baseline TOML + that member’s parameters**, not an average over the ensemble). **Default member selection:** **`--eki-member`** omitted (or **`best`** / **`auto`**) picks the member that minimizes **Mahalanobis squared error** using **`G`** from **`iteration_*/eki_file.jld2`** (`EnsembleKalmanProcesses.get_g_final`), **`y`** from **`observations.jld2`**, and **`Σ`** from **`observation_noise_std`** ([`va_eki_best_member_by_obs_loss`](lib/observation_map.jl)). Aggregate **`error_metrics["loss"]`** in that JLD2 is **one number per EKI iteration**, not one value per ensemble member — it cannot rank members. Set **`--eki-member=k`** to force a fixed index. **Ensemble-mean θ** in parameter space is **not** implemented.

### Pipeline contract (`scripts/run_full_study.jl`)

Use this as a checklist so **runs and plots stay aligned**:

1. **Order:** **`Pkg.instantiate`** → **EKI sweep** → **naive varfix-on forwards** → **forward resolution ladder** → **figures**. The default ladder **always** reads **merged EKI member TOMLs** — that is the calibrated-θ path.
2. **No EKI per ladder tier:** same as **Intended science workflow** above.
3. **Default vs optional branch:** **`forward_eki/`** is the **only** output tree for the main science ladder (**`--eki-calibrated-forward`**, the default). Do not mix it with figures from the optional branch below when interpreting a study.
4. **Plots:** [`va_plot_forward_sweep_comparisons!`](analysis/plotting/plot_forward_sweep_body.jl) writes under **`forward_sweep_eki_calibrated/`** by default. Standalone [`analysis/plotting/plot_forward_sweep.jl`](analysis/plotting/plot_forward_sweep.jl) must use **`--eki-calibrated-forward`** to match those runs.
5. **Naming:** **`--forward-baseline-only`** only reduces **how many vertical tiers** you run (one tier per case); it still uses **merged EKI θ** by default. That is **not** the same as **`--forward-baseline-scm`**.

### Optional: `--forward-baseline-scm` and the `forward_only/` directory (not your calibrated ladder)

**`--eki-calibrated-forward` (default)** loads **`_atmos_merged_parameters.toml`** from this study’s EKI outputs — **this is the calibrated-parameter forward sweep.**

**`--forward-baseline-scm`** / **`forward_only/`** runs with **only** the registry’s **`scm_toml`** and **does not** apply the EKI-merged member file. That means you are **not** using the **optimal θ from this calibration** for those runs. It is only for **debugging, one-off comparisons, or when EKI output is missing**. **Do not use it** for the main workflow you described (calibrate → forward with that θ → ladder). A checked-in SCM TOML might still contain *some* parameters, but it is **not** the θ you just obtained from EKI for this experiment.

**(Unrelated naming)** Naive varfix-on forwards also live under a path segment **`.../naive_from_varfix_off/forward_only/`** — that folder name is **not** the same switch as **`--baseline-scm-forward`**; it is just the layout for those naive runs.

### Observation vector `y` (“truth” for EKI) and plot references

**Vertical resolution of `y`:** [`va_z_centers_column`](lib/experiment_common.jl) builds the target heights from **`model_config_path`**. Checked-in slices use the **`*_hires.yml`** meshes (see **Idealized columns** / **GoogleLES** above for GCM/GoogleLES). Coarser archived meshes are **not** used operationally — see [**Legacy coarse column meshes (non-operational)**](#legacy-coarse-column-meshes-non-operational).

The **first** **`y`** for EKI should come from **LES** (see **`scripts/build_observations_from_les.jl`** and **`les_truth`** above), unless you supply your own **`observations.jld2`**. **Regenerated** **`y`** after calibration uses the **reference** Clima run under **`.../reference/output_active`** (see **`va_reference_output_dir`**): **SCM baseline + EKI `parameters.toml`** via **`reference_truth_from_eki`** ([`reference_generation.jl`](lib/reference_generation.jl)).

That reference is **model-generated** for the same column setup (e.g. a GCM cfSite column), **not** independent LES unless you replaced the pipeline. Forward-sweep **profile** figures overlay **that** reference as a **thick black** line where the plotting code labels it as the calibration reference; a **finer vertical tier** on the ladder may appear as **thick wheat** when present, then the sweep lines. **Scalar** figures (**`lwp`**, **`clivi`**) vs **`N_quad`** are under **`.../<res>/scalars/`** with a black horizontal line for the same reference when applicable. Add **`lwp` / `clivi`** to the case YAML `diagnostics` if missing.

#### Legacy coarse column meshes (non-operational)

Files matching **`model_configs/*_legacy_coarse.yml`** are **archived** vertical meshes (older, fewer `z_elem` / coarser spacing). **No** checked-in `experiment_configs/experiment_config*.yml`, **`config/experiment_config.yml`**, legacy root **`experiment_config.yml`**, or **`registries/forward_sweep_cases*.yml`** row points at them. The **operational** truth / EKI / forward baseline for each case family is the corresponding **`*_hires.yml`** (or `gcm_forced_column_varquad_site*.yml` wrappers that match the study mesh). Use `*_legacy_coarse.yml` only for **bisects**, **A/B tests**, or **reproducing old runs**; they are **not** part of the default study pipeline.

## One command: full README workflow

From **`calibration/experiments/variance_adjustments`**:

```bash
julia --project=. scripts/run_full_study.jl
```

**REPL (kwargs, no env):** `using Pkg; Pkg.activate("."); include("scripts/run_full_study.jl"); run_full_study!()` — or `run_full_study!(; skip_forward = true)`, `run_full_study!(; forward_baseline_only = true)`, etc. See the header in [`scripts/run_full_study.jl`](scripts/run_full_study.jl).

This runs, in order:

1. **EKI sweep** — calibration only for every YAML in [`calibration_sweep_configs.jl`](lib/calibration_sweep_configs.jl) (**requires** existing **`observations.jld2`** per slice; the sweep does **not** generate **`y`** first).
2. **Naive forwards** — for each varfix-off YAML in [`va_naive_varfix_off_source_configs()`](lib/calibration_sweep_configs.jl), one varfix-**on** ClimaAtmos run at the **case YAML** resolution using the **merged member TOML** from the varfix-off EKI (default: latest iteration, member 1). Output: `simulation_output/<CASE>/N_<n>/varfix_on/naive_from_varfix_off/forward_only/`. Script: [`scripts/run_naive_varfix_on_forwards.jl`](scripts/run_naive_varfix_on_forwards.jl).
3. **Forward grid** — every case in [`forward_sweep_cases.yml`](registries/forward_sweep_cases.yml) × **`N_quad` 1–5** × **varfix on/off** × **resolution ladder** from [`scripts/resolution_ladder.jl`](scripts/resolution_ladder.jl) (**~80** runs with the default registry and ladder settings; long). **Default:** merged EKI TOML → `simulation_output/<CASE>/z82_dt150s/N_3/varfix_on/forward_eki/` (stretched grids add `_dzb…` in the segment). **`--forward-baseline-scm`** → `.../forward_only/` instead.
4. **Post-analysis** — (a) **forward-sweep comparison profiles** (`analysis/figures/forward_sweep_eki_calibrated/.../profiles/` or `.../forward_sweep_baseline_scm/...` to match the forward mode); (a′) **scalar vs `N_quad`** (`.../scalars/scalar_lwp_vs_nquad.png`, `scalar_clivi_vs_nquad.png`, …) when outputs include those diagnostics; (b) **naive vs calibrated varfix-on** diagnostic profiles (`analysis/figures/..._naive_vs_calibrated_varfix_on/profiles/`) when the YAML pair exists and runs are on disk; (c) [`va_run_post_analysis!`](analysis/plotting/run_post_analysis.jl) per calibration YAML; (d) **naive overlay** (reference + naive `output_active`, losses/parameters still from varfix-off EKI) under `analysis/figures/<CASE>_N<n>_naive_varfix_on_from_<calibration_mode>/`.

**Fast / partial runs (CLI flags):**

```bash
julia --project=. scripts/run_full_study.jl --skip-forward                    # EKI + naive + figures (needs EKI for naive; forward-sweep PNGs need prior forward runs or are empty)
julia --project=. scripts/run_full_study.jl --figures-only                    # post-analysis only (same as skipping forward, calib, naive, instantiate)
julia --project=. scripts/run_full_study.jl --figures-only --forward-baseline-scm   # figures-only, but forward-sweep PNGs read forward_only/ (not forward_eki/)
julia --project=. scripts/run_full_study.jl --forward-baseline-only           # 20 forwards + rest (no resolution ladder; still uses merged θ unless --forward-baseline-scm)
julia --project=. scripts/run_full_study.jl --forward-baseline-scm            # exploratory: ladder uses registry scm_toml only (forward_only/)
julia --project=. scripts/run_full_study.jl --skip-naive                      # no naive forwards / naive figure overlays
julia --project=. scripts/run_full_study.jl --skip-instantiate --skip-figures # examples
```

**Skips:** `--skip-calib`, `--skip-figures`, `--skip-instantiate`. Reuse finished cells: `--forward-skip-done`, `--naive-skip-done`. **Failures:** by default each sweep **logs and continues** (forward grid, EKI sweep, naive forwards); use **`--forward-fail-fast`**, **`--calib-fail-fast`**, or **`--naive-fail-fast`** on **`scripts/run_full_study.jl`** to stop the subprocess on the first error (same as `--fail-fast` on the underlying scripts). Slurm **one-task-per-job** array runs still fail that job if their single run errors. **`scripts/run_full_study.jl --help`** lists all flags.

**Naive member choice** (on [`scripts/run_naive_varfix_on_forwards.jl`](scripts/run_naive_varfix_on_forwards.jl)): `--eki-iteration=N`, `--eki-member=M` (defaults: latest iteration, member 1).

## Study tracks (pieces)

| Track | What it does | Entry |
|-------|----------------|--------|
| **Full driver** | EKI sweep + **naive** varfix-on forwards + forwards (N×varfix×**resolution ladder** by default; merged θ → **`forward_eki/`**; registry [`forward_sweep_cases.yml`](registries/forward_sweep_cases.yml)) + figures. | [`scripts/run_full_study.jl`](scripts/run_full_study.jl) |
| **A — Single EKI slice** | One YAML only. | [`scripts/run_e2e.jl`](scripts/run_e2e.jl) (calibration first, then optional regenerate **`y`**; see header there) or **`scripts/run_calibration.jl`** alone if **`observations.jld2`** already exists |
| **B — Forward-only grid** | Registry cases × `N_quad` 1–5 × varfix × **`--resolution-ladder`** or **`--baseline-only`**; default **`--eki-calibrated-forward`** (needs EKI outputs on disk). | [`scripts/sweep_forward_runs.jl`](scripts/sweep_forward_runs.jl) (`--help`), [`forward_sweep_cases.yml`](registries/forward_sweep_cases.yml), Slurm: [`submit_forward_sweep_baseline.sh`](scripts/submit_forward_sweep_baseline.sh) / [`submit_forward_sweep_resolution.sh`](scripts/submit_forward_sweep_resolution.sh) plus the `.sbatch` bodies they submit |
| **C — Several EKI slices** | Same as full driver without figures/forward, or Slurm per task. | [`scripts/run_calibration_sweep.jl`](scripts/run_calibration_sweep.jl), [`calibration_sweep_configs.jl`](lib/calibration_sweep_configs.jl) |

**Uniqueness (avoid clobbering):** Whenever **case**, **`quadrature_order`**, **`sgs_distribution`**, or **`observation_fields`** changes, use **distinct** `output_dir` and **`observations_path`** in that YAML (see checked-in templates). Reference ClimaAtmos output defaults to **`simulation_output/<CASE>/N_<n>/varfix_on/reference/`** or **`.../varfix_off/reference/`** unless you set **`reference_output_dir`**.

**Layout:**

- **Reference + observations file:** under **`simulation_output/<CASE>/N_<n>/varfix_on/`** or **`varfix_off/`**, directory **`reference/`** (truth run) and the path in **`observations_path`** (usually `.../reference/observations.jld2`).
- **EKI:** sibling **`eki/`** under the same case / N / varfix branch (or whatever you set in **`output_dir`**).
- **Forward sweep:** `simulation_output/<CASE>/<resSegment>/N_<n>/<varfix_on|off>/forward_eki/` (**default**, merged EKI member TOML) or `.../forward_only/` (**`--baseline-scm-forward`** / **`scripts/run_full_study.jl --forward-baseline-scm`**). Here `resSegment` is `z<z_elem>_dt<dtSlug>`, or (if `z_stretch: true` in the case YAML) `z<z_elem>_dt<dtSlug>_dzb<dz_bottom>`. With the ladder off, there is a **single** tier from the baseline YAML.
- **Naive (varfix on, frozen EKI TOML):** `simulation_output/<CASE>/N_<n>/varfix_on/naive_from_varfix_off/forward_only/` (same `z_elem`/`dt` as the varfix-off slice’s case YAML).
- **Figures (EKI slice):** `analysis/figures/<CASE>_N<n>_<varfix>_<calibration_mode>/` by default from [`va_run_post_analysis!`](analysis/plotting/run_post_analysis.jl) so slices do not overwrite PNGs; naive overlays use `analysis/figures/<CASE>_N<n>_naive_varfix_on_from_<calibration_mode>/`.
- **Figures (naive vs calibrated varfix-on, same plot):** `analysis/figures/<CASE>_N<n>_naive_vs_calibrated_varfix_on/profiles/` — reference, varfix-off EKI member, **naive** varfix-on forward, and varfix-on EKI member with explicit colors/labels ([`va_plot_all_naive_vs_calibrated_varfix_on_profiles!`](analysis/plotting/plot_naive_vs_calibrated_varfix_on.jl)); YAML pairs from [`va_naive_vs_calibrated_varfix_on_yaml_pairs`](lib/calibration_sweep_configs.jl). **Not** the same as **`forward_sweep_*`** folders (those are the cold N×varfix grid overlays).
- **Figures (forward sweep, cross-cell):** `analysis/figures/forward_sweep_eki_calibrated/<CASE>/<resSegment>/profiles/profile_<diagnostic>.png` (or `forward_sweep_baseline_scm/...` for SCM-only sweeps) — one PNG per diagnostic, **multiple** `N_quad` / varfix series when those runs exist. **Encoding:** Wong palette **by `N_quad`**, **dashed = varfix off** / **solid = varfix on**, **thick black = finest completed forward grid** on the ladder (`N=3` **vf_off**) when the panel’s `z_elem` is coarser — *not* the EKI `reference/` truth (that appears in **slice** figures below). Subgrid `*en` diagnostics (e.g. **`clwen`**) can be near zero over much of the column; flat lines near the x-axis are often physical, not bad axis limits.
- **Figures (which EKI slices):** Controlled by [`va_calibration_sweep_configs()`](lib/calibration_sweep_configs.jl) (default: 5 GCM-forced cfSite slices, varfix-off).

## Science goal

**Implementation:** Add **subcell geometric variance** from vertical **background gradients** for **SGS quadrature** in **(q_tot, T)** (ClimaAtmos’s quadrature variables). Cached variances for **cloud fraction** and **mixing length** are **unchanged**; only **microphysics quadrature** and **SGS saturation adjustment** are augmented. Use **`ref_includes_gradient = false`** in the reference setup unless you change it on purpose.

[`variance_adjustments.jl`](lib/variance_adjustments.jl) holds **derivation notes** (e.g. **θ_li** notation); [`src/cache/microphysics_cache.jl`](../../../src/cache/microphysics_cache.jl) defines runtime behavior.

YAML: set **`sgs_distribution`** to an explicit gridscale-corrected string for varfix on (e.g. **`lognormal_gridscale_corrected`**, **`gaussian_gridscale_corrected`**, column-tensor / profile–Rosenblatt names); use base **`lognormal`** / **`gaussian`** for varfix off. ClimaAtmos `get_sgs_distribution` maps each string to a single distribution type (no separate boolean). Related API: `subcell_geometric_variance_increment`, `subcell_geometric_covariance_Tq`, `effective_sgs_quadrature_moments_matched_gaussian`, `sgs_quadrature_moments_from_gradients` (see `variance_statistics.jl` + `sgs_quadrature.jl`).

**Correlation:** Uncorrected SGS → scalar **`correlation_Tq(params)`** (ClimaParams). [`AbstractGridscaleCorrectedSGS`](../../../src/parameterized_tendencies/microphysics/sgs_distribution_types.jl) → **ρ_Tq** from  
`Cov_tot = ρ_param·σ_q,turb·σ_T,turb + (1/12)Δz²(∂T/∂z)(∂q/∂z)`  
with **∂T/∂z ≈ (∂T/∂θ_li)(∂θ_li/∂z)**, normalized by **σ_q,tot·σ_T,tot** (Cauchy–Schwarz floor `ϵ_numerics`), clamped to **[-1, 1]**.

## Where things live

| Topic | Location |
|-------|----------|
| `variance_adjustments.jl` vs implementation | Compare to [`microphysics_cache.jl`](../../../src/cache/microphysics_cache.jl). |
| No prognostic variance overwrite | **Revisiting variance in dynamics**. |
| GCM-forced cfSite cases, `N_quad` | [`model_configs/gcm_forced_column_varquad_hires.yml`](model_configs/gcm_forced_column_varquad_hires.yml) (study mesh), [`scripts/sweep_forward_runs.jl`](scripts/sweep_forward_runs.jl), **Baseline vs quadrature order**. |
| Vertical res ladder | [`scripts/resolution_ladder.jl`](scripts/resolution_ladder.jl), [`forward_sweep_cases.yml`](registries/forward_sweep_cases.yml); **`scripts/run_full_study.jl`** uses **`--resolution-ladder`** unless **`--forward-baseline-only`**. Default ladder forwards use merged EKI TOMLs; **`--forward-baseline-scm`** uses registry SCM only. |
| Varfix on/off | YAML + separate **`output_dir`**. |
| Naive vs calibrated | **Calibration modes (workflow)**. |
| Prior size | [`config/prior.toml`](config/prior.toml). |
| Output paths | **`simulation_output/...`**. |
| Figures | [`analysis/plotting/`](analysis/plotting/); per-slice subfolders via [`va_run_post_analysis!`](analysis/plotting/run_post_analysis.jl). |
| Multi-slice EKI | [`scripts/run_calibration_sweep.jl`](scripts/run_calibration_sweep.jl), [`scripts/calibration_sweep_array.sbatch`](scripts/calibration_sweep_array.sbatch). |
| Slurm | [`run_calibration.sbatch`](run_calibration.sbatch), [`scripts/submit_forward_sweep_baseline.sh`](scripts/submit_forward_sweep_baseline.sh). |
| cfSite / GoogleLES | [`cfsite_stub.yml`](registries/cfsite_stub.yml), [`../gcm_driven_scm/`](../gcm_driven_scm/), [`googleles/README.md`](googleles/README.md). |

## Parallel runs and GPU

A **nice-to-have** (not implemented here) is **model-level GPU batching**: **one** program advances **many** uncoupled SCM columns together—ensemble members with **different parameters**, no column-to-column coupling—using a **batch axis** in the state so kernels exploit **real on-device parallelism** (many columns per launch), instead of needing **one process or one dominant stream per member**. That is **not** the same as EKI **scheduling** members in rounds across **`VARIANCE_CALIB_WORKERS`** Julia processes (process parallelism, still **one** SCM domain per `forward_model`). It would need explicit support in ClimaAtmos and the calibration driver; this experiment does not provide it.

**What you have today:** each `forward_model` is **one** full SCM integration. **[`eki_calibration.jl`](lib/eki_calibration.jl)** uses **`ClimaCalibrate.calibrate`** with **`CAL.WorkerBackend()`**; **`VARIANCE_CALIB_WORKERS`** caps how many of those integrations run **concurrently** (see **Environment variables**). Device selection uses **`CLIMACALIB_DEVICE`** and **`va_comms_ctx()`** in [`model_interface.jl`](lib/model_interface.jl). For **embarrassingly parallel** forward grids, use **[`scripts/sweep_forward_runs.jl`](scripts/sweep_forward_runs.jl)** and **`sbatch --array=...`** (see **[`scripts/submit_forward_sweep_baseline.sh`](scripts/submit_forward_sweep_baseline.sh)**). This **`config/experiment_config.yml`** describes **one** case and **one** observation vector **y**; combining several cases in **one** EKI would need a different forward model or a **minibatch** pattern—see **[`../gcm_driven_scm/`](../gcm_driven_scm/)**.

## Repository layout

| Path | Role |
|------|------|
| [`model_configs/gcm_forced_column_varquad_hires.yml`](model_configs/gcm_forced_column_varquad_hires.yml) | GCM-forced SCM column — **operational mesh** for `y` and EKI (`z_elem: 120`) |
| [`model_configs/gcm_forced_column_varquad_legacy_coarse.yml`](model_configs/gcm_forced_column_varquad_legacy_coarse.yml) | **Non-operational** archive (`z_elem: 60`); see [**Legacy coarse**](#legacy-coarse-column-meshes-non-operational) |
| [`config/experiment_config.yml`](config/experiment_config.yml) | Default EKI slice (`GCM_CFSITE04`, N=3, varfix off) |
| [`config/prior.toml`](config/prior.toml) | EDMF prior (referenced as **`prior_path: config/prior.toml`** from slice YAMLs) |
| [`experiment_configs/`](experiment_configs/) | **Named EKI slice YAMLs** (per-case / per-parameter) |
| [`registries/`](registries/) | Forward-sweep case lists (`forward_sweep_cases*.yml`), GoogleLES sampling list, GCM artifact cfSite list, stubs |
| [`logs/`](logs/) | Driver `tee` targets; [`logs/slurm/`](logs/slurm/) for `#SBATCH --output`. Run [`scripts/move_root_logs.sh`](scripts/move_root_logs.sh) once to sweep stray **`*.log`** from the experiment root into **`logs/`** (git ignores `*.log`) |
| [`lib/`](lib/) | Shared Julia includes: `experiment_common.jl`, `forward_sweep_grid.jl`, EKI / LES / reference drivers, `worker_init.jl` |
| [`experiment_configs/experiment_config_gcm_cfsite04_N3_varfix_off.yml`](experiment_configs/experiment_config_gcm_cfsite04_N3_varfix_off.yml) | GCM-forced LES case `cfsite=4`, N=3, varfix off |
| [`experiment_configs/experiment_config_gcm_cfsite08_N3_varfix_off.yml`](experiment_configs/experiment_config_gcm_cfsite08_N3_varfix_off.yml) | GCM-forced LES case `cfsite=8`, N=3, varfix off |
| [`experiment_configs/experiment_config_gcm_cfsite11_N3_varfix_off.yml`](experiment_configs/experiment_config_gcm_cfsite11_N3_varfix_off.yml) | GCM-forced LES case `cfsite=11`, N=3, varfix off |
| [`experiment_configs/experiment_config_gcm_cfsite19_N3_varfix_off.yml`](experiment_configs/experiment_config_gcm_cfsite19_N3_varfix_off.yml) | GCM-forced LES case `cfsite=19`, N=3, varfix off |
| [`experiment_configs/experiment_config_gcm_cfsite23_N3_varfix_off.yml`](experiment_configs/experiment_config_gcm_cfsite23_N3_varfix_off.yml) | GCM-forced LES case `cfsite=23`, N=3, varfix off |
| [`experiment_configs/experiment_config_multifield_example.yml`](experiment_configs/experiment_config_multifield_example.yml) | Comments for stacking `thetaa` / `hur` / `cl` in **y** |
| [`scripts/run_full_study.jl`](scripts/run_full_study.jl) | **Entry:** activates project, includes [`lib/run_full_study.jl`](lib/run_full_study.jl) (full pipeline + `FullStudyOptions`) |
| [`lib/run_full_study.jl`](lib/run_full_study.jl) | Full study implementation (EKI sweep, naive, forward grid, figures) |
| [`lib/experiment_common.jl`](lib/experiment_common.jl) | Shared: observation field specs, noise matrix, `job_id`, comms context |
| [`lib/calibration_sweep_configs.jl`](lib/calibration_sweep_configs.jl) | `va_calibration_sweep_configs()` (EKI + figures) and `va_naive_varfix_off_source_configs()` (naive track) |
| [`scripts/run_naive_varfix_on_forwards.jl`](scripts/run_naive_varfix_on_forwards.jl) | Naive varfix-on forwards from varfix-off EKI member TOML |
| [`scripts/run_e2e.jl`](scripts/run_e2e.jl) | **Single YAML:** `Pkg.instantiate` + reference + EKI (one slice only) |
| [`scripts/run_calibration.jl`](scripts/run_calibration.jl) | EKI only (needs existing **`observations.jld2`**) |
| [`scripts/build_observations_from_les.jl`](scripts/build_observations_from_les.jl) | Build **`observations.jld2`** from LES **`Stats*.nc`** |
| [`scripts/generate_observations_reference.jl`](scripts/generate_observations_reference.jl) | Regenerate **`y`** from reference SCM + EKI member TOML |
| [`scripts/move_root_logs.sh`](scripts/move_root_logs.sh) | Move stray **`*.log`** from experiment root into **`logs/`** |
| [`lib/les_truth_build.jl`](lib/les_truth_build.jl) | Build **`observations.jld2`** from LES **`Stats*.nc`** (`get_profile` from [`gcm_driven_scm`](../gcm_driven_scm/helper_funcs.jl)) |
| [`lib/reference_generation.jl`](lib/reference_generation.jl) | Definitions: `generate_observations_reference!()` (include-only; no auto-run) |
| [`lib/eki_calibration.jl`](lib/eki_calibration.jl) | Definitions: `run_variance_calibration!()` (include-only; no auto-run) |
| [`scripts/sweep_forward_runs.jl`](scripts/sweep_forward_runs.jl) | Grid: cases × `N_quad` × varfix × optional res ladder |
| [`scripts/resolution_ladder.jl`](scripts/resolution_ladder.jl) | Generic `(z_elem, dt, dz_bottom?)` tiers from any column YAML |
| [`registries/forward_sweep_cases.yml`](registries/forward_sweep_cases.yml) | EKI-calibrated forward sweep (merged member TOML); see [`registries/forward_sweep_cases_uncalibrated.yml`](registries/forward_sweep_cases_uncalibrated.yml) for **baseline-only** grids |
| [`registries/forward_sweep_cases_uncalibrated.yml`](registries/forward_sweep_cases_uncalibrated.yml) | Uncalibrated cases × `N_quad` × varfix (`--baseline-scm-forward`; no `eki_varfix_off_config`) |
| [`experiment_configs/experiment_config_trmm_N3_varfix_off.yml`](experiment_configs/experiment_config_trmm_N3_varfix_off.yml) (and `dycoms_rf01`, `dycoms_rf02`, `bomex`, `gabls` in the same folder) | Idealized + PyCLES `Stats*.nc` (`les_truth.stats_file` / `LES_STATS_FILE`) |
| [`scripts/sweep_forward_array.sbatch`](scripts/sweep_forward_array.sbatch) | Slurm body only — **no** embedded `--array`; submit with [`submit_forward_sweep_baseline.sh`](scripts/submit_forward_sweep_baseline.sh) or `sbatch --array=0-$((N-1))` after `--print-task-count` |
| [`scripts/sweep_forward_array_resolution.sbatch`](scripts/sweep_forward_array_resolution.sbatch) | Same for the resolution ladder; [`submit_forward_sweep_resolution.sh`](scripts/submit_forward_sweep_resolution.sh) or manual `sbatch --array=...` |
| [`scripts/submit_forward_sweep_baseline.sh`](scripts/submit_forward_sweep_baseline.sh) | Runs `julia ... --print-task-count --baseline-only`, then `sbatch --array=0-(N-1)` (CLI overrides; no overwriting files) |
| [`scripts/submit_forward_sweep_resolution.sh`](scripts/submit_forward_sweep_resolution.sh) | Same for `--resolution-ladder` |
| [`scripts/run_calibration_sweep.jl`](scripts/run_calibration_sweep.jl) | EKI per YAML in [`lib/calibration_sweep_configs.jl`](lib/calibration_sweep_configs.jl) (in-process sweep); **does not** run reference generation |
| [`scripts/calibration_sweep_array.sbatch`](scripts/calibration_sweep_array.sbatch) | Slurm array over `va_calibration_sweep_configs()` (tune `--array` if you edit the list) |
| [`analysis/plotting/`](analysis/plotting/) | Makie scripts + [`run_post_analysis.jl`](analysis/plotting/run_post_analysis.jl) (all-in-one figures) |
| [`analysis/figures/`](analysis/figures/) | Default PNG output from plotting scripts (`.gitkeep` keeps the directory in git) |
| [`registries/cfsite_stub.yml`](registries/cfsite_stub.yml) | **Not implemented** — for GCM-driven / cfSite workflows see [`../gcm_driven_scm/`](../gcm_driven_scm/) |
| [`googleles/README.md`](googleles/README.md) | **Stub** — external columns as **y** or forcing (future) |

Outputs go under **`simulation_output/<CASE>/N_<n>/varfix_on|off/...`** (see sweep script and **`config/experiment_config.yml`**).

## Observation vector and noise

Default: one profile, **`thetaa`** (see `va_field_specs` in [`experiment_common.jl`](lib/experiment_common.jl)).

To align with cloud-related diagnostics, add to **`config/experiment_config.yml`** (or your active slice YAML):

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

`calibration_mode` in the active experiment YAML (often **`config/experiment_config.yml`**) is **logged** at EKI start (for your own bookkeeping). Output paths are controlled by **`output_dir`**.

1. **Calibrated, varfix off:** base **`sgs_distribution`** (e.g. **`lognormal`**), run **`scripts/run_calibration.jl`** (or include that YAML in [`va_calibration_sweep_configs()`](lib/calibration_sweep_configs.jl) and run [`scripts/run_calibration_sweep.jl`](scripts/run_calibration_sweep.jl)).
2. **Calibrated, varfix on:** set **`sgs_distribution: lognormal_gridscale_corrected`** (or another gridscale-corrected string), point **`output_dir`** and **`observations_path`** to a **new** varfix-on tree, run again — this is the **second calibration** (parameters re-optimized for varfix on). Use a separate YAML cloned from one of the `experiment_configs/experiment_config_gcm_cfsite*_N3_varfix_off.yml` files.
3. **Naive:** after (1) only — forward run(s) with varfix **on** and the **same** EKI **member** **`parameters.toml`** as in (1), **without** running (2). Automated by [`scripts/run_naive_varfix_on_forwards.jl`](scripts/run_naive_varfix_on_forwards.jl) and [`scripts/run_full_study.jl`](scripts/run_full_study.jl) (edit [`va_naive_varfix_off_source_configs()`](lib/calibration_sweep_configs.jl) to choose which varfix-off slices get a naive varfix-on forward). Default member: latest iteration, member 1; override with **`--eki-iteration`** / **`--eki-member`** on the naive script.

## Commands

**Recommended (one process, one precompile):** [`scripts/run_e2e.jl`](scripts/run_e2e.jl)

```bash
cd calibration/experiments/variance_adjustments
julia --project=. scripts/run_e2e.jl
```

Optional env: `VA_SKIP_INSTANTIATE=1`, `VA_SKIP_REFERENCE=1` (reuse existing **`observations_path`** from the active YAML), `VA_SKIP_CALIBRATION=1`. **`VA_EXPERIMENT_CONFIG=relative.yml`** selects a non-default experiment YAML for the whole session.

**Step-by-step** (separate processes if you prefer):

```bash
cd calibration/experiments/variance_adjustments
julia --project=. -e 'using Pkg; Pkg.instantiate()'

julia --project=. scripts/generate_observations_reference.jl
julia --project=. scripts/run_calibration.jl
```

Same session / REPL (after `Pkg.activate(".")`):

```julia
empty!(ARGS); include("scripts/run_e2e.jl")
# or step-by-step (no full e2e driver):
include("lib/reference_generation.jl"); generate_observations_reference!()
include("lib/eki_calibration.jl"); run_variance_calibration!()
```

**Forward-only sweep** (separate runs):

```bash
cd calibration/experiments/variance_adjustments
julia --project=. scripts/sweep_forward_runs.jl --skip-done
julia --project=. scripts/sweep_forward_runs.jl --task-id=0
```

**EKI sweep** (calibration once per YAML in [`scripts/run_calibration_sweep.jl`](scripts/run_calibration_sweep.jl); each slice must already have **`observations.jld2`** at **`observations_path`**; edit [`calibration_sweep_configs.jl`](lib/calibration_sweep_configs.jl) as needed):

```bash
cd calibration/experiments/variance_adjustments
julia --project=. scripts/run_calibration_sweep.jl
CALIB_SWEEP_TASK_ID=0 julia --project=. scripts/run_calibration_sweep.jl
```

### Analysis

Plotting code lives in **`analysis/plotting/`**. By default, figures go under **`analysis/figures/<CASE>_N<n>_<varfix>_<calibration_mode>/`** (including `profiles/` for case diagnostics) so different YAML slices do not clobber each other.

**Forward sweep (multi-`N_quad`, multi-varfix):** [`plot_forward_sweep_body.jl`](analysis/plotting/plot_forward_sweep_body.jl) implements **`va_plot_forward_sweep_comparisons!`**, which scans the sweep grid and overlays every existing **`output_active`** for that case and resolution tier. **[`scripts/run_full_study.jl`](scripts/run_full_study.jl)** calls it at the start of the figures phase with the same **`forward_resolution_ladder`**, **`forward_baseline_scm`**, and EKI/baseline parameter mode as the forward subprocess. If you only rerun figures (**`--figures-only`**), the driver still defaults to **ladder on** and **EKI-parameter** paths; use **`julia --project=. analysis/plotting/plot_forward_sweep.jl --baseline-only`** when your forwards used a single tier, and add **`--baseline-scm-forward`** if those runs live under **`forward_only/`** (not **`forward_eki/`**).

**All-in-one (recommended after a run):** from the REPL,

```julia
include("analysis/plotting/run_post_analysis.jl")
va_run_post_analysis!()
# Optional: another slice without changing the default config/experiment_config.yml file:
# va_run_post_analysis!(experiment_config = "experiment_configs/experiment_config_gcm_cfsite23_N3_varfix_off.yml")
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
  --experiment-config=config/experiment_config.yml \
  simulation_output/GCM_CFSITE04/N_3/varfix_off/reference/output_active

julia --project=. analysis/plotting/plot_profiles.jl path/to/output_active
julia --project=. analysis/plotting/plot_forward_sweep.jl   # optional: --eki-calibrated-forward, --baseline-scm-forward, --baseline-only, --figures-dir=...
julia --project=. analysis/plotting/plot_losses.jl path/to/eki_file.jld2
julia --project=. analysis/plotting/plot_parameters.jl path/to/eki_file.jld2 config/prior.toml
```

Use **`--figures-dir=DIR`** with `run_post_analysis.jl` to override the default per-slice figure root.

### Slurm

- Full pipeline: [`run_calibration.sbatch`](run_calibration.sbatch) runs [`scripts/run_e2e.jl`](scripts/run_e2e.jl). Set `VA_SKIP_REFERENCE=1` if the **`observations_path`** file for the active YAML already exists on the shared filesystem.
- Forward sweep: **`scripts/submit_forward_sweep_baseline.sh`** or **`scripts/submit_forward_sweep_resolution.sh`** (each queries `--print-task-count` then `sbatch --array=...`). The `.sbatch` files omit `--array` on purpose so the range is not hand-maintained. Custom registry/ladder flags: use the same Julia flags for counting and for the `julia` lines inside the `.sbatch`, then `sbatch --array=0-$((N-1)) ...`.
- **EKI multi-slice:** submit **`scripts/calibration_sweep_array.sbatch`**; edit **`va_calibration_sweep_configs()`** in [`calibration_sweep_configs.jl`](lib/calibration_sweep_configs.jl) and match **`#SBATCH --array`** to that length minus one.
- **Full study (forwards + EKI + figures):** [`run_full_study.sbatch`](run_full_study.sbatch) (add flags to the `julia ... scripts/run_full_study.jl` line, e.g. `--skip-forward`).

## Environment variables

| Variable | Effect |
|----------|--------|
| `VA_GCM_FORCING_FILE` | Overrides **`external_forcing_file`** for GCM column YAMLs: EKI via [`model_interface.jl`](lib/model_interface.jl), and **`scripts/sweep_forward_runs.jl`** (same `artifact"cfsite_gcm_forcing"/...` default if unset). The artifact’s groups are **`site2`**, **`site4`**, **`site23`**, … (**no** zero-padding). Use an absolute path to the same schema if resolution fails. |
| `VA_GOOGLELES_FORCING_FILE` | Overrides **`external_forcing_file`**: EKI when **`les_truth.source: googleles_cloudbench`**; forward sweep also honors it if set (after `VA_GCM_FORCING_FILE` when both are set, this wins). |
| `VA_GOOGLELES_CFSITE_GROUP` | Optional; forward sweep only: overrides **`cfsite_number`** in the case YAML (must match a group in the NetCDF from **`VA_GOOGLELES_FORCING_FILE`**, e.g. **`site_googleles_01`**). |
| `VA_GOOGLELES_ZARR_ROOT` | Optional local directory prefix for **`…/<site>/<month>/<exp>/data.zarr`** (mirrors GCS layout) when HTTP access to **`cloudbench-simulation-output`** is unavailable |
| `VA_CALTECH_HPC` | Set to **`1`**, **`true`**, or **`yes`** to allow **`les_truth.source: gcm_forced_cfsite`** resolution via **`GCMFORCEDLES_ROOT`** (cluster filesystem). Not required if **`LES_STATS_FILE`** / **`les_truth.stats_file`** points to a local **`Stats*.nc`**. |
| `GCMFORCEDLES_ROOT` | **Required** for automatic `les_truth.source: gcm_forced_cfsite` resolution (with **`VA_CALTECH_HPC`**): root directory containing `cfsite/<MM>/<model>/<exp>/...` (same tree as `gcm_driven_scm/get_les_metadata.jl`). |
| `LES_STATS_FILE` | Override path to LES `Stats*.nc` when building **`observations.jld2`** (also used by **`scripts/run_full_study.jl`** auto-build) |
| `VA_SKIP_LES_OBSERVATIONS_BUILD` | If `1`, **`scripts/run_full_study.jl`** does not build missing **`observations.jld2`** before EKI (same as **`--skip-les-observations`**) |
| `VA_FORWARD_SWEEP_REGISTRY` | Optional: relative or absolute **`--registry=`** for the forward sweep + forward-sweep figures in **`scripts/run_full_study.jl`** (same as **`--forward-registry=`**) |
| `VA_UNCALIBRATED_STUDY` | If **`1`** / **`true`** / **`yes`**, same preset as **`scripts/run_full_study.jl --uncalibrated-study`** |
| `VA_FORCE_LES_OBSERVATIONS` | If `1`, **`run_full_study`** rebuilds **`observations.jld2`** even when already present |
| `VA_EXPERIMENT_CONFIG` | Path to experiment YAML (relative to experiment dir unless absolute); default `config/experiment_config.yml` |
| `VA_SKIP_INSTANTIATE` | If `1`, **`scripts/run_e2e.jl`** skips `Pkg.instantiate()` |
| `VA_SKIP_REFERENCE` | If `1`, **`scripts/run_e2e.jl`** skips regenerating **`y`** after calibration (default: run **`generate_observations_reference!()`** **after** EKI so **`reference_truth_from_eki`** can apply) |
| `VA_SKIP_CALIBRATION` | If `1`, **`scripts/run_e2e.jl`** skips `run_variance_calibration!()` |
| `JOB_ID` | If set, prepended to ClimaAtmos `job_id` for forward runs |
| `VARIANCE_CALIB_WORKERS` | Worker count for EKI (default: min(4, CPUs−1)) |
| `SWEEP_TASK_ID` / `SLURM_ARRAY_TASK_ID` | Optional: single **forward** sweep task index if you do not pass `--task-id=` (job-array convenience) |
| `NAIVE_SWEEP_TASK_ID` / `SLURM_ARRAY_TASK_ID` | Optional: single naive slice index if you do not pass `--task-id=` on the naive script |
| `CALIB_SWEEP_TASK_ID` / `SLURM_ARRAY_TASK_ID` | Run a single **EKI** slice from `run_calibration_sweep.jl` (see `scripts/run_calibration_sweep.jl`) |
| *(prefer CLI)* | **`scripts/run_full_study.jl --help`**, **`scripts/sweep_forward_runs.jl --help`**, **`scripts/run_naive_varfix_on_forwards.jl --help`** — experiment driver options are flags or `run_full_study!(; kwargs...)` in the REPL, not `VA_*` env. |
| `CLIMACALIB_DEVICE` | Passed to `ClimaAtmos.get_comms_context` (`auto`, `CUDADevice`, …). Workers still set `CLIMACOMMS_CONTEXT=SINGLETON` in `run_variance_calibration!()`; adjust if you need a different layout. |

## GPU

Set **`CLIMACALIB_DEVICE=CUDADevice`** (or another device **`get_comms_context`** supports) when CUDA is available. SCM columns are often **CPU-bound**; profile before large sweeps. EKI concurrency is **worker-limited** (**`VARIANCE_CALIB_WORKERS`**); see **Parallel runs and GPU** for **model-level** multi-column GPU batching (not implemented here).

## Revisiting variance in dynamics

Overwriting **prognostic** variance fields or changing **cloud-fraction** covariances is **out of scope** for now; only quadrature inputs are modified. Revisit once the quadrature-only signal is understood.

## Tests

- `Pkg.test("ClimaAtmos")` includes **SGS quadrature** tests and [`test/calibration_experiment_variance_adjustments.jl`](../../../test/calibration_experiment_variance_adjustments.jl) for **`experiment_common.jl`** helpers.
