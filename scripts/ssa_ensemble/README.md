# Sea-salt history-tracer ensemble

Standalone EDMF columns (cfsite GCM forcing, 1M microphysics, 12 sea-salt
bins, passive SSLTH history tracers) emitting sea salt through the
`prognostic_aerosol_column_emission` driver. 24 sites: GPCI transect 16-33
and Southern Ocean 82, 92, 94, 96, 99, 100.

## Files

- `config/ensemble/ssa_history/base_gcmdriven_ssa.yml` — the single
  hand-edited config (stepper block is a placeholder, see comments there).
- `config/ensemble/ssa_history/sites/site<N>.yml` — generated; only
  `cfsite_number` and `job_id` differ from the base.
- `scripts/ssa_ensemble/make_site_configs.sh` — generator (bash + sed).
- `scripts/ssa_ensemble/submit_ensemble.sh` — slurm array job, one CPU task
  per site.

## Command sequence (on the clima server)

```bash
# 0. The probe layout: a worktree and an instantiated env for the commit.
SHA=<short sha>
WT=$HOME/autodebug/probe/wt_$SHA
ENV=$HOME/autodebug/probe/env_$SHA

# 1. Generate the per-site configs inside the worktree.
cd $WT
scripts/ssa_ensemble/make_site_configs.sh      # -> config/ensemble/ssa_history/sites/site*.yml

# 2. Submit the 24-task array.
sbatch --export=ALL,SHA=$SHA scripts/ssa_ensemble/submit_ensemble.sh
#    (or override the layout: --export=ALL,WT=...,ENV=...)

# 3. Watch.
squeue -u $USER
tail -f $HOME/autodebug/logs/ensemble_site23_<jobid>_7.log
```

A single site, without slurm:

```bash
cd $WT && CLIMACOMMS_DEVICE=CPU julia --project=$ENV .buildkite/ci_driver.jl \
    --config_file config/ensemble/ssa_history/sites/site23.yml --job_id ssa_history_site23
```

## Output convention

- Logs: `$HOME/autodebug/logs/ensemble_site<N>_<array job id>_<task id>.log`
  (task id = index into the site list: 0-17 are sites 16-33, 18-23 are
  82, 92, 94, 96, 99, 100).
- Model output: ClimaAtmos' default `output_dir`, i.e.
  `$WT/output/ssa_history_site<N>/output_active/` with hourly-averaged NetCDF
  diagnostics (`<short_name>_1h_average.nc`) and daily restart checkpoints.

## Before submitting

- The 12-bin ClimaParams override path in the base YAML is a server path
  (`/home/zglaser/autodebug/bomex/ssa_12bin.toml`); it must exist, and it must
  not repeat any table of `config/ensemble/ssa_history/prognostic_edmfx_gcmdriven_1M.toml`
  (the GCM-driven EDMF calibration merged with the 1M-only keys), because
  `CP.merge_toml_files` errors on duplicate tables.
- Every `short_name` in the base YAML must be registered or the run fails at
  diagnostics setup (~20 min in). The history-tracer names (`mmrssh*`) and the
  cached rate names (`washrss`, `incrss`) are owned by the SSLTH
  species implementation; adjust them here if that implementation registers
  different names.
