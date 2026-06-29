---
name: scm-les-tuning
description: Sweep ClimaAtmos single-column (SCM) EDMF/microphysics toml parameters to match path-integrated quantities (IWP, SWP, LWP, RWP, cloud top) against LES reference data such as Stats.TRMM_LBA.nc or Stats.DYCOMS_RF01.nc. Use when tuning microphysics or EDMF parameters for a config/model_configs/*_column.yml run against an LES Stats.*.nc file, or when asked to "match LES", "tune against LES", or investigate why a column run produces zero/wrong IWP-SWP-LWP-RWP.
---

# SCM-vs-LES parameter tuning

Iterative workflow: change one or a few toml parameters, run the column config,
compare path-integrated output against LES, checkpoint with the user, repeat.
Never commit anything — this is exploratory. Keep all scratch files in one
sweep directory (e.g. `/tmp/<case>_sweep/`) outside the repo.

## Before sweeping: confirm the environment works

```bash
julia --project=.buildkite -e 'using Pkg; Pkg.instantiate()'
```
Run once per machine if `Pkg.status()` shows undownloaded (`→`) packages —
otherwise the first `ci_driver.jl` run fails with a precompile error.

## One sweep run = one toml + one yml

`ClimaParams.merge_toml_files` errors on duplicate keys across files
(`override=false` by default), so **don't** layer multiple small toml files —
copy the full base toml per run and edit only the keys that change:

```bash
python3 .claude/skills/scm-les-tuning/scripts/set_toml_value.py \
  toml/prognostic_edmfx_1M.toml /tmp/trmm_sweep/myrun.toml \
  detr_massflux_vertdiv_coeff=0.1 fixed_snow_terminal_velocity=0.25
```

Then write a matching yml that points at it, with an explicit `output_dir`
(`RemovePreexisting` avoids numbered `ActiveLink` subfolders):

```bash
cat > /tmp/trmm_sweep/myrun.yml <<EOF
toml: ["/tmp/trmm_sweep/myrun.toml"]
job_id: "trmm_sweep_myrun"
output_dir: "/tmp/trmm_sweep/output_myrun"
output_dir_style: "RemovePreexisting"
EOF
```

## Running

`--config_file` can be passed multiple times and merges left-to-right (later
wins), so layer the override yml on top of the case's base config:

```bash
julia --project=.buildkite .buildkite/ci_driver.jl \
  --config_file config/model_configs/prognostic_edmfx_trmm_column.yml \
  --config_file /tmp/trmm_sweep/myrun.yml \
  > /tmp/trmm_sweep/myrun.log 2>&1
```

A normal 6h column run takes ~10s. **Extreme parameter values can take
minutes** (numerical stiffness, not a bug) — that's a signal worth reporting,
not just waiting out. Always `echo "EXIT $?"` / check the log for a clean
finish, not just process exit.

For more than one or two runs, batch them in a single backgrounded shell loop
(`for name in a b c; do julia ... ; done`) rather than one Bash call per run —
each `julia` invocation pays ~5-10s of startup/compile overhead.

### Don't let the laptop sleep mid-sweep

A backgrounded multi-run loop can take many minutes. Find the **parent shell
PID** running the `for` loop (not a single `julia` PID, which exits between
runs) and pin caffeinate to it:

```bash
PID=$(ps aux | grep "[z]sh -c.*myrun" | awk '{print $2}' | head -1)
caffeinate -i -w $PID &
```
Kill the caffeinate process once the batch's completion notification arrives.

## Extracting and comparing output

NetCDF diagnostics are tarred in `output_dir`:

```bash
mkdir -p /tmp/trmm_sweep/extracted_myrun
tar -xf /tmp/trmm_sweep/output_myrun/nc_files.tar -C /tmp/trmm_sweep/extracted_myrun
```

Compare hour-6 (or whatever the run's `t_end` is) path-integrated values
against the LES `timeseries` group as a percentage of the LES value:

```bash
julia --project=.buildkite .claude/skills/scm-les-tuning/scripts/compare_to_les.jl \
  /Users/zshen/Work/clima_repo/les_data/Stats.TRMM_LBA.nc \
  /tmp/trmm_sweep iwp,swp \
  baseline=extracted_baseline myrun=extracted_myrun
```

**Gotcha:** the `cl` (cloud fraction) diagnostic is in **percent**, not a
0-1 fraction. A naive `cl > 1e-3` threshold for "cloud top" catches trace
cloud everywhere and gives meaningless (huge) cloud-top heights for every
config indiscriminately. Use `cl > 1.0` (1%) or `cl > 5.0` (5%) instead, and
sanity-check against the LES `timeseries/cloud_top` value, which is usually
much higher than intuition suggests for deep convective cases (TRMM-LBA tops
out near 13-14km, not the cumulus-cloud heights you might expect).

**`cl` dimensions are `(time, z)`** — not `(z, time)`. To get the profile at
the last time step: `cl[end, :]`. The LES `timeseries/cloud_fraction` is in
0-1; model `cl` column-max is in % — compare directly as numbers (e.g. model
45 ≈ LES 0.45 × 100). `clt` (CMIP total-column cloud fraction) requires
radiation mode and is not available in the standard TRMM-LBA column config.
The compare_to_les.jl script supports `cl:cloud_fraction` name remapping and
takes column-max for profile variables automatically.

## Process: checkpoint after every round

Don't chain many rounds of sweeps autonomously. After each round:
1. Report the comparison table (param values vs. % of LES) plainly.
2. Flag anything surprising: non-monotonic response, a knob that overshoots
   in one direction, two metrics that trade off against each other and can't
   both hit 100% with the same knob, runs that took unusually long.
3. Ask the user how to proceed (fine-tune around the best point, try a
   different lever, accept the current result, etc.) — don't assume the next
   experiment.

A parameter that improves one target metric while making another worse (or
that helps until some threshold and then overshoots) is the norm, not the
exception — call it out rather than reporting only the metric that improved.

## Known-good levers for this repo (TRMM-LBA column case)

Starting from `config/model_configs/prognostic_edmfx_trmm_column.yml` /
`toml/prognostic_edmfx_1M.toml` committed defaults, the baseline column run
produces **zero** ice/snow because convection stays too shallow (cloud top
~1500m) to reach the freezing level — microphysics params alone can't fix
that. `detr_massflux_vertdiv_coeff` (default 1.0, EDMF detrainment vertical-
divergence coefficient) is the dynamics lever that unlocks deep convection;
~0.1 gets cloud top within ~100m of LES and IWP/SWP from 0% to 40-50% of
target. `pressure_normalmode_drag_coeff` (the more obvious-looking dynamics
knob) has almost no effect by comparison. Once convection is deep, the
microphysics timescales (`sublimation_deposition_timescale` for ice,
`condensation_evaporation_timescale` for liquid) are far stronger, more
direct levers on IWP/LWP than the size-relation/terminal-velocity params,
and can land both members of a pair (e.g. LWP and RWP) close to 100%
simultaneously at the right timescale — don't reach for terminal velocity or
chia first.

### Snow source pathway isolation (TRMM-LBA, detr_massflux_vertdiv_coeff=0.1 baseline)

Diagnosed by disabling each process individually (yml config switches, not
toml params). Contribution to SWP relative to baseline (40% of LES):

| Process disabled | yml key | SWP change |
|-----------------|---------|------------|
| Vapor deposition onto snow | `snow_deposition_sublimation: SublimationOnly` | 40% → 25% (dominant source) |
| Ice–snow accretion | `cloud_ice_snow_collision_efficiency=0` | 40% → 36% |
| Liquid riming onto snow | `cloud_liquid_snow_collision_efficiency=0` | 40% → 37% |
| Rain–snow freezing | `rain_snow_collision_efficiency=0` | negligible |

Disabling deposition also improves IWP (43%→88%) and LWP (48%→69%) because
ice and liquid that would have deposited onto snow instead remain in those
reservoirs. All three metrics improve simultaneously.

### Tuning snow deposition rate without disabling it

Physical parameters for snow vapor deposition in CloudMicrophysics 1M:
- `snow_ventilation_coefficient_b` (default 0.44, in ClimaParams registry, not in prognostic_edmfx_1M.toml — must **append** to sweep toml): ventilation slope in `f_v = a + b·Sc^(1/3)·Re^(1/2)`. **Very weak lever**: halving (0.44→0.22) moves SWP by only ~2 percentage points. The deposition rate is not ventilation-limited in this regime.
- `fixed_snow_terminal_velocity` (default 1.0 m/s): faster fallout reduces snow column residence time. Doubling (1.0→2.0) moves SWP by ~7 points. Larger effect than ventilation coefficient, but competing: higher vt also enhances ventilation slightly.
- Combined b=0.11 + vt=2.0: SWP 40%→31% — still far from a ~10% target.

**Floor on SWP via deposition-only changes**: even completely removing
deposition gives SWP=25%. The remaining 25% comes from accretion and
autoconversion. Reaching ~10% SWP requires also attacking those sources
(collision efficiencies, terminal velocity much higher than 2 m/s) or
accepting a higher floor.

### `cloud_fraction_steepness_scale` (cf_steepness_coeff)

Default 1.0; not in `prognostic_edmfx_1M.toml` — must **append** to sweep
toml. Controls `α = 1/k` which scales the effective SGS standard deviation
used in the truncated-Gaussian cloud fraction:

    σ_aug = α × sqrt(σ_S² + σ_S_floor²),   CF = Φ(z),  C = q_c / σ_aug

Higher k → smaller α → narrower PDF → higher CF for the same q_c (more
"all-or-nothing" cloud). Lower k → lower CF, more fractional.

**TRMM-LBA sweep results** (detr_massflux_vertdiv_coeff=0.1 base):

| k   | IWP | SWP | LWP | cl col-max |
|-----|-----|-----|-----|-----------|
| 0.5 | 38% | 32% | 51% | 43%       |
| 1.0 | 43% | 40% | 48% | 45%       |
| 2.0 | 33% | 25% | 16% | 48%       |
| 4.0 | 47% | 29% | 14% | ~79% peak |

LES column cloud fraction ≈ 50% (model `cl` is in %, LES `cloud_fraction`
is 0-1 — compare model col-max directly to LES value × 100).

**Profile behavior**: increasing k redistributes cloud fraction from low
levels (0–3km decreases) to mid-upper levels (5–13km increases strongly).
At k=4, cloud at 8–11km reaches 60–80% vs baseline 25–45%. Cloud top
extends slightly higher.

**Key coupling**: LWP collapses at k≥2 (48%→14%) because the narrower PDF
raises CF in liquid layers, accelerating autoconversion/accretion. IWP
partially recovers at k=4 (47%) because higher CF traps more ice aloft.
SWP decreases monotonically with k. This makes `cf_steepness_scale` a
strong lever on LWP and the vertical CF profile, but it couples low-level
liquid and upper-level ice budgets in opposite directions — not a clean
single-target knob.
