# Multi-column simulations in ClimaAtmos: checklist

Goal: run N independent columns in one simulation on a
`ClimaCore.Spaces.MultiColumnFiniteDifferenceSpace`, with results that match the
single-column (`config: column`) run up to floating-point rounding, following the shape
of ClimaLand's `ColumnEnsemble` (ClimaLand PR #1826): a grid constructor wrapping
`CommonGrids.MultiColumnGrid`, a handful of dispatch points widened to the new space, and
a single-vs-ensemble comparison script.

Companions: `DIFF_SPACE.md` (site-by-site list of where the two spaces differ) and
`THINGS_TO_ADDRES.md` (adversarial review of the commit; its items are tracked in §2b). This file records what was verified by running, what was changed, and what is
still needed, in Atmos and upstream. Environment: `julia +1.12 --project=.buildkite`
(the untracked Julia 1.12 manifest develops ClimaCore `kp/col-intp` and ClimaDiagnostics
`kp/multi-cols`; repo norms prefer 1.11, whose manifest pins released versions).

## 0. Decisions for the user

Each item says what exists, where to look, the options, and a recommendation. None of
them blocks the verification. Things that are broken on `main` independently of this work
are listed in §8, not here, including the two COSP test errors under Julia 1.12. The user owns the CI test selection, the manifest and
`.gitignore` clean-up before the PR, and what gets committed; the uninitialized cache
scratch (A9) only affects our bitwise checks, so those are not listed.

1. **ClimaUtilities change for single-site forcing on multi-column grids (U1): open a PR?**
   Where: worktree `~/worktree/ClimaUtilities.jl/forcing-intp` (branch `forcing-intp` on
   top of `Release v0.1.32`), uncommitted, formatted; `git diff` there shows all of it. It
   is dev'd into `.buildkite/Manifest.toml`.
   What: `InterpolationsRegridder` gets an opt-in keyword `horizontally_uniform = false`
   (a new `Bool` field). In `Regridders.regrid`, data with a single dimension on a
   `LatLongZPoint` target is an error unless the keyword is set (an informative message
   replaces the old `MethodError`); with it, the coordinates are reduced to `ZPoint(coord.z)`
   before evaluating the interpolant, so one site's `(z, time)` profile is applied to every
   column. Plus a docstring paragraph, a test in `test/regridders.jl` (error without the
   flag, equality with the 3-D result with it), and a NEWS entry. Not run through the
   ClimaUtilities test suite.
   Atmos side: the one call site that knows the data is a single site's profiles,
   `column_timevaryinginputs` in `src/column_datasets/ColumnDatasets.jl`, passes
   `horizontally_uniform = !(target_space isa Spaces.FiniteDifferenceSpace)` in
   `regridder_kwargs`. The branch's constructor (commit `1570399`, "Add regridding vertical
   profiles onto 3D spaces") rejects the flag on a Z-only space, so a single column must
   not set it; the flag is derived from the target space rather than hard-coded to `true`
   (2026-09-10, found by the GPU run of the ERA5 and diurnal cases, both failed at
   construction with "horizontally_uniform requires a lat-long-z or x-y-z space").
   Why opt-in (user's concern, agreed): implicit broadcasting would let a single-site
   profile be spread silently over a real three-dimensional space; ClimaUtilities' own
   multi-column precedent (the 0.1.32 matrix `TimeVaryingInput`) is opt-in by data shape.
   For implicit: a `(z, time)` dataset has only one possible meaning, and the regridder
   already has two implicit special cases (2-D data on a LatLongZ surface space, 3-D data on
   a Z-only target). ClimaLand's `ColumnEnsemble` forces each column at its own location
   only because its ERA5 forcing is gridded in lon/lat; Atmos's files have no location
   axis, so per-column forcing needs `(z, time, column)` data that does not exist (U2).
   The broadcast is the expected behaviour for ensembles of columns on one site's forcing.
   Evidence (opt-in form, Float64, 10 min, `multi_column_dev/forced_case.jl`): ARM VARANAL
   exact (18 state, 141 cache entries within 1e-9); ERA5-driven and the diurnal SCM case
   reproduce their real-grid results (state within 4e-9, and the EDMF signature,
   respectively), both proven metric rounding on the identity grid (§7); without the flag
   the regridder raises the informative error (`queue8d.jl`, "FLAGLESS CHECK").
   Status: the ClimaUtilities PR is open, [#251](https://github.com/CliMA/ClimaUtilities.jl/pull/251)
   "Add regridding vertical profiles onto 3D spaces" (2026-09-10, the user).
   **Superseded 2026-09-15 (§11):** Atmos no longer uses the `InterpolationsRegridder`
   for column forcing at all; `column_timevaryinginputs` builds the ragged in-memory
   `TimeVaryingInput` from `DataSource`s (ClimaUtilities `kp/ragged-data-intp`), which
   handles one file for every column and one file per column alike, on a single column
   and on a multi-column space. #251 is not needed by Atmos any more; the Atmos branch
   now depends on the ragged branch instead (see §10).
   *Status 2026-09-16:* superseded; the ragged `TimeVaryingInput` of ClimaUtilities
   `kp/ragged-data-intp` reads every column input (§12), `horizontally_uniform` is not used.

2. **Requirement (user, 2026-09-10): a multi-column run must be able to use different
   sites, each with its own forcing.** Identical columns everywhere are rarely useful; the
   common case is the ClimaLand `ColumnEnsemble` one, each column forced at its own
   location, and the same-forcing-for-every-column run (item 1) is the special case.
   Nothing in the multi-column grid or space stands in the way (every per-column quantity
   is already a field on the multi-column space); the work is in `ColumnDatasets` and the
   forcing inputs. **Status (2026-09-15): implemented in Atmos on top of the user's
   ClimaUtilities branch `kp/ragged-data-intp` (option (B) below, ragged CSR storage
   in memory with a per-column time search, plus the vertical regrid at load time);
   see §11 for the design, the verification, and what is still open.** The analysis
   below is kept as the record of the alternatives.
   *Status 2026-09-16:* implemented and verified bitwise on CPU and GPU, §12 and
   `MULTI_COL_FORCING.md` Part 4.

   **What exists today (2026-09-11).**
   - Atmos: one `ColumnDataset` is one file for one site (`site_latitude` /
     `site_longitude` scalars; `era5_observations_to_forcing_file.jl` writes
     `tv_forcing_<lat>_<lon>_<dates>.nc` per site). Its time axis is the file's `time`
     (`dates(d, ds)`, `ColumnDatasets.jl:152`), turned into seconds from `start_date`.
     Profiles become file-backed `TimeVaryingInput`s on the target space
     (`column_timevaryinginputs`, streamed: the `DataHandler` keeps the two bracketing time
     slices); surface series are read whole into memory as 0-D
     `TimeVaryingInput(times, vals)` (`surface_timevaryinginputs`, `ColumnDatasets.jl:558`).
     Every format uses `LinearInterpolation()` in time; the monthly diurnal format uses
     `LinearInterpolation(PeriodicCalendar())` (`ColumnDatasets.jl:212, 220`). Vertical
     regridding is `InterpolationsRegridder` in `z` at every evaluation.
   - ClimaUtilities: [#248](https://github.com/CliMA/ClimaUtilities.jl/pull/248) (merged,
     0.1.32) `TimeVaryingInput(times, vals::Matrix, space)`: one time series per point,
     matrix `(points, times)`, **one shared `times` vector** (`_check_dims`), spaces with a
     single vertical level only (`TimeVaryingInputs0DExt.jl:101`); "minimal checks ... it
     is the responsibility of the user to check that the data is correct". Nothing takes a
     per-point time axis and nothing takes `(z, points, times)`.
     [#241](https://github.com/CliMA/ClimaUtilities.jl/pull/241) (the user, open since
     2026-07-22, approved but not merged) is the per-column attempt: `MultiColumnNCFileReader`
     (one NetCDF file per column, per-column dates, columns matched to the target space by
     (lon, lat)), `ColumnRegridder` (vertical-only via `ClimaInterpolations.interpolate1d!`,
     which accepts a *different source grid per column*, `xsource::(n_source_levels,
     n_columns)`), `MultiColumnDataHandler` (LRU cache of regridded slices),
     `DataSource`. **The user's verdict: too slow.** The PR itself anticipated it: "I
     suspected that reading from N datasets can be slow. It might be worth looking into a
     ring buffer instead of a LRU cache". The cost is N NetCDF reads per variable per time
     crossing plus N-column bookkeeping in the loop; it is I/O and per-column dispatch, not
     the arithmetic. [#251](https://github.com/CliMA/ClimaUtilities.jl/pull/251) (open) is
     item 1.
   - ClimaLand: the `ColumnEnsemble` experiment on `main` uses one gridded ERA5 file (one
     time axis by construction), but ClimaLand does meet sites with different time axes and
     handles it by **interpolating onto a common axis before the simulation** (user; not
     visible on `main`). That is the CliMA precedent for this problem.
   - What Atmos uses from ClimaUtilities today, option by option: §0b "What the forcing
     code uses from ClimaUtilities today".
   - ClimaInterpolations 0.1.x (`~/.julia/packages/ClimaInterpolations`): `interpolate1d!`
     is a batched 1-D linear interpolation where "each column can have a different grid",
     all columns having the same number of source points; `Flat` extrapolation; no ragged
     (different-length) grids, no time-cursor logic.

   **Data examples (2026-09-14; read from the files on `/net/sampo/data1/ClimaArtifacts`
   and from the readers).** Every source the column code meets today, with the
   properties the representation choice depends on. "Julia order" is the `NCDatasets`
   index order, the reverse of `ncdump`.

   | Source | Container | Per-site layout | Vertical axis | Time axis | Shared across sites? |
   |---|---|---|---|---|---|
   | GCM cfsite forcing, `cfsite_gcm_forcing/HadGEM2-A_amip.2004-2008.07.nc` (101 MB) | one NetCDF-4 file, **51 groups** `site2`…`site30`, `site58`, … (the README says 21); no root dims or vars | per group: dims `lev = 38`, `time = 600`; profiles `(lev, time)` (`zg`, `ta`, `hus`, `ua`, `va`, `wap`, `alpha`, `tnt*`, `tnhus*`, `pfull`, `th`, `thli`, …); surface series `(time,)` (`ts`, `ps`, `rsdt`, `hfls`, `hfss`, `pr`, …); **scalars** `lat`, `lon`, `site`, `label`, `coszen`, `gustiness`, `windstd`, `windrms` | `lev` hybrid height, **ascending** 20 m … 39.25 km; `zg(lev, time)` equals `lev` to `float32` precision, constant in time and identical in every group | `int64` hours since 2004-07-01, **`360_day` calendar**, 6-hourly, 600 samples (five Julys); identical in every group | yes, both axes; not ragged |
   | PyCLES LES statistics, `pycles_scm_les_data/{Bomex,DYCOMS_RF01,DYCOMS_RF02,GABLS,Rico,Soares,TRMM_LBA}.nc`; the cfsite LES `Stats.cfsite<N>_HadGEM2-A_amip_2004-2008.07.nc` under `/resnick/groups/esm/zhaoyi/GCMForcedLES/` (not mounted on `clima`) | one NetCDF-4 file per case or site, groups `profiles`, `timeseries` (+ `reference` in TRMM and the cfsite files); **two layouts**: root dims `z`, `t` with `t(t)`, `z_half(z)` at the root (Bomex style), or dims and `z`, `z_half`, `t` (UNLIMITED) inside `profiles` (TRMM style) | profiles `(z, t)`, `rho(z)`, `p0(z)`; timeseries `(t,)`; `z_half` centers, `z` faces, most variables at centers despite the `z` dim (`helper_funcs.jl:73`) | per file: 75–300 levels, Δz 3.1–40 m, top 0.4–6 km | per file: `t` in seconds from 0, **no `units`**, Δt 60–600 s, 49–865 samples, 4 h–1 day (cfsite LES: 6 days, calibration averages days 5.5–6) | **no**: ragged in `z` and `t`; observation side only, reduced to a time-window mean interpolated onto the SCM levels below `z_max` (`observation_map.jl`) |
   | ERA5-derived site file, generated at run time: `tv_forcing_<lat>_<lon>_<dates>.nc`, `monthly_diurnal_cycle_forcing_<lat>_<lon>_<date>.nc` | flat NetCDF-4 in the ClimaColumn schema, global attrs `site_latitude`, `site_longitude`; one file per site | profiles `(z, time)`, surface `(time,)`, CMIP names with SI `units` | `z` = time mean of the geopotential height of the 37 pressure levels at the site, sorted ascending (`era5_observations_to_forcing_file.jl:518`); 37 levels everywhere, **values differ per site** | hourly, `hours since <start_date>`, standard calendar, covers `start_date`…`t_end`; diurnal file: exactly 24 h, wrapped by `PeriodicCalendar` | a batch built from the same raw days shares the time axis by construction; `z` counts shared, values not |
   | Raw ERA5 behind it, `era5_hourly_atmos_raw/forcing_and_cloud_hourly_profiles_YYYYMMDD.nc` (+ `hourly_inst_`, `hourly_accum_`) | gridded NetCDF-4, one file per day | `(longitude, latitude, pressure_level, valid_time)`; 1440 × 721 × 37 × 24 | `pressure_level` hPa, decreasing | `valid_time` seconds since 1970, `proleptic_gregorian`, 24 hourly samples | one grid for every site, so a `(z, column, time)` extraction is one indexed read (the ClimaLand `ColumnEnsemble` shape) |
   | ARM VARANAL, `arm_sgp_varanal_forcing/sgp60varanarucC1.c1.20100901.000000.cdf` | **NetCDF classic (`.cdf`)**, flat, one site (SGP) | profiles `(lev, time)`: `T`, `q`, `u`, `v`, `omega`, `T_adv_h`, `q_adv_h`; series `(time,)`: `LH`, `SH`, `T_skin`; `-9999` fills, non-CMIP names, mixed units | `lev` 37 pressure levels 1000 … 100 hPa, descending; height derived hydrostatically by `to_climacolumn` | `base_time` (epoch s) + `time_offset`; 720 samples from 2010-09-01, **Δt = 3600.44 s** (hourly, not exact; §0b's "3-hourly" is wrong for this file) | n/a (one site); converted once to a ClimaColumn file |
   | Steady in-memory (`InMemoryColumnData`) | none | `z::Vector` + `(z,)` vectors + surface scalars | source levels, ascending | none | n/a |

   What this says about the three axes below. *Time*: within one source the axis is
   shared (cfsite, an ERA5 batch, one LES file); different axes only arise when sources
   are mixed (a `360_day` 2004 calendar next to hourly 2007 ERA5 next to 3600.44-s ARM
   steps), and the LES files are the only genuinely ragged time data, on the observation
   side. *Vertical*: level counts are shared within a source (38, 37, 37) but the
   heights differ per site for ERA5 (time-mean geopotential) and per case for LES
   (75–300 levels); every reader already interpolates onto the model levels. *Container*:
   groups (cfsite, PyCLES), flat files (ClimaColumn), classic `.cdf` (VARANAL), gridded
   4-D (raw ERA5); the multi-site case that exists today, cfsite, keeps N sites in one
   file with identical axes, which is the `(z, site, time)` matrix of option (A)
   already. Mock-file drift to fix when the tests grow a multi-site case:
   `write_test_cfsite_file` (`test/column_datasets_tests.jl`) writes `zg` descending and
   says the real files are top-down (they are ascending), gives `coszen` a `(time,)`
   dimension (a scalar in the real file), and names the level dim `z` (`lev`); the reader
   is indifferent to all three.

   **The problem, decomposed.** "Different sites" is three independent axes of
   heterogeneity, and a design has to say what it does on each:
   1. *I/O layout*: N files (one per site) vs one file with a `column` dimension. #241
      chose N files and paid for it at every step.
   2. *Time axes*: shared (the ERA5 batch, the diurnal cycle, GCM steady) vs different
      sampling / ranges / lengths / epochs (ARM 10-minute or hourly VARANAL next to ERA5
      hourly, campaign data, files generated for different dates). "Ragged" = different
      lengths per column, which no dense `(column, time)` array holds without padding.
   3. *Vertical axes*: shared model levels vs per-site source levels, and for reanalysis
      model levels the source heights even change with time. #241's `ColumnRegridder`
      handles per-column source levels of equal count; ragged `z` is also possible (ARM
      pressure levels vs ERA5 model levels).
   Two facts make the problem smaller than it looks. (i) Atmos only ever interpolates
   linearly in time, and linear interpolation is *idempotent on a superset of nodes*: a
   piecewise-linear series resampled onto any axis that contains its own nodes is
   reproduced exactly at those nodes and, between them, is the same line. So resampling a
   site's series onto the union of all sites' time points (restricted to the run window)
   loses nothing; each column still sees exactly its own data at every model time. The
   periodic-calendar case needs the sites to share the period, which the monthly diurnal
   files do by construction. (ii) The same holds in `z`: interpolating each site's profiles
   onto the model's own levels once, before the run, is what the regridder would do at
   every evaluation anyway (linear in `z`), so it can be moved out of the loop entirely.

   **The user's two thoughts, evaluated.**
   - *(A) Interpolate everything beforehand* (time onto a common axis, `z` onto the model
     levels), then the run only does `dest = w₁·slice(t₁) + w₂·slice(t₂)` with one shared
     time cursor: no ragged representation, no per-column dispatch, no I/O in the loop, one
     `evaluate!` per variable per step, trivially GPU-friendly, exact for linear
     interpolation. Cost: memory, `|axis| × nlev_model × ncol × nvars × sizeof(FT)`, and a
     preprocessing step. Numbers (Float32, ~12 profile variables, 60 model levels): one
     year hourly (8760 times) is 25 MB per site, so 100 sites = 2.5 GB, 1000 sites = 25 GB;
     a 10-minute site over a year is 150 MB, and an axis that is the *union* of 100 hourly
     sites and one 10-minute site is dominated by the fine site, 100 × 150 MB = 15 GB.
     Typical SCM use (hours to days, tens of sites) is kilobytes to megabytes. So (A) is the
     right default and fails only in the "many sites × long run × fine sampling" corner.
   - *(B) Efficient ragged interpolation in ClimaInterpolations*: store the sites' axes as
     they are, CSR-style (`times` and `values` concatenated over columns with an `offsets`
     vector), keep one integer cursor per column, and let a kernel over columns advance its
     cursor (times are monotone, so O(1) amortized, no binary search) and form the two-point
     combination; the same kernel handles per-column `z` if the values are stored on model
     levels after a once-off vertical interpolation. Memory is `Σᵢ |axisᵢ|`, the minimum
     possible, which beats (A) by the ratio `ncol × |union| / Σᵢ|axisᵢ|` (≈ 60× in the "one
     10-minute site among hourly ones" example, 1× when the axes are shared). It is a
     moderate ClimaInterpolations feature (a new `interpolate_ragged!`-like routine plus the
     cursor state; `interpolate1d!` already has the per-column-grid structure, minus the
     ragged offsets and minus statefulness) and a small ClimaUtilities `TimeVaryingInput`
     wrapper around it. What it does **not** solve is memory when even `Σᵢ|axisᵢ|` does not
     fit: then chunks must be streamed, and with ragged axes every column's window advances
     at its own rate, which is the genuinely hard "worst worst case". Streaming a *shared*
     axis is easy (one window for all columns, a ring buffer, one read per window).
   - Where they meet: (B) only pays off when the sites' sampling rates differ a lot *and*
     the union is too big for memory *and* the sum of the ragged axes is not. That corner
     is narrow. If the data are that heterogeneous, resampling the fine sites to a coarser
     common axis offline is a scientific choice that also fixes the memory problem, and it
     is a decision a person should make, not the run.

   **Representations, with costs (per variable per step unless noted).**
   | | I/O in loop | Memory | Ragged time | Ragged z | Kernel launches | New code |
   |---|---|---|---|---|---|---|
   | (a) N per-column file-backed inputs (#241 shape) | N reads per time crossing | O(2·nlev·N) | yes | yes | N | none in Atmos beyond wiring; #241 exists |
   | (A) pre-interpolate to common axis, in memory | none | |union|·nlev·N | resolved offline | resolved offline | 1 | small: `(z, col, time)` in-memory input (#248 generalized to levels), preprocessing in `ColumnDatasets` |
   | (A′) pre-interpolate to one `(z, col, time)` file, streamed | 1 read per window | O(window·nlev·N) | resolved offline | resolved offline | 1 | `DataHandler` with a `column` axis (read = index match, no horizontal regridding) |
   | (B) ragged CSR + per-column cursors, in memory | none | Σ|axisᵢ|·nlev | yes | after once-off z interpolation | 1 | ClimaInterpolations ragged kernel + ClimaUtilities wrapper |
   | (B′) ragged and streamed | per-column windows | bounded | yes | yes | 1 | hard; avoid by resampling offline |
   Different time *ranges* are a policy, not a representation: Atmos already requires a
   file to cover the run (`file_time_span`); keep that per site so no column is ever
   extrapolated silently, and take the union axis inside the run window only.

   **Recommendation (conditional on the needs; see status).** Fix the data model now,
   choose the representation later: one dataset per site, sites listed in the same order
   as `column_latitudes` / `column_longitudes`, a preprocessing step in `ColumnDatasets`
   that brings every site to the model `z` levels and to a common time axis (union within
   the run window; lossless because interpolation is linear), and the run doing only time
   interpolation. Then:
   - default: (A) in memory through a levels-aware version of #248's matrix input (the
     natural next ClimaUtilities PR after #248/#251; the loader can compute the memory
     estimate above and refuse or warn);
   - if the estimate is too large: (A′), write the preprocessed `(z, column, time)` array
     to one NetCDF file and stream windows of it (one read per variable per window,
     N-independent, the ring buffer #241 wanted), which is also the natural home for
     ERA5 batches generated by `era5_observations_to_forcing_file.jl`;
   - (B) only if a real dataset shows the ratio `ncol·|union| / Σ|axisᵢ|` matters in
     practice; it is the only option that keeps ragged axes without resampling, and it
     should live in ClimaInterpolations next to `interpolate1d!`;
   - (a)/#241 is the no-new-feature fallback and is known to be too slow at scale.
   In all cases: per-column surface temperature, fluxes and insolation (A4:
   `TimeVaryingInsolation` with an explicit site is broadcast today; per site it becomes a
   row of the same matrix input, or the columns' own coordinates are used), and
   `read_initial_profiles` filling each column from its own file (already pointwise).
   Questions that decide the choice: how many columns; which sources are mixed; the finest
   sampling and the run length; whether sites must share a run window; whether surface
   forcing and insolation come per site or from the columns' coordinates.

## 0b. Single-column forcing inventory: what depends on the site, and on time

Background for §0 items 1 and 2. "Site" means the forcing is tied to one (lat, long);
"time axis" says where the times come from, which matters if several columns are ever to
carry different sites (ClimaLand's problem: one time series per column works through
`TimeVaryingInput` matrix input only when all columns share the time axis).

### Analytic setups (`src/setups/`; no files, no site, no time axis)

| Setup (`initial_condition`) | Forcing used | Site-dependent? |
|---|---|---|
| Bomex, Rico | large-scale subsidence `w(z)`, large-scale advection `dT/dt(z)`, `dq/dt(z)`, geostrophic wind + f-plane `coriolis_param` (all `AtmosphericProfilesLibrary` z-profiles), prescribed surface fluxes (Bomex) or bulk `MoninObukhov(z0)` fluxes from a fixed `T_sfc` (Rico) | no |
| GABLS | geostrophic wind + f-plane Coriolis, prescribed surface temperature `T_sfc(t)` (analytic in time) | no |
| DYCOMS RF01/RF02 | subsidence `-D z`, geostrophic wind, prescribed `shf`/`lhf`, `RadiationDYCOMS` (analytic radiative flux from the column's own `ρq` integral) | no |
| ISDAC | `ISDACForcing` (analytic), `RadiationISDAC`, prescribed surface | no |
| Soares, SimplePlume, TRMM_LBA (prescribed `dT/dt_rad(t)` profile), Larcform1 (geostrophic wind, slab ocean, `Larcform1Insolation` polar night, all-sky RRTMGP), advection test, ShipwayHill2012 (prescribed vertical velocity `w(t)`) | analytic, time only | no |
| Radiative-equilibrium columns (gray, clear-sky, all-sky, slab ocean) | RRTMGP with `IdealizedInsolation` | **yes through the grid**: insolation and the default zonally-symmetric surface temperature read the coordinate latitude; a Cartesian single column is treated as the equator (`callbacks.jl:203`), a multi-column column uses its own lat/long (DIFF_SPACE P2, P3) |

These are unaffected by the multi-column question: every column applies the same
analytic forcing, and only the grid-coordinate physics (Coriolis, insolation, default SST)
differ between columns, which is intended.

### File-forced setups (all through `Setups.ForcingFromFile` and `ColumnDatasets`)

One file, one site, one time axis. The forcing terms (`src/prognostic_equations/forcing/
forcing_terms.jl`) read canonical variables: `HorizontalAdvection` (`tntha`, `tnhusha`),
`VerticalFluctuation` (`tntva`, `tnhusva`), `Subsidence` (`wa`), `Nudging` (`ta`, `hus`,
`ua`, `va`); surface: `ExternalTemperature` (`ts`), `FileHeatFluxes` (`hfls`, `hfss`);
insolation: `ExternalTVInsolation` (`coszen`, `rsdt`) or `TimeVaryingInsolation(latitude,
longitude)` computed from the site coordinates.

| Case | File and site | Terms | Time axis |
|---|---|---|---|
| GCM-driven (`initial_condition: GCM`, `prognostic_edmfx_gcmdriven_column.yml`) | HadGEM2 cfsite file, group `cfsite_number`; the site's lat/long are in the file | default terms (horizontal advection, vertical fluctuation, nudging of `ta`,`hus`,`ua`,`va`, subsidence); interactive Monin-Obukhov with the file's `ts`; file `coszen`/`rsdt` | **none**: `GCMColumnData.read_cfsite` time-averages every variable over the file period (`tmean`), so the forcing is steady (`TimeVaryingInput(Returns(field))`). This is why GCM-driven never needed the regridder fix |
| ARM VARANAL (`initial_condition: ARMVARANAL`, `prognostic_edmfx_armvaranal_column.yml`) | one ARM SGP file, converted to the ClimaColumn schema by `VaranalFiles.to_climacolumn` (site lat/long copied from the file's `lat`/`lon`) | horizontal advection, nudging (`ta`,`hus`; `ua`,`va`), subsidence from `wa`; `MoninObukhov(z0, ustar, FileHeatFluxes)`; `TimeVaryingInsolation` at the **site** lat/long | the file's `time` (3-hourly), shared by all variables; surface series read into 0-D `TimeVaryingInput(times, values)` |
| ERA5-driven (`initial_condition: ReanalysisTimeVarying`, `prognostic_edmfx_tv_era5driven_column.yml`) | a per-site file **generated at run time** by `era5_observations_to_forcing_file.jl` from the gridded `era5_hourly_atmos_raw` artifact at the quarter-degree point nearest `site_latitude`/`site_longitude` (config keys); file name encodes site and dates | default terms; file `ts`, `coszen`, `rsdt` | hourly, from the raw ERA5 files, shared by all variables of the site file |
| Diurnal SCM (`external_forcing: ReanalysisMonthlyAveragedDiurnal` + `initial_condition: ReanalysisTimeVarying`, `prognostic_edmfx_diurnal_scm_imp.yml`) | same generator, monthly-mean diurnal cycle at the site | default terms with `periodic_calendar_method()` (one stored day repeats) | one day, repeated periodically |
| Generic `initial_condition: ForcingFromFile` | any ClimaColumn-schema file (`site_latitude`/`site_longitude` global attributes) | default terms | the file's |

How a term reaches the columns: `column_timevaryinginputs` (`ColumnDatasets.jl:495`)
builds `TimeVaryingInput(path, var, target_space)` per profile variable, which regrids the
file's `(z, time)` data onto the model space at each time (`InterpolationsRegridder`);
surface variables become 0-D `TimeVaryingInput(times, values)` and are broadcast to every
column. On a multi-column space the profile path is where the regridder needs the
`forcing-intp` prototype (§0 item 2): one site's `z`-profile is applied to every column.

### What reads latitude and longitude in a file-forced single-column run

Only two quantities in ClimaAtmos read a latitude at all: the Coriolis term and the
insolation (plus the default zonally symmetric SST, which the file-forced setups do not
use because the surface temperature comes from the file). For ARM VARANAL and ERA5 on a
**single column** (`config: column`, Cartesian `ZPoint` coordinates, no lat/long):

| Quantity | Single column | Multi-column, column at (0, 0) | Multi-column, column placed at the true site |
|---|---|---|---|
| Insolation | ARM VARANAL: `TimeVaryingInsolation(latitude, longitude)` built from the site coordinates read from the file, applied through the explicit override in `callbacks.jl:246` (the grid is not consulted). ERA5: `coszen`/`rsdt` read from the site file, which the generator computed at the site. So the **site is used** | same (override/file), identical to the single column | same: the override wins over the column's own coordinates (A4) |
| Coriolis (`compute_coriolis`, `cache.jl:317`) | `ZPoint` branch: `ᶜf³ = f_plane_coriolis_frequency`, a ClimaParams parameter whose default is **0**, and none of the forced configs' toml files set it. So the single-column ARM VARANAL and ERA5 runs have **no Coriolis** | `2Ω sin(0) = 0`, identical | `2Ω sin(lat_site) ≠ 0`: differs from the single column |
| Everything else (forcing terms, surface fluxes, `ts`) | from the file, no coordinates | same | same |

The "Cartesian column is treated as the equator" rule is `IdealizedInsolation`
(`callbacks.jl:203`, used by the radiative-equilibrium columns), not the file-forced
cases. Net effect: a multi-column column at (0, 0) reproduces the single-column forced run
exactly (§7); a column placed at the site's real coordinates would add the site's Coriolis
term that the single-column run silently lacks, and would still get the site's insolation
through the override.

### What the forcing code uses from ClimaUtilities today (read from the code, 2026-09-11)

All file-forced column setups go through `ColumnDatasets` (`src/column_datasets/`), with
three data sources: `ClimaColumnFile` (the ERA5-driven and diurnal files, and ARM VARANAL
after `VaranalFiles.to_climacolumn` converts it), `GCMColumnData.read_cfsite` (steady
profiles, held as `InMemoryColumnData`), and nothing else. What each piece uses:

| Data | Object built | Space regridding | Time interpolation | Out-of-range |
|---|---|---|---|---|
| Profile forcing (`ta`, `hus`, `ua`, `va`, `wa`, tendencies) from a file | file-backed `TimeVaryingInput(path, var, target_space; start_date, regridder_kwargs, method)` (`column_timevaryinginputs`, `ColumnDatasets.jl:495`) | default `:InterpolationsRegridder` (Atmos passes no `regridder_type`), linear (`interpolation_method` default `Intp.Linear()`), `extrapolation_bc = (Intp.Flat(),)` in `z` (`extrapolation_bc(::AbstractColumnFormat)`, line 199; no format overrides it), `dim_names`/`dim_increasing` derived by the `DataHandler` from the file (`DataHandlingExt.jl:307-335`, not set by Atmos), `horizontally_uniform` derived from the space type (this branch) | `LinearInterpolation()` for every format (line 212) = linear with `Throw()` (ClimaUtilities default, `TimeVaryingInputs.jl:230`); the diurnal case passes `LinearInterpolation(PeriodicCalendar())` (`model_getters.jl:873`), `PeriodicCalendar()` = `(nothing, nothing)`, i.e. the file's whole time range is the period | error (`Throw`), except periodic wrap for the diurnal case; `wraps_periodically` (line 433) skips the run-coverage check only in that case |
| Surface series (`ts`, `hfls`, `hfss`, `coszen`, `rsdt`, ...) | in-memory 0-D `TimeVaryingInput(times, vals; method)` (`surface_timevaryinginputs`, line 558), whole series read at construction | none (a scalar per time, broadcast to the whole surface, i.e. to every column) | same `method` as the profiles | same |
| Steady GCM cfsite profiles (`InMemoryColumnData`) | `TimeVaryingInput(Returns(field))` (line 672) after `_interp_column` | Interpolations.jl `Gridded(Linear())` onto the model `z`, `Flat()` beyond the source range (lines 647-653), done once | none (constant) | n/a |
| Steady GCM surface values | `TimeVaryingInput(Returns(FT(value)))` (line 686) | none | none | n/a |
| Initial profiles | `read_initial_profiles` (line 449): the file time **closest** to `start_date` (`time_index_closest`, no time interpolation), sorted ascending in `z`, returned as arrays for the setup to place on the grid | (in the setup, not traced here) | none | n/a |
| Insolation with `ExternalTVInsolation` | the `coszen`/`rsdt` 0-D inputs above, `evaluate!`d in `set_insolation_variables!` (`callbacks.jl:185`) and written to every RRTMGP column | none | linear | `Throw` |
| Surface temperature with `ExternalTemperature` | the `ts` 0-D input above | none | linear | `Throw` |

`preprocess(format, name)` exists as a hook but is `identity` for every format (line 143),
so no `preprocess_func` is passed. No format overrides `extrapolation_bc`, `dates`, or
`time_interpolation_method`.

Not used anywhere in the column forcing (available in ClimaUtilities): `NearestNeighbor`,
`Flat()` as a *time* boundary, `PeriodicCalendar(period, repeat_date)` with an explicit
period, `LinearPeriodFillingInterpolation`, `:TempestRegridder`, `compose_function`,
explicit `dim_names`/`dim_increasing`, `interpolation_method = Intp.Constant()`, and the
0.1.32 matrix `TimeVaryingInput`. Elsewhere in Atmos, for context only: ozone and
aerosols use file-backed `LinearInterpolation()` + `InterpolationsRegridder`
(`tracer_cache.jl:20, 120`); the ERA5 cloud fields for radiation use
`LinearInterpolation(PeriodicCalendar(Year(1), Date(2010)))` (`radiation.jl:493`); CO2 is
an in-memory 0-D series; topography, OGW tables and the `WeatherModel` /
`overwrite_from_file` initial conditions use `SpaceVaryingInput` (the latter with a
user-selectable `interpolation_method`).

What this means for §0 item 2: every column forcing is *linear in time*, so resampling a
site's series onto a finer common axis is exact at its own nodes; out-of-range is an
error today, so a "every site must cover the run window" policy is the status quo, not a
new restriction; the only periodic case (`PeriodicCalendar()` over the file's range) needs
sites to share the period, which the monthly diurnal files do by construction; and all
surface forcing, insolation included, is already a per-time scalar broadcast to every
column, which is exactly the row-per-column shape of the 0.1.32 matrix input.

### Consequences for multi-column runs

- Today every column gets the **same** site's forcing, so a multi-column forced run is a
  clone of the single-column one. This is what §7 verifies (GCM-driven, ARM VARANAL, ERA5
  match exactly). `TimeVaryingInsolation` with an explicit site (ARM VARANAL) is also
  broadcast to every column, overriding the columns' own coordinates (A4); columns at
  other locations would get the site's insolation, not their own.
- Per-column (multi-site) forcing does not exist (U2). What it would need, in order of
  difficulty: (1) surface series per column: ClimaUtilities 0.1.32 `TimeVaryingInput`
  with a matrix (`ntimes × ncolumns`) on the multi-column surface space already does
  this, **provided all sites share one time axis**; (2) profile forcing per column: a
  `(z, time, column)` variable and a regridder that maps column `h` of the data to column
  `h` of the space (nothing exists; the prototype only broadcasts one site); (3) a
  multi-site file format or one file per column in `ColumnDatasets`, and per-column
  `site_latitude`/`site_longitude`; (4) insolation from the columns' own coordinates
  (drop the explicit site override when the grid carries lat/long).
- Time axes: ARM VARANAL is 3-hourly, ERA5 hourly, the diurnal case a repeated day, and
  GCM-driven has none. Mixing sites of the same product (e.g. several ERA5 points) gives
  a common time axis by construction; mixing products does not, and a matrix
  `TimeVaryingInput` cannot represent that. Interpolating every site onto a common axis
  in the file generator (as `era5_observations_to_forcing_file.jl` already does for one
  site) is the natural place to solve it.

## 1. Status at a glance (2026-09-10)

- Every column config in `config/model_configs` builds and runs on a multi-column grid
  through `config: multicolumn` (§7). The three file-forced cases run too; ARM VARANAL and
  ERA5-driven and the diurnal SCM case need the ClimaUtilities prototype in §0 item 1.
- Co-located duplicate columns are bitwise identical in every case (only uninitialized
  `similar` cache scratch differs, A9).
- With columns at (lat, long) = (0, 0), `deep_atmosphere: false`, and Float64, column 1
  matches the single column within rtol 1e-9 in state, cache, and NetCDF output for every
  non-EDMF case, for the EDMF cases GABLS, simple plume, advection test, DYCOMS RF02,
  Bomex tracerA, LARCFORM1, and for GCM-driven and ARM VARANAL forcing. The EDMF cases
  that differ on the real grid (Bomex and its variants, DYCOMS RF01, TRMM, RICO, ERA5)
  are **bitwise identical** on a multi-column grid with an identity horizontal metric
  (§7 "attribution by construction"), so their difference is the sphere metric's rounding
  in the vertical operators and nothing else. Float32 runs agree to 1e-5..1e-4.
- Still failing, on a single column too (pre-existing, §8): the two sparse-autodiff
  column configs and 2M precipitation. Nothing fails only on multi-column.
- 2026-09-10 evening (handoff §0c): the ClimaCore PR branch `multi-col-extras` gives
  `MultiPointGrid` a unit horizontal metric (the user's commit `a8ae99ac`), which is the
  identity-metric grid of §7, so the "real grid" differences above are gone by
  construction and column 1 is expected bitwise identical to the single column. The three
  hardest configs were run on one GPU on the PR branches (§9 "GPU results on the PR
  branches"): LARCFORM1 passed as is; ERA5-driven and the diurnal SCM case needed three
  fixes, all for failures that the *single* column has too: the regridder flag derived
  from the target space (§0 item 1, kept), and two GPU fixes (surface-conditions kernel
  argument in Atmos, type-valued `Ref` eltype in ClimaCore's CUDA extension) that were
  used for the runs and then reverted as pre-existing and out of scope (§8).
- Decisions the user has to make are collected in §0.

## 2. Atmos changes made on this branch (committed in `12a1c8f7c`, `b70b98bf5`)

All small and in the spirit of ClimaLand's PR: widen dispatch, do not special-case.

- [x] `src/simulation/grids.jl`: `MultiColumnGrid(FT; points, radius, context, z_elem,
      z_max, z_stretch, dz_bottom, z_mesh)` wrapping `CommonGrids.MultiColumnGrid`
      (same vertical mesh as `ColumnGrid`); exported. `get_spaces(grid)` is now generic
      via `Spaces.space(grid, Grids.CellCenter()/CellFace())`, which covers extruded,
      single-column, and multi-column grids (it used to error on anything else).
- [x] `src/utils/utilities.jl`: `const ColumnSpace = Union{FiniteDifferenceSpace,
      MultiColumnFiniteDifferenceSpace}`; `iscolumn`, `has_topography`, `do_dss`, and
      `horizontal_filter_scale` dispatch on it. This makes the NOGW column branch, the
      EDMF horizontal-flux skip, the DSS skip, and the `Inf` filter scale apply to
      multi-column spaces.
- [x] `src/parameterized_tendencies/sponge/viscous_sponge.jl`: `uₕ` sponge guard uses
      `iscolumn` instead of an `isa FiniteDifferenceSpace` test.
- [x] `src/prognostic_equations/implicit/autodiff_utils.jl`: `column_index_iterator`
      yields `(1, 1, h)` for a `MultiPointSpace` horizontal space (needed by
      `AutoSparseJacobian`/`AutoDenseJacobian`; verified with the `*_sparse_autodiff`
      configs, see §7).
- [x] `src/parameterized_tendencies/radiation/radiation.jl`: RRTMGP `ncol` is
      `Spaces.ncolumns(axes(Y.c))` instead of `length(Spaces.all_nodes(level space))`
      (`all_nodes` is a spectral-element concept with no `MultiPointSpace` method; this
      blocked every RRTMGP run).
- [x] Config plumbing (setup-level hardcoding, as agreed): `config: "multicolumn"` with
      new keys `column_latitudes` / `column_longitudes` (default `[0.0]`) in
      `config/default_configs/default_config.yml`; `get_grid` builds `MultiColumnGrid`
      with the planet radius from `params` and skips topography keys for both column
      kinds; `check_case_consistency` accepts `"multicolumn"`.
- [x] `examples/multi_column/comparison_utils.jl` and
      `examples/multi_column/single_vs_multi_column.jl`: ClimaLand's comparison utilities
      ported (vector fields compared in physical components, Fields of structs compared
      per component, NetCDF diagnostics compared per column) and a config-driven driver:
      `julia --project=.buildkite examples/multi_column/single_vs_multi_column.jl
      --config_file <yml> --job_id <id>`.
- [x] `src/config/model_getters.jl`: `ReanalysisMonthlyAveragedDiurnal` accepts `config`
      `"column"` or `"multicolumn"` (A8); it only ever needed a column geometry.
- [x] (uncommitted, 2026-09-10 evening) `src/column_datasets/ColumnDatasets.jl`:
      `column_timevaryinginputs` passes
      `horizontally_uniform = !(target_space isa Spaces.FiniteDifferenceSpace)`; the
      ClimaUtilities constructor on `forcing-intp` rejects the flag on a Z-only space.
- Not changed: `Project.toml` compat `ClimaDiagnostics = "0.3.9"`; the `kp/multi-cols`
  branch is versioned 0.3.10, so no bump is needed until upstream goes to 0.4.

## 2b. Review items (`THINGS_TO_ADDRES.md`) and their status

Uncommitted follow-up edits made after the review, without a REPL (unverified):

- [x] 1.2 docs: `ClimaAtmos.MultiColumnGrid` added to `docs/src/api.md`, a row in
      `docs/src/interfaces.md`, and the `AtmosSimulation` docstring.
- [x] 2.2 `MultiColumnGrid` errors unless `context` is a `SingletonCommsContext`
      (mirrors `CommonGrids.ColumnGrid`).
- [x] 2.3 (part) empty `column_latitudes` errors in `get_grid`.
- [x] 2.7 stale comment in `autodiff_utils.jl` fixed.
- [x] 3.1 vector fields are compared again (rank-2 tensors only are skipped), 3.2
      diagnostic names come from the dataset, 3.3 exceptions are recorded with their type,
      non-finite entries are counted, deep nesting warns.
- [x] 4 driver writes each run to its own `output/<job_id>`; test semantics documented.
- [x] 5 checklist corrected (commit status, ClimaDiagnostics version, vector caveat).
- [ ] 1.1 `.buildkite/Manifest.toml` is in the commit with local dev paths (the user is
      fine with paths in the manifest, but CI's `Pkg.instantiate` needs those paths to
      exist); `.gitignore` line `*/Manifest*.toml  # docs, test` never matches because
      gitignore has no trailing comments. Decide before a PR.
- [x] 1.3 pinned formatter (JuliaFormatter 2.10.1) run on every changed Julia and
      Markdown file in Atmos and ClimaCore.
- [x] 2.1 = A1: ClimaCore C1 landed in `kp/col-intp`; no Atmos guard needed.
- [x] 2.3 (rest) = A13: the two list rules (equal length, non-empty) live in
      `check_case_consistency`; no warning for unused keys (no precedent for other configs).
- [x] 2.4 = A2 (deep flag forwarded; A8 still open, see §3). 2.5 = A6 done (`issphere`
      via `Spaces.global_geometry`). 2.6: ClimaCore C2 landed; the Atmos
      `do_dss(::ColumnSpace)` method stays because the single column needs it.
- [x] Re-ran GABLS (match, vectors included), `t0_diag.jl`, and `bomex_diag.jl` (B1
      resolved: EDMF rounding sensitivity).

## 3. Atmos items still open (decisions or larger changes)

- [x] A1 (resolved by ClimaCore C1) Hyperdiffusion on columns (`hyperdiffusion.jl:22`, `ν₄`): calls
      `Spaces.node_horizontal_length_scale(horizontal_space)`, which has no
      `MultiPointSpace` method. On a single column it silently returns the `PointSpace`
      placeholder 1 and the operators are zero, so hyperdiffusion is a no-op; most column
      configs leave `hyperdiff` on. Options: (a) ClimaCore adds the `MultiPointSpace`
      method (C1, preferred: parity with `PointSpace`, zero Atmos change); (b) Atmos skips
      hyperdiffusion for `ColumnSpace` (guards in `hyperdiffusion_cache` and the four
      tendency entry points, or return `nothing` from `get_hyperdiffusion_model` for
      column configs). Same pattern in the opt-in LES closures (`smagorinsky_lilly.jl:97`,
      `anisotropic_minimum_dissipation.jl:70,211`).
- [x] A2 (resolved: `MultiColumnGrid(...; deep_atmosphere)` mirrors `SphereGrid`, `get_grid` forwards the config flag; a single-vs-multi comparison must set `deep_atmosphere: false`) `deep_atmosphere` vs the always-shallow multi-column grid. The grid ignores the
      flag (ClimaCore never forwards `deep`), Coriolis follows the geometry (shallow),
      but RRTMGP scales fluxes by `((z+R)/R)^-2` whenever the radiation mode has
      `deep_atmosphere = true` (config default) and the space is spherical. Verified: 2%
      differences in `rsdt`/`rlut`/`ρe_tot` vs the single column with the default, exact
      agreement with `deep_atmosphere: false`. Decide: assert/warn in
      `check_case_consistency` for `multicolumn && deep_atmosphere`, or pass
      `planet_radius` to RRTMGP only for `DeepSphericalGlobalGeometry`, or add a `deep`
      path once ClimaCore forwards it (C4).
- [ ] A3 Latitude/longitude-dependent physics differs from a single column unless the
      columns sit at (0, 0): Coriolis `2Ω sin(lat)` (single column uses the f-plane
      parameter, default 0); insolation and gray optical thickness (single column assumes
      the equator); default zonally symmetric surface temperature and slab-ocean initial
      temperature; `DecayingProfile` longitude perturbation. Verified at lat 30: `T_sfc`
      differs by 14 K and everything downstream. This is intended physics, not a bug;
      document it in the multi-column docs and in the comparison driver (done in the
      driver header).
- [ ] A4 `TimeVaryingInsolation` with an explicit `latitude`/`longitude` (set for
      `ForcingFromFile`/ARM VARANAL) broadcasts one site to every column, overriding the
      grid coordinates (`callbacks.jl:246-252`). Fine for identical columns; wrong for
      columns at different sites.
- [ ] A5 `horizontal_integral_at_boundary` (`utilities.jl`) asserts extruded/spectral
      spaces; reached by `flux_accumulation!` when `check_conservation: true` (also fails
      on a single column today). Keep `check_conservation` off for columns or add a
      `ColumnSpace` method that sums the per-column values.
- [x] A6 (done) `issphere` (`utilities.jl`) queries the topology and errors on a
      `MultiPointSpace`; unreachable today because `iscolumn` is tested first in NOGW.
      Use `Spaces.global_geometry` instead of the topology (see DIFF_SPACE E5).
- [x] A7 `compute_rv` DSS guarded with `do_dss` (also makes `rv` work on a single column).
- [x] A8 (done: the assertion at `model_getters.jl:853` accepts `"multicolumn"`; the diurnal case matches, §7) `ReanalysisMonthlyAveragedDiurnal` asserted `config == "column"`
      (`model_getters.jl:853`); decide whether `multicolumn` is allowed once per-column
      forcing exists (§6).
- [ ] A9 (only affects our bitwise checks; not for the PR) Uninitialized cache fields: `p.precomputed.ᶜmp_tendency` is allocated with
      `similar` and, for the Beres NOGW EDMF+0M case, `dq_tot_dt` held garbage in one run
      and zeros in the other while the state matched exactly. Not a multi-column issue,
      but it shows in any bitwise comparison; either zero-initialize or keep it in the
      comparison ignore list.
- [ ] A10 GPU and MPI: `MultiPointGrid` is single-process only (`SingletonCommsContext`),
      so multi-column + MPI is unsupported by construction; GPU was not tested here.
- [x] A11 (done: `test/grids.jl`, `test/config/model_from_config.jl`, `test/diagnostics/unit_diagnostics.jl` multicolumn fixture, `test/restart.jl`, `test/restart_AtmosSimulation.jl`, two `full_pipeline.yml` steps, `docs/src/configuration.md`, `api.md`, `interfaces.md`) Tests, CI, docs: add a `multicolumn` job (the driver above, like ClimaLand's
      "ERA5 Column vs ColumnEnsemble" step), a `MultiColumnGrid` entry in
      `test/restart_AtmosSimulation.jl` and `test/diagnostics/unit_diagnostics.jl`
      fixtures, a `check_case_consistency` test for `valid_configs`, and
      `MultiColumnGrid` in `docs/src/api.md` / `docs/src/interfaces.md`.
- [x] A12 (plots done 2026-09-11: one file per column, see §9 "Known gaps"; MSE tooling still open) Post-processing: `post_processing/ci_plots.jl` column plots read NetCDF with a
      `(time, z)` layout; multi-column files carry a `column` dimension with `lat`/`lon`
      coordinates and a CF `featureType`, so the `Val(:single_column_*)` plot methods will
      need a column selection before they work on multi-column output.

## 4. ClimaCore changes (PR branch `multi-col-extras`, `~/worktree/ClimaCore.jl/multi-col-extras`; C1-C4 committed there by the user; worktree clean)

The worktree `col-intp` is superseded: `multi-col-extras` contains C1-C4, `field2arrays`,
the multi-column unit test, and the user's unit-metric change (`a8ae99ac`, handoff §0c).

- [x] C1 (done in `kp/col-intp`) `Spaces.node_horizontal_length_scale(::MultiPointSpace) = 1`, mirroring the
      `PointSpace` method. Hard prerequisite: every run in §7 used it as a REPL shim, and
      without it (or an Atmos guard, A1) a default `config: multicolumn` run crashes in
      hyperdiffusion. Also unblocks the LES closures.
- [x] C2 (done) `Spaces.quadrature_style(::MultiPointSpace) = nothing` at the space level
      (`Grids.quadrature_style(::MultiPointGrid)` already returns `nothing`); then the
      generic `do_dss` works without the Atmos `ColumnSpace` method.
- [x] C3 (done, plus `MatrixFields.field2arrays` on multi-column fields) `Spaces.all_nodes(::MultiPointSpace)` for parity with `PointSpace`
      (`((1, 1), h)` for `h in 1:ncolumns`); no longer needed by Atmos after the
      `ncolumns` change, but `MatrixFields.column_map` still relies on it.
- [x] C4 (done: `deep` keyword) `CommonGrids.MultiColumnGrid` does not forward `deep`; the multi-column grid is
      always `ShallowSphericalGlobalGeometry`. Needed only if deep multi-columns are
      wanted (see A2).
- [ ] C5 (nice to have) an abstract type or trait shared by `FiniteDifferenceSpace` and
      `MultiColumnFiniteDifferenceSpace` ("no horizontal connectivity") so downstream
      packages do not need their own `Union`.
- [ ] C6 (found, fixed for the GPU runs in §9, then **reverted** at the user's request,
      2026-09-11: pre-existing in the released ClimaCore 0.16.0, unrelated to
      multi-column; recorded in §8) `ext/cuda/operators_fd_eager.jl`:
      `Utilities.unsafe_eltype(::CUDA.CuRefType{T})` returns `T` where the host returns
      `Type{T}` for the same `Ref(T)` argument. See §8 for the mechanism and reproducer.
- Verified fine without changes: `Spaces.space(grid, staggering)` for all grid kinds,
  `Fields.level`, `Fields.column`, `Fields.field2array` (`(nlevels, ncolumns)`),
  `Fields.array2field`, `column_integral_definite!`, `column_reduce!`, the MatrixFields
  operator matrices and the manual sparse Jacobian, HDF5 checkpoint write/read, the
  horizontal spectral operators (zero on both column kinds because `Ni = Nj = 1`), and
  the `PressureInterpolator` path from `kp/col-intp` (pending: §7 output variants).

## 5. ClimaDiagnostics (branch `kp/multi-cols`; no change needed)

- Works: `NetCDFWriter` on `MultiColumnFiniteDifferenceSpace`/`MultiPointSpace` writes
  every default diagnostic along a `column` dimension with `lat`/`lon` auxiliary
  coordinates and `featureType` (`timeSeriesProfile` / `timeSeries`); values match the
  single-column files per column (rtol 1e-9 in Float64). `netcdf_interpolation_num_points`
  horizontal entries are ignored with a warning, as designed.
- [ ] D1 Layout difference is inherent: single-column files are `(time[, z])`,
      multi-column files `(time, column[, z])`. Consumers (ClimaAnalysis, `ci_plots.jl`,
      reproducibility MSE tooling) need a column selection.
- [ ] D2 Release 0.4.0 and bump the Atmos compat.

## 6. ClimaUtilities / forcing data (the "very hard" cases)

File-forced setups (`ForcingFromFile`: GCM-driven, ARM VARANAL, `ReanalysisTimeVarying`)
read one site's `(z, time)` profiles and `(time)` surface series:

- Initial profiles are 1-D interpolants in `z`, evaluated pointwise: fine for any number
  of columns (all identical).
- Profile forcing goes through `TimeVaryingInput(path, var, target_space)` →
  `InterpolationsRegridder`, which builds an interpolant over the file dimensions and
  evaluates it at `totuple(coord)` of every target coordinate. A single column has
  `ZPoint` coordinates (1 argument); a multi-column space has `LatLongZPoint` (3
  arguments) against a 1-D `(z)` interpolant. The regridder has a special case only for
  the opposite situation (Z-only target with 3-D data: it takes the centre column).
  [x] U1 ClimaUtilities: when the target has lat/long but the data has only `z`,
      evaluate the `z` interpolant at each coordinate's `z`. Prototype in
      `~/worktree/ClimaUtilities.jl/forcing-intp` (§0 item 1). GCM-driven never needed it
      (its file dimensions already fit); ARM VARANAL and ERA5-driven match exactly with it.
      **Obsolete since 2026-09-15**: the regridder path is no longer used for column
      forcing (§11).
- Surface series (`ts`, fluxes, `coszen`, `rsdt`) are 0-D `TimeVaryingInput`s and
  broadcast to every column: fine for identical columns.
- [x] U2 (required, user 2026-09-10; see §0 item 2, including the ragged-time-axis analysis; **done 2026-09-15, §11**) Per-column (multi-site) forcing (partly started upstream: ClimaUtilities 0.1.32
      `TimeVaryingInput` accepts matrix input, one time series per column) does not exist end to end: `ColumnDatasets` reads
      one file for one site, `site_latitude`/`site_longitude` are scalars,
      `era5_observations_to_forcing_file.jl` produces one site. Supporting different
      forcing per column needs a column dimension (or a file per column) in
      `ColumnDatasets`, per-column `TimeVaryingInput`s (`InterpolatingTimeVaryingMultiPoint`
      in ClimaUtilities is a start), and per-column surface temperature / fluxes /
      insolation. This is a design project, not a fix.

## 7. Verification ladder (runs in this session)

Setup for every row: three columns at (0, 0), (0, 0), (30, -50); columns 1 and 2 must be
identical; column 1 is compared with the single-column run for `Y`, the cache `p`
(bookkeeping and scratch excluded), and every NetCDF diagnostic. "Match" means rtol 1e-9
(atol 1e-10) unless noted. Column 3 differs wherever latitude enters (§3 A3). The rows
below were recorded when `atol` was applied to the RMS error, which let an isolated
absolute difference be averaged away; `atol` is now applied to the worst absolute error,
so rows whose margin came from the RMS need a re-run to still count as a match.

| Config | Runs on multi-column | Column 1 vs single column | Notes |
|---|---|---|---|
| `single_column_hydrostatic_balance_ft64` | yes | match (Float64, 10 days); re-verified through `config: multicolumn` + `deep_atmosphere: false` | NetCDF match; CI driver `single_vs_multi_column.jl` passes on it |
| `single_column_radiative_equilibrium_gray` | yes (after `ncolumns` fix) | 2% off with `deep_atmosphere: true` (RRTMGP area scaling); Float32 + shallow: 1e-4; Float64 + shallow: match | A2 |
| `..._clearsky` | yes | same pattern as gray | A2 |
| `..._allsky_idealized_clouds` | yes | same pattern as gray | A2 |
| `..._clearsky_prognostic_surface_temp` | yes | Float64 + shallow: match, incl. `Y.sfc.T` | slab ocean on `MultiPointSpace` fine |
| `single_column_nonorographic_gravity_wave` | yes | match (Float64; 444 cache entries, vectors included) | NOGW takes the column branch |
| `column_nogw_3d_test` | yes | match (Float64; re-verified with vectors) | |
| `single_column_beres_nogw_test` | yes | match (206 cache entries); columns 1 vs 2 differ only in the uninitialized `ᶜmp_tendency` entry | A9 |
| `single_column_precipitation_test` (1M) | yes (hyperdiffusion on; ClimaCore C1 landed) | Float64: match (9 state, 59 cache entries); Float32: 1e-5 rounding growth | |
| `single_column_precipitation_2M_test` | fails on a single column too | n/a | 2M disabled upstream (CloudMicrophysics 0.37) |
| `prognostic_edmfx_simpleplume_column` | yes | match (Float64, shallow, 1 h; re-verified with vectors, 108 cache entries) | |
| `prognostic_edmfx_adv_test_column` | yes | match (80 NetCDF files); only uninitialized `ᶜmp_tendency`/`ᶜsgs_moments` entries differ | A9 |
| `prognostic_edmfx_gabls_column` | yes | match (Float64, 1 h; 18 state and 96 cache entries, vectors included) | SCM Coriolis + geostrophic wind path fine |
| `prognostic_edmfx_bomex_column` | yes | rounding-flip sensitivity of the EDMF itself: a 1e-15 perturbation of the single column reproduces the multi-column difference exactly | B1 resolved below |
| `prognostic_edmfx_dycoms_rf02_column` | yes | match (18 state, 116 cache entries) | |
| `prognostic_edmfx_dycoms_rf01_column` | yes | differs after 1 h (13/18 state entries, `ρtke` rel. err 1); single column perturbed by 1e-15 in `uₕ` gives the identical table; perturbation in `ρ` gives nothing | EDMF rounding sensitivity, proven |
| `prognostic_edmfx_trmm_column`, `..._0M` | yes | differ after 1 h (updraft `ρa` flips); perturbed single column gives the same fields and magnitudes (not identical numbers) | EDMF rounding sensitivity |
| `prognostic_edmfx_bomex_fixtke_column` | yes | differs after 1 h (12/18); perturbed single column 14/18 with the same fields | EDMF rounding sensitivity |
| `prognostic_edmfx_soares_column` | yes | `ρtke` 7e-9 only; perturbed single column: 1 entry | rounding growth |
| `prognostic_edmfx_bomex_mlcloud_column` | yes | `ρtke` 4e-8, `sgsʲs.1.u₃` 2e-7; perturbed single column (whole state) 8 entries | rounding growth |
| `prognostic_edmfx_rico_column` | yes | on the real grid: 1 step → updraft `mse`/`q_tot` 7e-5, 1 h → 14/18 entries; **on the identity-metric grid (`identity_metric.jl`): bitwise identical after 1 h** (18/18 state, cache except uninitialized scratch) | proven metric rounding; the state-perturbation tests could not excite it because the sensitivity is to per-step operator rounding (`g³³`, `J`), not to the initial state |
| `larcform1_1M_prognostic_edmfx` | yes | state match (20 entries); cache `ᶜl_mix`/`ᶠK_h`/`ᶠK_u` 3e-7; NetCDF 112/114 match, `lmix`/`edt` 1e-7 | mixing-length amplification of ≤1e-9 state rounding |
| `kinematic_driver` | yes (after the `src/types.jl` fix) | match (Float64, 2 min; 9 state, 51 cache entries, 18 NetCDF files) | single column failed on main (`ITime < Float64` in `ShipwayHill2012VelocityProfile`); side fix verified |
| `prognostic_edmfx_diurnal_scm_imp` | yes, after widening the `config == "column"` assertion to both column kinds (A8) and with the ClimaUtilities change | Float64 (10 min): EDMF-level differences on the real grid (`ρe_tot` 3.5e-4, updraft `ρa` flips); **identity-metric grid: bitwise identical** | A8 done; metric rounding proven |
| `prognostic_edmfx_bomex_tracerA_column` | yes | match (20 state, 110 cache entries; NetCDF 112/113, `cl` 2e-9); perturbed single column: 0 entries | |
| `prognostic_edmfx_rico_column_2M` | fails on a single column too | n/a | 2M disabled upstream (CloudMicrophysics 0.37) |
| `bomex_sparse_autodiff`, `gabls_sparse_autodiff` | fail on a **single** column, Float32 and Float64 | n/a | pre-existing, not CI-covered: `AutoSparseJacobian.update_jacobian!` → `set_implicit_precomputed_quantities!` → `update_implicit_microphysics_cache!` → `set_precipitation_velocities!` (`src/cache/microphysics_cache.jl:226`) writes Dual-typed terminal velocities into the plain `ᶜwₗ`/`ᶜwᵢ`/`ᶜwᵣ`/`ᶜwₛ` cache fields (`precomputed_quantities.jl:295-298`), which are not part of `implicit_precomputed_quantities`. The multi-column `column_index_iterator` branch stays unexercised by a run |
| `prognostic_edmfx_gcmdriven_column` | yes (10 min) | Float64: match (10 state, 108 cache entries); Float32: 1e-6..1e-5 rounding growth | file forcing works unchanged for one site |
| `prognostic_edmfx_armvaranal_column` | yes, with the ClimaUtilities `forcing-intp` change (`horizontally_uniform = true`; without it an informative error, U1) | Float64 (10 min): **match** (18 state, 141 cache entries within 1e-9); Float32: 1e-7..1e-5 rounding growth | needs `ENV["BUILDKITE"] = "true"` here for the converted forcing file |
| `prognostic_edmfx_tv_era5driven_column` | yes, with the ClimaUtilities change (`horizontally_uniform = true`; artifact present) | Float64 (10 min, 60 steps, 200 levels, all-sky RRTMGP): state within 4e-9 (`uₕ` 3.7e-9, `ρe_tot` 3e-9) except one near-zero `ρtke` cell (relative 2e-2, RMSE 4e-9); cache `ᶠK_h` 1e-2 relative at RMSE 4e-7; **identity-metric grid: bitwise identical** | U1; metric rounding proven |
| checkpoint + restart (multi-column) | yes | restart from the day-1 HDF5 checkpoint matches the continuous 2-day run to 1e-12 | `restart_file` through the config path |
| config-driven `config: multicolumn` | yes | every run in this table now goes through `CA.get_simulation` with `config: multicolumn` (`multi_column_dev/common.jl`) | |
| pressure-coordinate diagnostics / fake pressure levels | yes | 21 and 17 NetCDF files match column by column (rtol 1e-9) | ClimaCore `kp/col-intp` `PressureInterpolator` |

### Metric-rounding attribution by construction (`multi_column_dev/identity_metric.jl`)

`BITWISE_COMPARSION_HANDOFF.md` shows where the real grid rounds differently: `J = J_h·J_v`
and the 3×3 inverse of `∂x∂ξ = diag(R·π/180, R·π/180, Δz)` give `g³³`, `∂ξ∂x[3,3]`, and `J`
that differ from the single column's by up to 3 ulp, so every metric-dependent vertical
operator rounds differently at every step. `identity_metric.jl` rebuilds the multi-column
grid (REPL only) with an identity horizontal metric, which removes exactly those
differences. Results after the full run: hydrostatic balance (1 day), **RICO, Bomex,
DYCOMS RF01, TRMM (1 h each), ERA5-driven and the diurnal SCM case (10 min) bitwise identical** to the single
column in state and cache (only uninitialized `similar` scratch differs). A case that matched bitwise on the identity grid owed its whole
difference on the real grid to metric rounding; this is the constructive test to use before
calling any difference "rounding". Bitwise identity on the real grid is not a goal (user).

### B1 resolved: the Bomex difference is the EDMF's own rounding sensitivity

`multi_column_dev/t0_diag.jl`: at t = 0 the state, the cache after
`set_precomputed_quantities!`, `implicit_tendency!`, and `remaining_tendency!` agree to
1e-13 (the only entries flagged are a 2e-16 rounding in `Y.c.uₕ`, whose covariant
components carry the spherical metric, and the uninitialized hyperdiffusion scratch
`p.hyperdiff.ᶜ∇²u`). `multi_column_dev/bomex_diag.jl`: after one 120 s step, single vs
column 1 differs in `sgsʲs.1.mse` (7.1e-5), `sgsʲs.1.q_tot` (8.7e-5), and `sgsʲs.1.u₃`
(relative error 1, RMSE 1e-1, a zero/nonzero flip); a single column whose initial `uₕ` is
multiplied by `1 + 1e-15` differs from the unperturbed single column by *exactly the same
numbers*. The discrepancy is therefore one rounding flip in the updraft limiters, present
in the single-column model itself, and not a multi-column error. Bomex is expected to
diverge at this level between any two runs that differ by one ulp.

The same holds for DYCOMS RF01 after 1 h (`multi_column_dev/sens.jl`): column 1 vs single
differs in 13 of 18 state entries (`ρtke` relative error 1, `ρq_tot` 0.06, `u₃` 70), and a
single column with `uₕ` multiplied by `1 + 1e-15` reproduces those numbers exactly. The
perturbation has to be in `uₕ` (the field whose covariant components round differently on
the spherical grid): a 1e-15 perturbation of `ρ` changed only `ρtke`, by 1e-9. DYCOMS RF02
(same setup code, different profiles and fluxes) matches exactly, as do GABLS, simple
plume, and the advection test. For EDMF cases the meaningful single-vs-multi check is
therefore "does the multi-column difference equal the single column's own 1-ulp
sensitivity", which `sens.jl` provides.

### Original open question B1 (kept for the record)

Simple plume, advection test, and GABLS match exactly; Bomex does not. Bomex adds
subsidence, large-scale advection, prescribed surface fluxes
(`MoninObukhov(; θ_flux, q_flux, ustar)`), and `SurfaceBoundaryOverrides(p, q_vap)`; GABLS
shares the SCM Coriolis/geostrophic forcing, so that path is not the cause. Step-growth
(Float64, dt = 120 s; scalar fields only, see the vector caveat): 1 step → `sgsʲs.1.mse`
7e-5 and `q_tot` 9e-5 relative, all grid-mean scalars ≤ 1e-12; 3 steps → `ρa` flips (relative error 1) in some cells; 10 steps
→ `ρtke` 0.86. Next action: `multi_column_dev/t0_diag.jl` compares the cache after
`set_precomputed_quantities!` and the implicit/explicit tendencies on the identical
initial state and names the first differing field.

Order in which a default multi-column run fails without the fixes: hyperdiffusion length
scale (A1/C1) → RRTMGP `all_nodes` (fixed) → nothing else.

## 8. Broken independently of this work (not addressed here)

Single-column failures that predate the branch and are not covered by CI. They should
probably be fixed at some point, but they are out of scope for the multi-column work.

- **Sparse-autodiff column configs** (`prognostic_edmfx_bomex_column_sparse_autodiff.yml`,
  `prognostic_edmfx_gabls_column_sparse_autodiff.yml`) fail on a single column, Float32
  and Float64, with `Float64(::ForwardDiff.Dual{ClimaAtmos.Jacobian})`. Mechanism:
  `AutoSparseJacobian.update_jacobian!` (`auto_sparse_jacobian.jl:481`) evaluates
  `set_implicit_precomputed_quantities!` with a Dual-typed state; since #4744
  (`c3c493a49`) `update_implicit_microphysics_cache!` (`microphysics_cache.jl:746`)
  recomputes the sedimentation velocities when a terminal velocity is
  `DiagnosticTerminalVelocity` (rain, by default) through `set_precipitation_velocities!`
  (`:226`), writing into the plain `FT` fields `ᶜwₗ`/`ᶜwᵢ`/`ᶜwᵣ`/`ᶜwₛ`
  (`precomputed_quantities.jl:295-298`) instead of Dual-typed copies; the comment at
  `precomputed_quantities.jl:106-108` predates #4744. The default `ManualSparseJacobian`
  never evaluates with Duals and the only sparse-autodiff CI job is the dry baroclinic
  wave, so the combination is never exercised. Fix: Dual-typed copies of the four fields in
  `implicit_precomputed_quantities`, as `ᶜρ_dq_tot_dt` has for 0M. Not verified on `main`
  (needs a `main` worktree and environment); none of the files in the stack is changed on
  this branch. Consequence for us: the multi-column branch of `column_index_iterator`
  (`autodiff_utils.jl`) is only covered by ClimaCore's `all_nodes` unit test, not by a run.
- **2M and 2M+P3 microphysics** (`single_column_precipitation_2M_test.yml`,
  `prognostic_edmfx_rico_column_2M.yml`): `precomputed_quantities.jl:164` asserts they are
  disabled pending a CloudMicrophysics 0.37 fix.
- **`check_conservation: true` on any column** (A5): `horizontal_integral_at_boundary`
  (`utilities.jl:793-800`) asserts an extruded spectral-element space, so the conservation
  callback and the `energyo`/`watero` diagnostics fail on a single column as well.
- **GPU: ClimaCore's eager stencil kernel fails to compile under Julia 1.12.5 / CUDA.jl 6.2.2** for any stencil whose argument is a view-backed `Covariant12Vector` field converted with `CT3` (e.g. the first stencil of `set_implicit_precomputed_quantities!`, `ᶠwinterp(ρ J, CT3(Y.c.uₕ))`), on every space type; passes under Julia 1.11.4 / CUDA.jl 5.11.3. Details and reproducers in §9 "GPU". Not multi-column related.
- **GPU: every file-forced column config fails at initialization** (GCM-driven, ARM
  VARANAL, ERA5-driven, reanalysis diurnal; single column too; CPU unaffected).
  `update_surface_conditions!` (`src/surface_conditions/surface_conditions.jl:52`)
  broadcasts `surface_state_to_conditions` with the whole `AtmosModel` as an argument,
  and with file forcing that object is not isbits: `scm_setup.external_forcing.dataset`
  is a `ColumnDataset` (`String` path, `Vector{Symbol}` variable lists) and the `Nudging`
  terms hold `Tuple{Symbol, Symbol}`. CUDA rejects the kernel: "Argument 3 to your kernel
  function is of type Broadcasted{...} which is not isbits" (full dump of the stack in
  `multi_column_dev/logs/gpu_error_tv_era5driven_single.txt`, first version). The
  function only reads `atmos.microphysics_model isa DryModel`, so passing
  `atmos.microphysics_model` instead of `atmos` (and renaming the parameter) fixes it;
  that one-line change was made, used for the GPU runs in §9, and then reverted at the
  user's request (2026-09-11): pre-existing on `main`, not CI-covered, out of scope.
  Consequence: the ERA5-driven and diurnal GPU results in §9 were obtained *with* that
  change (and the ClimaCore one below) applied; on the branches as they stand those two
  configs do not start on the GPU, on one column or many.
- **GPU: ClimaCore's eager stencil kernel fails on an operand that carries a vector
  type through `Ref`** (Atmos subsidence forcing,
  `@. ᶠinterp(ᶜls_subsidence * CT3(unit_basis_vector_data(CT3, ᶜlg)))`,
  `src/prognostic_equations/forcing/external_forcing.jl:230`; single column too; every
  file-forced config with `Subsidence`). Error at the first time step: `InvalidIRError
  ... unsupported dynamic function invocation (call to CuDynamicSharedArray)` in
  `calc_level_val` (`ext/cuda/operators_fd_eager.jl:332`). Mechanism
  (`multi_column_dev/gpu_probe7.jl`, `gpu_probe8.jl`): on the host `@.` wraps `CT3` as
  `Ref{Type{CT3}}`, whose eltype is `Type{CT3}`, so the launch-side
  `cached_operand_type` infers `Contravariant3Vector{FT}` and sizes the shared memory;
  on the device the same argument is `CUDA.CuRefType{CT3}` and ClimaCore defines
  `Utilities.unsafe_eltype(::CuRefType{T}) = T`, i.e. the vector type itself rather than
  `Type{T}`, so the operand eltype infers to `Union{}`, `cached_operand_type` returns
  `nothing`, and `CuDynamicSharedArray(nothing, ...)` is a dynamic call. Changing that
  method to `Type{T}` (both CUDA-version branches, `operators_fd_eager.jl:153-155`) makes
  host and device agree and the stencil compiles (`gpu_probe8.jl`: "copyto! ok"). The
  change was applied for the ERA5-driven and diurnal GPU runs in §9 and then reverted at
  the user's request (2026-09-11); the released ClimaCore 0.16.0 has the same line, so
  this is pre-existing and belongs in a ClimaCore issue/PR of its own. As with the
  surface-conditions note above, the two GPU rows in §9 hold only with it applied.
- **`@fused_direct` with a local destination variable fails under Julia 1.12.5**
  (`UndefVarError: ... not defined in Main`), which breaks the CloudSat subcolumn path
  (`src/cosp/cloudsat.jl:271`, also `cloudsat_optics.jl`, `hydrometeor_subcol.jl`) and
  the two COSP testsets in `test/cosp/subcol_test.jl` under `TEST_GROUP=diagnostics`.
  Standalone check: `multi_column_dev/fused_check.jl` (fails for `ᶜq` and plain `q`
  alike). MultiBroadcastFusion is 0.3.4 in both manifests and the ClimaCore branch does
  not touch it; CI runs Julia 1.10 and 1.11 and is unaffected. Probably to be addressed
  when the repo moves to Julia 1.12 (MultiBroadcastFusion `@make_fused`, nested `quote`
  with `esc`).

## 9. Everything left for the user, including the small things (2026-09-10)

### Decisions and PR preparation
- ClimaUtilities PR for the opt-in `horizontally_uniform` regridder keyword (§0 item 1);
  rename the keyword if wanted; run the ClimaUtilities test suite (not run here).
- ClimaCore PR from `~/worktree/ClimaCore.jl/col-intp` (C1-C4, `field2arrays`, unit test,
  NEWS; only `test/Spaces/unit_multicolumn.jl` was run, not the whole suite), then a
  release; ClimaDiagnostics 0.3.10 release (D2) and any Atmos compat bump; then the
  `.buildkite/Manifest.toml` clean-up and `.gitignore:28` (user-owned).
- Which CI steps to keep: the two `single_vs_multi_column.jl` steps in
  `.buildkite/full_pipeline.yml` (hydrostatic verified end to end; gray radiative is 654
  model days twice). If an EDMF config is ever added there, its bitwise duplicate-column
  check will trip on uninitialized cache scratch (A9) unless those fields are zero-filled
  or ignored.
- What to commit: `multi_column_dev/` (the newest scripts `identity_metric.jl`,
  `queue7*.jl`, `queue8b-d.jl`, `forced_case.jl`, `fused_check.jl` are unformatted),
  the Markdown documents (the CI formatter hook also formats Markdown; they were not run
  through it), and the other agent's `hb_bitwise.jl`/`hb_1day.yml`/
  `BITWISE_COMPARSION_HANDOFF.md`.

### Not verified here
- GPU: nothing was run on GPU. `MultiPointGrid` builds its local geometry on the device
  (`DataLayouts.rebuild`), the comparison utilities copy to `Array`, and the autodiff
  `column_index_iterator` branch is plain Julia, but none of it was exercised (A10).
- MPI: unsupported by construction (`MultiPointGrid` is `SingletonCommsContext` only;
  `MultiColumnGrid` errors otherwise). Documented, not tested.
- Julia 1.10/1.11: every run here used Julia 1.12.5 with the user's manifest; CI runs
  1.10 and 1.11 with registered packages.
- Tests not run: `test/restart.jl` "multicolumn" configuration (MANYTESTS mode only),
  `test/restart_AtmosSimulation.jl` (not in `runtests.jl`), the infrastructure, dynamics,
  parameterizations, restarts and era5 `Pkg.test` groups (grids and config tests were run
  by `include`; the diagnostics group ran, see §8 for its two Julia-1.12 errors).
- Docs build: failed locally before `makedocs` (InterLinks inventory download, no network);
  `checkdocs = :exports` should pass since `MultiColumnGrid` is in `docs/src/api.md`.
- The example driver was run end to end only on the hydrostatic config; the gray radiative
  config was compared through the ladder scripts, not the driver.
- The multi-column `column_index_iterator` branch (`autodiff_utils.jl`) has no run
  coverage because the sparse-autodiff configs fail on a single column (§8).

### Known gaps, documented and intentionally left
- A12/D1, plots part **done (2026-09-11, uncommitted)**: `post_processing/ci_plots.jl`
  `make_plots_generic` now loops over the `column` dimension of multi-column NetCDF
  output and writes one file per column (`summary_column_<i>.pdf`, and likewise for every
  other `output_name`, e.g. LARCFORM1's `summary_profiles_last_column_<i>.pdf`); vars are
  sliced with `slice(var; column = i, by = ClimaAnalysis.Index())`, nested (tuple) var
  groups included, and files handed on from an earlier per-column stage are indexed by
  column (`column_files`). Single-column output has no `column` dimension and is
  unchanged. Verified on `multi_column_dev/output`: hydrostatic balance (`ColumnPlots`),
  EDMF advection test (`EDMFColumnPlots`, chained `tmp` + `zt_contour` stages),
  LARCFORM1 (four plot stages incl. the 1M time series), and Bomex run for its full 6 h
  with the CI configs (`config/common_configs/diagnostics_column_progedmf_1M.yml` +
  `prognostic_edmfx_bomex_column.yml`, `EDMFColumnPlotsWithPrecip`; the CI plot methods
  need both `inst` and `average` reductions, which only the common diagnostics config
  provides). A single-column output still produces the one `summary.pdf`. Still open: the
  reproducibility MSE tooling and `ci_driver.jl` regression comparisons still read
  `(time, z)`.
- A4: `TimeVaryingInsolation` with an explicit site (ARM VARANAL) is broadcast to every
  column, overriding the columns' own coordinates. Right for one-site forcing, wrong for
  columns at other locations.
- Coriolis in file-forced single columns is the f-plane parameter (default 0); a
  multi-column column at the true site latitude gets `2Ω sin(lat)` instead (§0b).
- A9 uninitialized `similar` cache fields (bitwise checks only).
- A3: latitude-dependent physics differs by design for columns away from the equator.
- ClimaCore nits (worktree `~/worktree/ClimaCore.jl/col-intp`), neither needed for the
  multi-column work:
  1. `Spaces.all_nodes` element form. `all_nodes(::SpectralElementSpace2D)`
     (`src/Spaces/spectralelement.jl:202`) iterates `((i, j), h)` tuples, and the new
     `all_nodes(::MultiPointSpace)` (`src/Spaces/multicolumn.jl:153`) iterates
     `((1, 1), h)` to match it, but `all_nodes(::PointSpace) = (1,)`
     (`src/Spaces/pointspace.jl:68`) yields a bare `1`. Any generic
     `for ((i, j), h) in all_nodes(hspace)` loop therefore works on spectral-element and
     multi-point spaces and fails on a single column. Nothing hits it today:
     `MatrixFields.all_columns` (`src/MatrixFields/field2arrays.jl:99`) has its own
     `FiniteDifferenceField` method returning `(((1, 1), 1),)`, and Atmos no longer calls
     `all_nodes` (the RRTMGP `ncol` now uses `Spaces.ncolumns`). Fix, if wanted:
     `all_nodes(::PointSpace) = (((1, 1), 1),)`, then `all_columns` for
     `FiniteDifferenceField` can go through `all_nodes` like the other two.
  2. `Spaces.global_geometry(::PointSpace)`. `global_geometry(space::AbstractSpace) =
     global_geometry(grid(space))` (`src/Spaces/Spaces.jl:89`) and `PointSpace` has no
     `grid` method (it stores only a context and a local geometry,
     `src/Spaces/pointspace.jl`), so the call is a `MethodError`. Atmos's `issphere`
     (`src/utils/utilities.jl:948`) now uses `Spaces.global_geometry`; both call sites
     (`non_orographic_gravity_wave.jl:185, 378`) pass `axes(Y.c)`, a 3-D space, so this
     is unreachable, and the old topology-based `issphere` errored on a `PointSpace` too.
     There is no clean one-line fix: a `PointSpace` can come from `Spaces.level` of a
     Cartesian column (`ZPoint`) or from `Spaces.column`/`slab` of a spherical
     spectral-element or multi-point space (`LatLongZPoint`), so a Cartesian default would
     be wrong for the latter; a real fix would store the global geometry in `PointSpace`.
     Leave it unless a caller needs it.
- Per-column (multi-site) forcing (U2): implemented 2026-09-15 for the ClimaColumn-file
  cases (`ForcingFromFile` with a list of files, `ReanalysisTimeVarying` and the diurnal
  case at the columns' coordinates); GCM cfsites per column, per-column prescribed
  surface fluxes (`FileHeatFluxes`), and `TimeVaryingInsolation` with an explicit site
  are still one-site-for-all (§11).
- §8: sparse-autodiff configs, 2M, `check_conservation` on columns, `@fused_direct` under
  Julia 1.12.

### GPU (2026-09-10, one A100, `srun --gpus=1`, `CLIMACOMMS_DEVICE=CUDA`, Julia 1.12.5 manifest)
- Ran the whole ladder (`multi_column_dev/gpu_ladder.jl`, log `multi_column_dev/logs/
  gpu_ladder.log`, and the REPL 1 pane for the first stages, which scrolled away). Result:
  the kinematic driver matches single vs multi-column on the GPU (9 state, 51 cache
  entries, 18 NetCDF files); every other case fails while building the **single-column**
  simulation (and the multi-column one) with
  `InvalidIRError: compiling ... eager_copyto_stencil_kernel!` for a `Contravariant3Vector`
  face field, reasons "unsupported dynamic function invocation (call to calc_level_val(arg::F,
  hidx, space) where F<:Field ...)" and "(call to get_op_row(op_matrix::FDOperatorMatrix, ...))"
  in `ClimaCore/ext/cuda/operators_fd_eager.jl`. GCM-driven fails differently
  (`KernelError: passing non-bitstype argument` in a `view` kernel, not investigated).
- Reduced to a minimal reproducer (`multi_column_dev/gpu_probe2.jl`-`gpu_probe5.jl`):
  `@. ᶠuₕ³ = ᶠwinterp(ᶜρ * ᶜJ, CT3(Y.c.uₕ))`, the first stencil of
  `set_implicit_precomputed_quantities!`, fails when `uₕ` is a **property field of a
  `FieldVector`** (its data is a non-contiguous `SubArray` view of the state array) and the
  `CT3` conversion of a `Covariant12Vector` happens inside the eager stencil kernel. The same
  broadcast works with a plain (copied) field, with a property `ρ` or a property
  `Covariant3Vector`, with the conversion done outside the stencil, and on a 300-level
  column, which takes the non-eager (`copyto_stencil_kernel!`) path. It fails identically on
  a single column, a multi-column grid, **and a sphere (`SphereGrid(h_elem = 2)`)**, so it
  is not a column or multi-column issue.
- Environment here vs GPU CI: Julia 1.12.5 / CUDA.jl 6.2.2 / GPUCompiler 1.23.0 /
  ClimaCore 0.16.1 (+ `kp/col-intp`, whose only CUDA change is a kernel-naming call) vs
  `climacommon/2025_03_18` = Julia 1.11.4 / CUDA.jl 5.11.3 / GPUCompiler 1.17.1 /
  ClimaCore 0.16.0 (`.buildkite/Manifest-v1.11.toml`). GPU CI is green on `main`, so the
  failure is environmental: **under CI's module (Julia 1.11.4, CUDA.jl 5.11.3, ClimaCore
  0.16.0, `Manifest-v1.11.toml`) the same reproducer passes on the single column and the
  sphere** (`multi_column_dev/gpu_probe6.jl`), and **with the three dev'd worktrees
  (ClimaCore 0.16.1 branch) under Julia 1.11.4 / CUDA.jl 5.11.3 it passes on the single
  column, the multi-column grid, and the sphere** (`gpu_probe5.jl`). Verdict: the GPU
  failure is the Julia 1.12.5 / CUDA.jl 6.2.2 / GPUCompiler 1.23 environment. GPU CI is
  unaffected (Julia 1.11). Not investigated further (it is in ClimaCore's CUDA extension:
  a view-backed `Covariant12Vector` argument converted inside the eager stencil kernel makes
  `calc_level_val`/`get_op_row` dispatch dynamically); it belongs with the Julia 1.12 items
  in §8 for whoever moves the repo to 1.12.
- How the GPU ladder was made to run: a scratch environment outside the repo (a copy of
  `.buildkite/Project.toml` + `Manifest-v1.11.toml` with `[sources]` pointing at the three
  worktrees and the repo by absolute path, under the session scratch directory), Julia
  1.11.4 from `climacommon/2025_03_18`, one GPU via `srun --gpus=1`. Building it needed a
  depot garbage collection first: the home export was at 98% and `Pkg.gc()` had been
  blocked by an interleaved write in `~/.julia/logs/scratch_usage.toml` (two duplicated
  lines from 2026-05-07, removed; backup in the session scratch directory). `Pkg.gc()`
  freed 3.2 GB (134 orphaned packages, 29 artifacts).
- Manifest pitfall found on the way: a manifest entry for a dev'd package keeps the
  registry metadata of the version it replaced until the path is `Pkg.develop`ed again.
  The scratch manifest (and `.buildkite/Manifest.toml`, line ~387) record the dev'd
  ClimaCore's CUDA extension as `ClimaCoreCUDAExt = "CUDA"`, but the branch declares
  `["CUDA", "GPUCompiler"]` (the extension imports GPUCompiler). Whenever that extension
  has to be recompiled from the manifest (here: starting Julia with `-O1`), it fails with
  "Package ClimaCoreCUDAExt does not have GPUCompiler in its dependencies", Julia
  continues without the extension, and every GPU run dies with a `StackOverflowError` in
  `DataLayouts.DataScope(::Type{CuArray})` (the generic fallback recurses on
  `parent(::CuArray)`). Fix: `Pkg.develop(path = <worktree>)` again, which refreshes the
  entry to 0.16.1 with the right extension deps. The user's 1.12 manifest should get the
  same refresh before anyone recompiles that extension.
- Consequence: multi-column GPU verification is blocked until a single column runs on the
  GPU in this environment; the multi-column grid construction, the kinematic-driver run,
  and the comparison utilities themselves work on the GPU.

### GPU results under Julia 1.11.4 / CUDA.jl 5.11.3 (scratch environment, one A100)
Same ladder, same comparison as on CPU (column 1 vs the single column, both on the GPU):
- Match (rtol 1e-9, state, cache, NetCDF): hydrostatic balance; gray radiative
  equilibrium Float64 (Float32: rounding growth as on CPU); Beres NOGW (cache except
  uninitialized `ᶜmp_tendency`); precipitation 1M; kinematic driver (run under Julia 1.12
  before the environment problem was understood); EDMF simple plume (state exact, cache
  except scratch); EDMF advection test (state exact, 80 NetCDF files match, cache except
  scratch **and `ᶜwᵣ` at 1.4e-7 relative, RMSE 3e-8 on a scale of 7.65**, which was within
  1e-9 on CPU: a GPU-only difference in the diagnostic rain terminal velocity recomputed
  from a state that agrees to 1e-9; not explained, to be settled with an identity-metric
  run on the GPU). Remaining EDMF cases, step 5, and the remaining configs continue in a
  second allocation (`multi_column_dev/gpu_ladder2.jl`).
- Single-column GPU failures in this environment (the multi-column run is never reached,
  so these are pre-existing single-column GPU gaps, not CI-covered, not multi-column):
  - `single_column_radiative_equilibrium_clearsky_prognostic_surface_temp` (slab ocean):
    `InvalidIRError` in a `view` kernel (`DataLayouts` `kernel_function{typeof(view)}`)
    during initialization.
  - `single_column_nonorographic_gravity_wave` and `column_nogw_3d_test`:
    `InvalidIRError ... bycolumn_kernel!(single_column_accumulate!, ClimaAtmos.var"#439#443"{:ad99, 376, NTuple{376, Float64}, ...})`,
    "unsupported call to an unknown function (call to gpu_gc_pool_alloc)": the NOGW column
    branch's accumulate closure captures a 376-element tuple and allocates in the kernel.
  - 2M precipitation: disabled upstream (as on CPU).
- Under Julia 1.12.5 / CUDA.jl 6.2.2 (the user's manifest) only the kinematic driver
  runs; everything else fails in ClimaCore's eager stencil kernel (§8).

### GPU results on the PR branches (2026-09-10 evening; handoff §0c)

Environment: one A100 (`srun --gpus=1`, slurm job 257500, cancelled afterwards), Julia
1.11.4 / CUDA.jl 5.11.3 scratch environment (§9 "GPU results under Julia 1.11.4"),
`julia -O1`, ClimaCore `multi-col-extras` (unit horizontal metric), ClimaUtilities
`forcing-intp` at `1570399`, `ENV["BUILDKITE"] = "true"`. Configs as shipped (Float32),
`multi_column_dev/gpu_hard3.jl`, columns at (0, 0), (0, 0), (30, -50). The user asked for
the three hardest configs only, not the whole ladder.

| Config | Single column on GPU | Column 1 vs single | Notes |
|---|---|---|---|
| `larcform1_1M_prognostic_edmfx` (EDMF, 1M, all-sky RRTMGP, slab ocean; 1 h) | runs as is | state 20/20, cache 97/97 within rtol 1e-9; NetCDF 114/114 | duplicate column 2 bitwise identical to column 1; column 3 (30°N) differs in 10 state / 53 cache entries by design (insolation) |
| `prognostic_edmfx_tv_era5driven_column` (ERA5 file forcing, EDMF, 0M; 10 min) | failed twice before the fixes below, runs now | state 10/10, cache 150/150 within rtol 1e-9 (no NetCDF written in 10 min) | duplicate column bitwise; column 3 differs by design |
| `prognostic_edmfx_diurnal_scm_imp` (reanalysis diurnal forcing, EDMF, 0M; 10 min) | same failures, runs now | state 10/10 within rtol 1e-9; cache 148/150, the 2 are `ᶜmp_tendency.dq_tot_dt` / `e_tot_hlpr` (A9: `similar` scratch that the EDMF + 0M path never writes, `set_microphysics_tendency_cache!(…, ::EquilibriumMicrophysics0M, ::PrognosticEDMFX)` only fills `ᶜmp_tendency⁰` and `ᶜmp_tendencyʲs`; uninitialized device memory happened to differ); NetCDF 55/55 | duplicate column bitwise; column 3 differs by design |

Bitwise check on the unit-metric grid (`multi_column_dev/gpu_bitwise.jl`, ERA5-driven,
10 min, GPU, `rtol = atol = 0`): the state of column 1 and of duplicate column 2 is
**bitwise identical** to the single column (10/10 entries), and the cache is bitwise
identical in 148/150 entries; the two others are the A9 scratch fields
`ᶜmp_tendency.dq_tot_dt` / `e_tot_hlpr`, which even the two duplicate columns disagree on
(1e-19 level garbage), so they carry no information. Column 3 at 30°N differs in every
state entry and in 61 cache entries, starting with `p.core.ᶜf³` (Coriolis, zero at the
equator) and the surface conditions (insolation), as designed. So on the PR branches the
multi-column column is the single column, on the GPU, to the bit.

Failures found and fixed, all on the *single* column too (so pre-existing single-column
GPU gaps, not multi-column problems), in the order they appeared:

1. Construction, ERA5 and diurnal: "horizontally_uniform requires a lat-long-z or x-y-z
   space" from the new ClimaUtilities constructor; Atmos now derives the flag from the
   target space (§0 item 1, §2).
2. Initialization, ERA5 and diurnal: `update_surface_conditions!` kernel argument not
   isbits because the whole `AtmosModel` (with `ColumnDataset` and `Nudging{Tuple{Symbol,
   Symbol}}`) is broadcast. Worked around for these runs by passing
   `atmos.microphysics_model`; **reverted afterwards** (user, 2026-09-11: pre-existing on
   `main`, not CI-covered), so it is recorded in §8 and the ERA5/diurnal rows above hold
   only with that change applied. Affects every file-forced config on the GPU.
3. First time step, ERA5 and diurnal: `InvalidIRError ... call to CuDynamicSharedArray`
   in ClimaCore's eager stencil kernel for the subsidence forcing. Worked around by the
   one-line ClimaCore change described in §8, **reverted afterwards** like item 2, so the
   ERA5/diurnal rows above hold only with both changes applied.
   Full error dumps: `multi_column_dev/logs/gpu_error_tv_era5driven_{single,multi}.txt`.

Not rerun on the GPU after the fixes: the step-4 EDMF cases beyond GABLS/Bomex/Soares,
step 5 (restart, pressure diagnostics, GCM-driven, ARM VARANAL), kinematic driver. The
earlier Julia 1.11 ladder results (hydrostatic, gray, Beres NOGW, precip 1M, simple plume,
advection test, GABLS, Bomex) were obtained on the `col-intp` sphere-metric grid; with the
unit metric they should now match bitwise, which also settles the advection-test `ᶜwᵣ`
1.4e-7 GPU-only difference recorded above if rerun (not done: the user asked to stop at
the three hardest cases). The single-column GPU failures listed above (slab-ocean `view`
kernel, NOGW `gpu_gc_pool_alloc`) were not revisited; the slab-ocean one is the same
"argument not isbits" family as item 2 (`kernel_function{typeof(view)}` with the
`surface_state_to_conditions` broadcast), unverified.

### Housekeeping
- tmux sessions `julia-mc3` (formatter REPL), `julia-mc5` (identity grid installed),
  `julia-mc6`-`julia-mc8` (ClimaAtmos loaded with the opt-in ClimaUtilities) and `docs`
  are still open; `julia-mc`, `julia-mc2`, `julia-mc4` have exited Julia. Outputs are under
  `multi_column_dev/output/` (gitignored) and logs under `multi_column_dev/logs/`.

## 10. Before opening the Atmos PR (2026-09-11)

The user wants a PR for feedback now. What is done, what blocks CI, and what should not
ship. Recommended: open it as a **draft** that links the three dependency PRs, so the
design review starts while the releases are prepared; CI cannot be green before them.

### Dependencies and compat (blocking for green CI, not for review)
1. **ClimaCore.** Atmos needs, beyond the released 0.16.1: the multi-column methods
   `node_horizontal_length_scale` / `quadrature_style` / `all_nodes` for `MultiPointSpace`
   (C1-C3; without C1 a default `config: multicolumn` run crashes in hyperdiffusion),
   `field2arrays` on multi-column fields, the `deep` kwarg (C4), the unit metric, and
   pressure interpolation on multi-column spaces. Upstream state: pressure interpolation
   [#2629](https://github.com/CliMA/ClimaCore.jl/pull/2629) merged (unreleased); unit metric
   [#2635](https://github.com/CliMA/ClimaCore.jl/pull/2635) open; C1-C4 and `field2arrays`
   are on `kp/multi-col-extras` (pushed; commits `fee26092`, `97dc769e`, part of #2635 or a
   PR to open). Needs a ClimaCore release (0.16.2) and `Project.toml` compat
   `ClimaCore = "0.16.2"`. Released 0.16.1 has none of C1-C4 (checked in
   `~/.julia/packages/ClimaCore/lvrmc`).
2. **ClimaDiagnostics.** NetCDF output on a multi-column space needs
   [#188](https://github.com/CliMA/ClimaDiagnostics.jl/pull/188) "Add support for spaces of
   multiple columns" (open since 2026-08-26; the `kp/multi-cols` worktree is 0.3.9 + 3
   commits). Without it every multi-column run with NetCDF diagnostics fails, including the
   two new CI steps. Needs a release (0.3.10) and compat `ClimaDiagnostics = "0.3.10"`.
3. **ClimaUtilities.** Since 2026-09-15 (§11) Atmos needs the user's branch
   `kp/ragged-data-intp` (`~/worktree/ClimaUtilities.jl/ragged-data-intp`, 14 commits on
   v0.1.32, not yet a PR): `FileReaders.DataSource`, the `TimeVaryingInput(::DataSource
   / ::Vector{DataSource}, space; start_date, method, preprocess_func)` constructors and
   `RaggedInterpolatingTimeVaryingInput`. Released 0.1.32 has none of it, so **every
   file-forced column job fails at construction** (a `MethodError` on `DataSource`) until
   the branch is released; then `Project.toml` compat `ClimaUtilities = "0.1.33"` (or
   whatever version ships it). [#251](https://github.com/CliMA/ClimaUtilities.jl/pull/251)
   is no longer needed by Atmos (the `horizontally_uniform` keyword is gone from the Atmos
   code).
4. `.buildkite/Manifest.toml` (absolute `path = ...` entries for the three dev'd packages)
   is untracked after the squash; keep it out of the PR, CI resolves from `Project.toml`
   compat. The user owns this file.

### State of the branch (2026-09-11, after the user's squash)
`kp/multi-col` is one commit on `origin/main`, `7f61525d5` "Add support for multi col
simulation": 21 files, `src/` + `config/default_configs/default_config.yml` + docs +
tests + `post_processing/ci_plots.jl`. Left out on purpose and now untracked or
uncommitted: `.buildkite/Manifest.toml`, the analysis markdown files, `multi_column_dev/`
(gitignored in the working tree), `examples/multi_column/`, and the two
`full_pipeline.yml` comparison steps (still in the working tree, not committed). The
example reference was removed from `docs/src/configuration.md`.

### Other concerns before the PR (besides per-site forcing, §0 item 2, and the releases)
1. **No NEWS entry** in `NEWS.md` yet (suggested content above).
2. **No CI job exercises `config: multicolumn`** once the example steps are dropped;
   coverage is the unit tests only (`test/grids.jl`, `model_from_config.jl`,
   `unit_diagnostics.jl`, `restart*.jl`). Cheapest fix: one `ci_driver.jl` job with a
   small overlay (`config: multicolumn`, two or three columns) on an existing column
   config, e.g. hydrostatic balance; it would also exercise the per-column plots and the
   ClimaDiagnostics NetCDF path. Caveat: `ci_driver.jl` also runs the reproducibility
   comparison, whose tooling reads `(time, z)` output and has not been tried on a
   `column` dimension (A12/D1 remainder), so such a job may need
   reproducibility disabled or that tooling taught to loop over columns like the plots.
3. **Single-column CI against released ClimaUtilities**: the unconditional
   `horizontally_uniform` keyword (§10 item 3 above) is the one change that fails
   existing jobs until the #251 release; decide between waiting and the conditional form.
4. **Tests not run on the final tree**: `Pkg.test()` as a whole, and Julia 1.10/1.11 on
   CPU (CI covers them; every local run here was Julia 1.12.5, plus 1.11.4 on the GPU).
   `test/restart*.jl` were run in the second session and are unchanged since.
5. **Reviewer design questions** (from §2b/§9, none blocking): `ColumnSpace` union and
   `iscolumn` living in Atmos vs a ClimaCore trait (C5); the NOGW column branch now
   taken by N columns (I1); `horizontal_filter_scale = Inf` and `has_topography = false`
   for N columns (I4, I5); the `do_dss` guard in `compute_rv`; `check_case_consistency`
   rules (equal-length, non-empty lists); `deep_atmosphere` defaulting to `true` so a
   default multi-column run is not a single-column run (documented); the
   `MultiColumnGrid` constructor API (`points` kwarg) and `column_latitudes` /
   `column_longitudes` as the config surface.
6. **Behaviour gaps to state in the PR description**: per-site forcing for ClimaColumn
   files only (GCM cfsites, ARM, prescribed fluxes stay one-site-for-all, §11); the
   McICA cloud-cover diagnostics `clt`/`cltl` are sampled from one random stream shared
   by the solver's columns, so they differ between a one-column and a multi-column run
   and between co-located columns while states and fluxes agree (§11);
   `TimeVaryingInsolation` with an explicit site is broadcast to every column (A4);
   Coriolis is `2Ω sin(lat)` at each column's latitude where the Cartesian column has
   the f-plane default of zero; MPI unsupported by construction (`SingletonCommsContext`,
   A10); GPU not in CI (verified by hand for three configs, §9; two pre-existing
   single-column GPU failures for file-forced configs, §8); sparse autodiff / 2M /
   `check_conservation` broken independently (§8).
7. **No performance measurement** of N columns in one layout vs one column (not asked
   for; the `[perf]` pipeline is unaffected, but a reviewer may ask).
8. **ClimaCore side**: #2635 replaces the sphere's horizontal metric by the identity;
   a ClimaCore reviewer may want that argued (columns become Cartesian columns with
   spherical coordinates and global geometry), and C1-C4 ride on the same branch.

### Verified on the current tree (2026-09-11)
- `test/grids.jl` and `test/config/model_from_config.jl` pass on Julia 1.12.5 /
  `.buildkite` env (`julia-mc5`, isolated modules). `TEST_GROUP=diagnostics` passed earlier
  in the session apart from the two pre-existing COSP errors (§8). `test/restart*.jl` were
  run in the second session; not rerun after the last edits (the edits do not touch
  restart).
- Formatting: all touched `src/`, `test/`, `examples/`, and `multi_column_dev/` files were
  run through the pinned JuliaFormatter (`.dev/format`, `julia-mc3`).
- No `println`/`@show`/`TODO` left in the `src/` diff against `origin/main`.

## 11. Per-site forcing with the ragged `TimeVaryingInput` (2026-09-15)

The user's ClimaUtilities branch `kp/ragged-data-intp` (`~/worktree/ClimaUtilities.jl/
ragged-data-intp`, 14 commits on v0.1.32; design notes `RAGGED_FORCING_DESIGN.md`,
`RAGGED_FORCING_PLAN.md`, `RAGGED_INTP_PERFORMANCE.md` in that worktree) adds
`FileReaders.DataSource` (a description of one NetCDF variable: paths, name, dates,
coordinate names), `TimeVaryingInput(sources, space; start_date, method, preprocess_func)`
with one `DataSource` per column of `space` (or one source shared by all columns), and the
`RaggedInterpolatingTimeVaryingInput` behind it: every column's time series is regridded
onto the model levels when the input is built (linear in `z`, held constant beyond the
file's levels), the series are stored contiguously with offsets so that columns may have
different time axes, columns whose sources compare equal share one segment, and
`evaluate!` does a per-column binary search in time and the linear blend, in one broadcast
(CPU and GPU). It is used here for all column forcing.

### Design in Atmos (all in the working tree, formatted, uncommitted)
- **One code path for all column inputs.** `ColumnDatasets.column_timevaryinginputs(data,
  names, target_space, start_date; method)` builds
  `TimeVaryingInput(data_source(data, name), target_space; start_date, method,
  preprocess_func = preprocess(format, name))` for every variable, where `data_source`
  gives one `DataSource` for a `ColumnDataset` (shared by every column of the space) and a
  vector of them for a vector of `ColumnDataset`s (one per column). The single column and
  the multi-column space go through the same constructor; nothing dispatches on the space
  type any more. `surface_timevaryinginputs` is the same call on the surface space (a
  `(time,)` variable becomes a one-level segment). The `InterpolationsRegridder`, the
  `horizontally_uniform` flag (§0 item 1), the 0-D in-memory surface inputs, and the
  format hook `extrapolation_bc` are gone from `ColumnDatasets.jl`
  (`read_surface_series` stays for `FileHeatFluxes`).
- **Data model.** `ColumnDatasets.ColumnData = Union{AbstractColumnData,
  AbstractVector{<:ColumnDataset}}`: one source for every column, or one `ColumnDataset`
  per column in column order. `ExternalDrivenTVForcing{CD <: ColumnData}` and
  `ForcingFromFile{CD <: ColumnData}` accept both. The vector gets five one-line methods:
  `require_forcing_variables` (every file must carry the variables), `_format` (the files
  share their format), `time_interpolation_method`, `file_time_span` (the shortest file
  bounds the run), `read_initial_profiles` (one profile set per column).
- **Pairing is positional for the forcing** (source `c` feeds column `c` of
  `field2array`, the user's ClimaUtilities decision Q6-Q8) **and by location for the
  initial condition**: `center_initial_condition` is pointwise in the local geometry and
  has no column index, so `ForcingFromFile` takes `sites` (the columns' `LatLongPoint`s,
  built by `column_sites(parsed_args, FT)` with the same expression `get_grid` uses) and
  picks the profiles of the site whose coordinates equal the point's. The two agree
  whenever coordinates are unique; two columns at one location must therefore share their
  file (construction error otherwise, "columns at one location must share their file"),
  which is also the case the shared-segment dedup covers. `TimeVaryingInsolation` is
  location-based in the same way.
- **Config surface.** `external_forcing_file` may be a list with one file per column of a
  `multicolumn` configuration (`ForcingFromFile`; `column_datasets(parsed_args)` checks the
  length). `ReanalysisTimeVarying` and `ReanalysisMonthlyAveragedDiurnal` in a
  `multicolumn` configuration take the sites from `column_latitudes` /
  `column_longitudes` (`era5_datasets`, one `era5_dataset` per distinct site, shared by
  the columns at that site); `site_latitude` / `site_longitude` apply to `config: column`
  only. **Decision to confirm with the user**: this changes an existing `multicolumn` +
  ERA5 run, which used `site_latitude` / `site_longitude` for every column; the
  same-site-everywhere run is now `column_latitudes: [17, 17, 17]`. Help texts in
  `default_config.yml` and `docs/src/configuration.md` / `column_datasets_reference.md`
  say so.
- Files: `src/column_datasets/ColumnDatasets.jl`, `src/setups/ForcingFromFile.jl`,
  `src/types.jl` (`ExternalDrivenTVForcing`), `src/config/type_getters.jl`
  (`get_setup_type`, `column_sites`, `column_datasets`), `src/config/
  era5_observations_to_forcing_file.jl` (`era5_datasets`), `src/config/model_getters.jl`
  (diurnal case), `config/default_configs/default_config.yml`, `docs/src/api.md`
  (`ColumnData` added, `extrapolation_bc` removed), `docs/src/configuration.md`,
  `docs/src/column_datasets_reference.md`, `test/column_datasets_tests.jl` (new testset
  "One file per column": two files with different values and different time axes on a
  three-column grid, per-column profiles, surface series, `file_time_span`,
  `require_forcing_variables`, per-column initial condition, the two construction
  errors; the surface-series test now evaluates on the surface space, as the model does,
  since a per-column input cannot be broadcast into a 3-D field like the old 0-D one).

### Verified (CPU, `.buildkite` env, Julia 1.12.5, ClimaCore `multi-col-extras`,
### ClimaUtilities `ragged-data-intp`, ClimaDiagnostics `multi-cols`)
- `test/column_datasets_tests.jl`: 100/100 pass (8 testsets, 245 s), including the new
  one (14 assertions).
- `multi_column_dev/ragged_inputs_check.jl` on the real ERA5 ClimaColumn file
  (`tvi_examples/data/tv_forcing_17.0_-149.0_20070701_20070701.nc` in the ClimaUtilities
  worktree; 37 levels, 24 hourly nodes, Float32 data), 200-level column, 139 times every
  10 min as `Float64` and as `ITime`: the surface series `ts`, `coszen`, `rsdt` are
  bitwise identical to the old 0-D inputs (same data, same stencil); the profiles differ
  from the old `InterpolationsRegridder` path (see the next item); a shared file on three
  columns (one dataset, or a vector of the same dataset) gives every column bitwise the
  single-column values at every time, with one segment in memory (`vals` is
  `(200, 24)`).
- `multi_column_dev/ragged_precision_check.jl`, both paths against a Float64
  Interpolations.jl reference at the 24 file nodes (no time interpolation involved), error
  `max|ref - x| / max|ref|`: on a **Float64** column the old path is exact (0 for every
  variable) and the new path is off by 2e-9 (`ta`) to 1.4e-7 (`wa`), i.e. one Float32
  rounding: `Utils.linear_interpolation` forms `(y1 - y0) / (x1 - x0)` in the file's
  `Float32` before promoting to the model's `Float64` (the old regridder converted the
  data to Float64 first). On a **Float32** column both are 4e-8 to 1.8e-7 from the
  reference, on equal footing. **For the ClimaUtilities branch:** promoting `block` and
  `z_src` to `eltype(model_z)` in `_regrid_block` (`ext/MultiSiteInputsExt.jl`) before
  `interpolate_columns!` would make the Float64 result exact again; not changed here (the
  user's branch). Consequence for Atmos: the Float64 file-forced single-column results
  move at the 1e-7 relative level, so reproducibility references of the ERA5 / ARM CI
  jobs may need regenerating either way.
- Cost of `evaluate!` for one 200-level profile (CPU, Float64): old 0.9 us when the two
  bracketing slices are cached and 48 us when a new slice has to be regridded; new 27 us
  always (per-level binary search over 24 nodes plus blend, no I/O). Twelve inputs per
  step is 0.3 ms, negligible against a model step; on the GPU it is one kernel per input.
- `multi_column_dev/era5_sites_check.jl` (`ENV["BUILDKITE"] = "true"`, files written to
  temporary directories): `era5_datasets` on the ERA5 config with `config: multicolumn`,
  `column_latitudes = [17, 17, 17]`, `column_longitudes = [-149, -150, -149]` generated
  the two distinct sites' daily files from the raw artifact in 74 s and returned a
  3-vector whose first and third entries are the same object; `get_setup_type` built the
  `ForcingFromFile` with the vector and `sites = LatLongPoint{Float32}[(17, -149),
  (17, -150), (17, -149)]`. (Under `BUILDKITE` every `era5_dataset` call uses a fresh
  temporary directory, so the setup's own call regenerated both files, as the
  single-column CI job has always done.)
- `multi_column_dev/multisite_case.jl` + `multisite_compare.jl` (full runs, CPU, the
  ERA5 EDMF config `prognostic_edmfx_tv_era5driven_column.yml` in its Float32 with
  `initial_condition: ForcingFromFile`, 200 levels, `deep_atmosphere: false`, 10 min =
  60 steps): file A is the real ERA5 site file above; file B is a copy of A that is 2 K
  warmer (`ta`, `ts`), 10 % moister (`hus`), and keeps every other node (two-hourly, 12
  nodes, so a different time axis and a shorter span). Three runs: single column on A,
  single column on B, and three columns at the equator (longitudes 0, 90, 0) reading
  `external_forcing_file: [A, B, A]`. **Result: the final state `Y` of column 1 and of
  column 3 is bitwise identical to the single column on A, and column 2 is bitwise
  identical to the single column on B** (10 state entries each, `rtol = atol = 0`);
  columns 1 and 3 are identical to each other; all 10 state entries of column 2 differ
  from column 1, so the second file was really used. Cache: the only differences are the
  known A9 uninitialized `ᶜmp_tendency` scratch and, for columns 1 and 3, the inputs'
  `range[2]` (the ragged input's covered range is the intersection over its segments,
  22 h with B present against A's 23 h), which is bookkeeping, not data. No NetCDF
  diagnostics were written in 10 min (the default ones are hourly), so that comparison
  was empty. Wall time: 446 s (first case, compilation), 2.3 s, 761 s (multi-column,
  compilation); the 60 steps themselves take about 2 s. A 3-hour rerun that crosses two
  nodes of A and one of B is recorded below.
- The same three cases for **3 hours** (`MULTISITE_T_END = "3hours"`, 1080 steps, 18
  radiation calls; the model crosses the file nodes at 1 h and 2 h of A and at 2 h of B,
  while B's columns sit inside a two-hour interval the whole time): **the final state
  `Y` of columns 1 and 3 is bitwise identical to the single column on A and column 2 to
  the single column on B**; columns 1 and 3 identical to each other; all 10 state
  entries of column 2 differ from column 1. Cache: as in the 10-minute run (A9 scratch
  and the `range[2]` bookkeeping only). Hourly NetCDF diagnostics (70 files per run,
  3-D fields and 2-D surface fields such as `hfls`, `pr`, `lwp`, `rlds`): **69 of 70 are
  bitwise identical per column**; the one that differs is `clt`, the McICA shortwave
  cloud cover, and `cltl` would be the same. Cause, read from the code and checked on
  the solver buffers: RRTMGP samples the McICA cloud mask with `Random.rand()` from the
  global RNG per cloudy layer per column (`RRTMGP/src/optics/cloud_optics.jl:237-245`,
  `MaxRandomOverlap`), and the cloud-cover diagnostics count the sampled cloudy
  subcolumns, so with three columns in one solver the stream is shared by the columns
  and the sampled cover differs from a one-column solver (column 1's shortwave cover did
  match, 0.4464 %, as the first draws of the stream; its longwave cover and columns 2
  and 3 did not; two co-located columns with bitwise identical states get different
  draws). The cloud fraction the sampling sees is at most 0.19 % (A) and 0.42 % (B) here
  and the state and every flux diagnostic were bitwise identical, so the sampled cover
  had no effect on the fluxes at Float32 in this case. Not a forcing or multi-column
  bug; a property of McICA that holds for any multi-column all-sky run and belongs in the
  PR's behaviour list (§10 item 6). The `radiation_reset_rng_seed` option reseeds the
  global RNG before each radiation call, which makes a run reproducible but does not
  make the columns' draws independent of the solver's column count. Wall time: 376 s
  (single A, compiling the hourly NetCDF path), 13.6 s (single B), 982 s (three columns;
  the comparison itself, walking the cache, takes several minutes on top).

### Still one-site-for-all (documented gaps)
- `GCM` (`cfsite_number` is one group; steady `InMemoryColumnData` profiles are
  broadcast) and `ARMVARANAL` (one `.cdf` file, converted once) take a single file; a list
  in `external_forcing_file` is not supported there.
- `FileHeatFluxes` returns one `HeatFluxes` per surface update, broadcast to every column
  (ARM only); per-column prescribed fluxes would need a `Field`-valued flux scheme.
- `TimeVaryingInsolation(; latitude, longitude)` with an explicit site (ARM) is
  broadcast (A4); without the override it uses the columns' own coordinates.
- The ragged input holds every column's series in memory (`|times| x nlevels x ncols x
  nvars`); nothing streams. The user's ClimaUtilities plan (`RAGGED_FORCING_PLAN.md`
  §3b) has the budget; with hourly ERA5 days this is kilobytes per column.
- `ReanalysisTimeVarying` in `multicolumn` generates one ERA5 file per distinct site from
  the raw artifact (`era5_dataset` per site); under `BUILDKITE` each goes to its own
  temporary directory, as before.

## 12. Per-site forcing implemented as in `MULTI_COL_FORCING.md` Part 3 (2026-09-16)

Eight commits on `kp/multi-col` (`ec4a5facb` .. `94c354886`), one per step of Part 3:
ragged inputs; vector of datasets; initial condition matched by site; config lists and
`get_grid(...; sites)`; steady GCM profiles per column; RRTMGP latitude guard; ARM
per-column fluxes (and the explicit insolation site in `FT`); docs. Deviations from the
plan: a dataset without a site inside a *list* is an error at `ForcingFromFile`
construction (pass `sites`); the YAML loader coerces a list for a scalar default entry by
entry (`coerce_to_default(::Type{T}, ::AbstractVector)`), which the plan had not foreseen;
`column_points` lives in `type_getters.jl`. Two fixes found by the runs were squashed into
their steps: the ARM conversion writes one directory per call (a file listed twice converts
to one path, so the same-site check, which compares paths, passes), and the scalar
`InMemoryColumnData` inputs go through the per-column host builder (the former
`_interp_column` broadcast an `Interpolations` object into a GPU kernel).

### Unit tests (CPU, `.buildkite`, Julia 1.12.5, isolated module)
- `test/column_datasets_tests.jl`: 8 testsets, 121 tests (One file per column 20, In-memory
  22, surface seams 13). `test/config/model_from_config.jl` 13, `test/grids.jl` 83.
- `<scratchpad>/inmemory_bitwise_check.jl`: the scalar `InMemoryColumnData` inputs, now
  built through the per-column host path, are bitwise the former in-place broadcast
  (`field .= itp.(ᶜz)`) for `ta`, `wa`, `tntha` of the real cfsite `site23` on Float32
  and Float64 60-level columns (0 differing values).
- `multi_column_dev/ragged_precision_check.jl`: unchanged from §11 (Float64: `ta` 2.16e-9 ..
  `wa` 1.44e-7 vs the Float64 reference, old path exact; Float32: both at 1e-7).

### Full runs (template: two single columns + one multi-column run)
All runs on the shipped Float32 configs with `multi_column_dev/short_forcing.yml` (10 min,
no state output) and `shallow.yml`; comparison at `rtol = atol = 0` on the final state `Y`
and cache `p` (`field_diffs`, `CACHE_IGNORE`); a 10-minute run writes no NetCDF (hourly).
Wall time is compilation: the ERA5 EDMF single column solves 10 minutes in 2 s once
compiled (435 s for the first, 741 s for the three-column run).

1. **ERA5 EDMF (`ForcingFromFile`), CPU, before the RRTMGP guard** (commits 1-5 loaded,
   `multi_column_dev/multisite_case.jl`): A = the real (17, -149) file, B = a warmer
   two-hourly copy of A at site (0, 90); the three columns sit at (17, -149), (0, 90),
   (17, -149), placed by the files. Column 2 (equator) == single B: `Y` 10/10 bitwise,
   `p` 178/180 (the two are the inputs' `range` bookkeeping). Columns 1 and 3 (17 N)
   differ from single A in every state entry and 70/180 cache entries, and are bitwise
   identical to each other. `nbad = 162`. This is the RRTMGP dry-air-amount latitude
   dependence of 2.1 row 5, measured; the guard of commit 6 removes it (next item).
2. **GCM (`prognostic_edmfx_gcmdriven_column.yml`), CPU** (`multi_column_dev/gcm_multisite_case.jl`):
   single columns on `site23` and `site17`, three columns on `cfsite_number: [site23,
   site17, site23]`, placed by the groups' `lat`/`lon` at (17, 211), (35, 235), (17, 211).
   Every column's state is bitwise its single column's (`Y` 10/10 for all three); cache
   106/108 each, the two being the uninitialized `ᶜmp_tendency.dq_tot_dt` /
   `e_tot_hlpr` scratch (A9); columns 1 and 3 identical, column 2 differs from column 1
   in all 10 state entries. `nbad = 6` (the A9 entries).
3. **ERA5 EDMF (`ForcingFromFile`) on the GPU** (one A100, Julia 1.11.4 / CUDA.jl 5.11.3,
   `multi_column_dev/gpu_multisite.jl`: the case of item 1 with the RRTMGP guard, the
   uncommitted surface-conditions kernel argument, and the GPU-only subsidence method
   override of `gpu_subsidence_override.jl`; shipped resolution, 200 levels). Column sites
   (17, -149), (0, 90), (17, -149). **Every column's state is bitwise its single
   column's on the GPU** (`Y` 10/10 for columns 1, 2, 3); columns 1 and 3 identical;
   column 2 differs from column 1 in all 10 entries. Cache: column 2 178/180 (the A9
   `ᶜmp_tendency` scratch), columns 1 and 3 166/180: the same two plus the twelve
   `range.2.counter` bookkeeping entries of the ragged inputs (the multi-column input's
   current interval is the intersection over the hourly A and two-hourly B axes, so it
   matches single B and not single A; no data). `nbad = 30`, all admissible.
4. **ERA5 EDMF (`ForcingFromFile`), CPU, after the RRTMGP guard** (all eight commits plus
   the uncommitted surface-conditions kernel argument, REPL restarted; same files and
   sites as item 1; `MULTISITE_TAG = "_after"`). **Every column's state is bitwise its
   single column's** (`Y` 10/10 for columns 1, 2, 3, against 10/10 *differing* for
   columns 1 and 3 before the guard); columns 1 and 3 identical; column 2 differs from
   column 1 in all 10 entries. Cache exactly as on the GPU (item 3): 178/180 for column 2,
   166/180 for columns 1 and 3 (A9 scratch and the twelve `range.2.counter` entries).
   `nbad = 30`, all admissible. Together with item 1 this is the before/after measurement
   behind commit `87e0e0617` (Keep RRTMGP's latitude at the equator for columns).
5. **ERA5 with site lists (`ReanalysisTimeVarying`), CPU** (`multi_column_dev/era5_lists_case.jl`,
   `ENV["BUILDKITE"] = "true"`, files generated from the raw artifact into temporary
   directories): single columns at (17, -149) and (17, -150), three columns with
   `site_latitude: [17, 17, 17]`, `site_longitude: [-149, -150, -149]`. `era5_datasets`
   returned a 3-vector whose first and third entries are the same object (two files
   generated). **Every column's state is bitwise its single column's** (`Y` 10/10 for all
   three); cache 178/180 each (A9 scratch only: both files are hourly, so the inputs'
   `range` bookkeeping matches too); columns 1 and 3 identical, column 2 differs from
   column 1 in all 10 entries. `nbad = 6`.
6. **GCM on the GPU** (`multi_column_dev/gpu_gcm.jl`, same environment and overrides as
   item 3; the scalar in-memory path now goes through the host builder, without which the
   single column itself fails: `_interp_column` broadcast an `Interpolations` object into
   a kernel, "not isbits", pre-existing on `main`). **Every column's state is bitwise its
   single column's on the GPU** (`Y` 10/10 for all three); cache 106/108 for columns 1
   and 3 and 108/108 for column 2 (A9 scratch); columns 1 and 3 identical, column 2
   differs from column 1 in all 10 entries. `nbad = 4`.
7. **ARM on the GPU: blocked before any multi-column code runs.** The ARM *single column*
   fails at initialization in a microphysics-cache broadcast (`microphysics_cache.jl:226`,
   from `set_precomputed_quantities!`): the kernel receives the `AtmosModel`, whose ARM
   flux scheme `MoninObukhov{..., FileHeatFluxes}` holds `Interpolations` interpolants
   ("`.shf_interp` ... `.coefs` is of type `Vector{Float64}` which is not isbits"). The
   same class of pre-existing gap as §8 (kernels taking `atmos` instead of the piece they
   read); not addressed here. `multi_column_dev/gpu_arm.jl` is ready for when it is.
8. **ARM (`prognostic_edmfx_armvaranal_column.yml`, 1M, prescribed fluxes), CPU**
   (`multi_column_dev/arm_multisite_case.jl`, `ENV["BUILDKITE"] = "true"` so the converted
   file goes to a temporary directory; `external_forcing_file: [file, file]`): the ARM
   single column against a two-column run at the SGP site, exercising the per-column
   `FileHeatFluxes` / `resolve_flux_scheme` `DataLayout` and the grid-coordinate
   `TimeVaryingInsolation` against the single column's explicit site. **Both columns are
   bitwise the single column** (`Y` 18/18; column 2 == single, column 1 == column 2),
   **all 63 NetCDF diagnostics** of the config's 10-minute output are bitwise per column,
   cache 155/159 per column (the four uninitialized 1M `ᶜmp_tendency.dq_{lcl,icl,rai,sno}_dt`
   scratch fields, A9). `nbad = 8`. (The REPL's 2000-line scrollback lost the column-1
   header lines; the column-2 and column-1-vs-column-2 results imply them.)
9. **ERA5 EDMF (`ForcingFromFile`), CPU, 3 hours** (`MULTISITE_T_END = "3hours"`, `TAG =
   "_3h"`; crosses the hourly nodes of A and the two-hourly nodes of B, writes hourly
   NetCDF). **Every column's state is bitwise its single column's** (`Y` 10/10 for all
   three), columns 1 and 3 identical, column 2 differs from column 1 in all 10 entries;
   NetCDF 69/70 per column, the exception `clt_1h_average.nc` (RRTMGP McICA cloud cover
   from the solver's shared random stream, 2.1 row 11); cache as in item 4 (A9 scratch and
   the `range` bookkeeping). `nbad = 30`, all admissible. A first comparison of these runs
   used a `COLUMN_MAP` left in `Main` by the ARM script and compared the wrong runs (3
   comparable entries, all differing); `multisite_compare.jl` now requires the case script
   to set the map.

### Summary

| Case | Device | State per column | NetCDF | Cache differences (all admissible) |
|---|---|---|---|---|
| ERA5 `ForcingFromFile` [A, B, A], 10 min, before guard | CPU | columns at 17 N differ (RRTMGP latitude) | none written | 70/180 |
| same, after guard | CPU | bitwise | none written | A9, `range` |
| same, 3 h | CPU | bitwise | 69/70 (`clt`) | A9, `range` |
| same, 10 min | GPU | bitwise | none written | A9, `range` |
| GCM [site23, site17, site23] | CPU | bitwise | none written | A9 |
| GCM | GPU | bitwise | none written | A9 |
| ARM [file, file] | CPU | bitwise | 63/63 | A9 (1M) |
| ARM | GPU | blocked (`atmos` into a microphysics kernel, pre-existing) | | |
| ERA5 site lists [(17,-149), (17,-150), (17,-149)] | CPU | bitwise | none written | A9 |

Not run: the 3-hour variant on the GPU, GPU runs at the shipped resolution without the
subsidence override (impossible with the current ClimaCore worktree), NetCDF for the GPU
runs (10-minute runs write none).


### GPU environment
- Julia 1.12.5 / CUDA.jl 5.11.3 (`.buildkite` manifest of 2026-09-16) still fails for the
  single column with `InvalidIRError: compiling eager_copyto_stencil_kernel!(...,
  FiniteDifferenceSpace)`: "unsupported call to an unknown function (call to
  gpu_gc_pool_alloc)" and "unsupported dynamic function invocation (call to
  calc_level_val ... operators_fd_eager.jl:647)" (§8, §9; the CUDA.jl downgrade from 6.2.2
  did not change it). GPU runs below use Julia 1.11.4 (`climacommon/2025_03_18`) with a
  scratch environment built from `.buildkite/Manifest-v1.11.toml` and the four worktrees
  developed (`<scratchpad>/env111c`; the Sep-10 `env111` no longer resolves: ClimaCore
  `multi-col-extras` needs UnrolledUtilities 0.1.11, `unrolled_insert`).
- File-forced GPU runs need the uncommitted `surface_conditions.jl` change (the kernel
  receives `atmos.microphysics_model`, not the `AtmosModel`, §8), applied in the working
  tree for these runs only.
- Under Julia 1.11.4 the ERA5 single column then fails in the *forcing* tendency:
  `apply_subsidence_forcing!` (`external_forcing.jl:229`, unchanged from `main`),
  `@. ᶠls_subsidence³ = ᶠinterp(ᶜls_subsidence * CT3(unit_basis_vector_data(CT3, ᶜlg)))`,
  `InvalidIRError ... eager_copyto_stencil_kernel!`, "unsupported dynamic function
  invocation (call to CuDynamicSharedArray)" from `calc_level_val`
  (`operators_fd_eager.jl:334`). Reproduced stand-alone on a 200-level `ColumnGrid` with a
  zero Float32 field (`<scratchpad>/gpu_stencil_probe.jl`): the same stencil on a
  materialized `Contravariant3Vector` field compiles, the fused product inside `ᶠinterp`
  does not. Independent of this PR (the same ClimaAtmos code ran on the GPU on 2026-09-10
  with the ClimaCore worktree of that day; the worktree has since been rebased onto main,
  0.16.2, UnrolledUtilities 0.1.11). Workaround used for the GPU runs: `z_elem: 256`
  (`multi_column_dev/lazy_gpu.yml`), since ClimaCore takes the lazy stencil kernel above
  256 face levels (`operators_finite_difference.jl:35`), so no ClimaAtmos code changes.
