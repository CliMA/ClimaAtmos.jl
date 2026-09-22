# Multi-column runs identical to single-column runs: forcing per column and the coordinate problem

Two things stand between the current `kp/multi-col` branch and "a multi-column run is N
single-column runs": every column must be able to read its own forcing data, and the
multi-column space carries `LatLongZPoint` coordinates while a single column has `ZPoint`s.
Part 1 is the plan for the forcing; Part 2 is the inventory of everything that reads the
coordinates and the design options. Facts below were read from `HEAD` (`1be4b99cd`), the
ClimaUtilities worktree `~/worktree/ClimaUtilities.jl/ragged-data-intp` (14 commits on
v0.1.32), ClimaCore `~/worktree/ClimaCore.jl/multi-col-extras`, and ClimaDiagnostics
`~/worktree/ClimaDiagnostics.jl/multi-cols`, and from runs on 2026-09-15/16 (CPU, Julia
1.12.5, `.buildkite` environment).

## Part 1. Forcing per column

### 1.1 What ClimaUtilities `kp/ragged-data-intp` provides

- `FileReaders.DataSource(paths, varname; time_transform, coord_names)`: a description of
  one NetCDF variable (paths, name, dates, time index, coordinate names). `==`/`hash` on
  the stored fields, so two sources of the same file compare equal.
- `TimeVaryingInput(sources::Vector{DataSource}, space; start_date, method,
  preprocess_func)`: one source per column of `space`, in `field2array` column order (the
  order of the points given to `MultiPointGrid`). Each source is read whole, regridded
  onto the levels of `space` at construction (`Utils.interpolate_columns!`, linear in `z`,
  constant beyond the file's levels), and the series are stored contiguously with offsets
  (`RaggedInterpolatingTimeVaryingInput`), so columns may have different time axes and
  lengths. Columns whose sources compare equal share one segment. `evaluate!(dest, itp,
  t)` does a per-column binary search in time and the linear blend in one broadcast (CPU
  and GPU). `TimeVaryingInput(source::DataSource, space)` is the shared-file shorthand;
  `TimeVaryingInput([[ta...], [hus...]], space; compose_function)` composes variables.
- `SpaceVaryingInput(sources, space)` for static per-column profiles.
- Accepted spaces: `PointSpace`, `FiniteDifferenceSpace`, `MultiPointSpace`,
  `MultiColumnFiniteDifferenceSpace`, so the single column and the multi-column space go
  through the same constructor.
- Methods: `LinearInterpolation()` (`Throw` outside the range, checked on the host against
  the intersection of the columns' ranges), `LinearInterpolation(PeriodicCalendar())`
  (each column repeats its own record; a warning when the periods differ),
  `NearestNeighbor`, `Flat`. `LinearPeriodFillingInterpolation` and
  `PeriodicCalendar(period, date)` are construction errors.
- Times: `DateTime`s become `ITime`s in milliseconds with `epoch = start_date`;
  `evaluate!` accepts the model's `ITime` (Atmos always integrates in `ITime` with the
  start date as epoch, `AtmosSimulations.jl:198-202`) and plain seconds.

What it does not do: stream (everything is in memory, `|times| x nlevels x ncols x nvars`
floats; kilobytes per column for hourly ERA5 days), or vary the vertical grid in time.
Different files per column are fine, from different places and with different time axes
and level counts, as long as each variable is a NetCDF `(z, time)` or `(time,)` variable
with `z` in metres (the ClimaColumn schema; an ERA5 file next to a converted ARM file
works). What cannot go into one input is a file-backed column next to a steady in-memory
(GCM) column, because the in-memory kind is not a `DataSource`.

One numerical detail, found by comparison against the old path
(`multi_column_dev/ragged_precision_check.jl`): on a Float64 column the new path is
2e-9 (`ta`) to 1.4e-7 (`wa`) off the Float64 reference, i.e. one Float32 rounding,
because `Utils.linear_interpolation` forms `(y1 - y0) / (x1 - x0)` in the file's Float32
before promoting; the old `InterpolationsRegridder` converted the data to Float64 first.
Promoting `block` and `z_src` to `eltype(model_z)` in `_regrid_block`
(`ext/MultiSiteInputsExt.jl`) makes it exact again. On a Float32 column both paths are at
1e-7, on equal footing. Surface series are bitwise identical to the old 0-D inputs.

### 1.2 How the single-column case is wired today (`HEAD`)

`get_setup_type(parsed_args, thermo_params)` (`src/config/type_getters.jl:182`) turns the
config into one dataset handle and hands it to `Setups.ForcingFromFile(dataset,
start_date)`:

| `initial_condition` | dataset | site comes from |
|---|---|---|
| `ReanalysisTimeVarying` (+ `external_forcing: ReanalysisMonthlyAveragedDiurnal`) | `era5_dataset(parsed_args, FT)` → `ColumnDataset` of the generated daily (monthly) ERA5 file | `site_latitude`, `site_longitude`, rounded to 0.25°, written into the file's attributes |
| `ForcingFromFile` | `ColumnDataset(external_forcing_file)` | the file (attributes only range-checked) |
| `ARMVARANAL` | `ColumnDataset(to_climacolumn(external_forcing_file))` | the file, read by `site_location` for `TimeVaryingInsolation(; latitude, longitude)` |
| `GCM` | `read_cfsite(external_forcing_file, cfsite_number)` → steady `InMemoryColumnData` | not read (`site_location = nothing`; the group has `lat`/`lon` scalars) |

`ForcingFromFile` (`src/setups/ForcingFromFile.jl:54`) uses the handle twice:
`ExternalDrivenTVForcing(dataset; forcing)` becomes the forcing, and
`read_initial_profiles(dataset, start_date)` (profiles at the file time closest to
`start_date`) becomes `ColumnProfiles` for the initial condition. **Initial condition and
forcing are paired by being read from the same object; no coordinate is compared.** The
forcing is consumed in `external_forcing_cache` (`column_timevaryinginputs(cd, vars,
axes(Y.c), start_date)` per term, `surface_timevaryinginputs(cd, (:ts, :coszen, :rsdt),
surface space, start_date)`), the initial condition pointwise in
`center_initial_condition(setup, local_geometry, params) = column_profiles_ic(setup.profiles,
local_geometry)`, which reads only `local_geometry.coordinates.z`.

Today `column_timevaryinginputs` builds a streamed `InterpolationsRegridder` input per
variable and `surface_timevaryinginputs` an in-memory 0-D input that broadcasts one scalar
to every column; on a multi-column space the former needs ClimaUtilities #251
(`horizontally_uniform`) to apply one profile to every column.

### 1.3 Design

Decisions taken with the user (2026-09-16): one dataset per column, built from vectors in
the config, the order of the vector being the order of the columns; all four dataset kinds
support it; the initial condition of column `i` comes from dataset `i`.

**Data model.** `ColumnDatasets.ColumnData = Union{AbstractColumnData,
AbstractVector{<:AbstractColumnData}}`. A scalar is read by every column (the shared
segment); a vector is one dataset per column, typed to one kind (`Vector{ColumnDataset}`
or `Vector{InMemoryColumnData}`; mixing kinds cannot be expressed as one input).
`ExternalDrivenTVForcing{CD <: ColumnData}` and `ForcingFromFile{CD <: ColumnData}`.
Methods for the vector, each one line: `require_forcing_variables` (every dataset),
`time_interpolation_method` (`only(unique(...))`), `file_time_span` (minimum: the run
must end before the shortest file), `read_initial_profiles` (map), `site_location`
(`(; latitude = [...], longitude = [...])`, a NamedTuple of vectors so `(; latitude,
longitude) = site_location(data)` works for both), `surface_vars` (intersection).

**File-backed inputs (ERA5, `ForcingFromFile`, ARM).** `column_timevaryinginputs(data,
names, space, start_date; method)` builds `TimeVaryingInput(data_source(data, name),
space; start_date, method, preprocess_func = preprocess(format, name))` with
`data_source(cd) = DataSource(cd.path, format_variable_name(cd.format, name))` and
`data_source(::Vector) = map(...)`. `surface_timevaryinginputs` is the same call on the
surface space (`axes(Fields.level(Y.f.u₃, half))`): a `(time,)` variable becomes a
one-level segment, one value per column. This is a drop-in replacement of
the streamed `InterpolationsRegridder` input: same `TimeVaryingInput`/`evaluate!`
interface, so `external_forcing_cache` and the forcing terms do not change; what changes
is that the file is read once into memory instead of two time slices at a time, and that
the vertical regridding happens once at construction. It was implemented and verified on
2026-09-15 (Section 1.5: bitwise state per column in full runs; profiles within one
Float32 rounding of the old path on Float64 columns, surface series bitwise), then
reverted with the rest. The `InterpolationsRegridder` path, the `horizontally_uniform`
flag, the 0-D surface inputs, and the format hook `extrapolation_bc` go away
(`read_surface_series` stays for `FileHeatFluxes`).

**Steady in-memory inputs (GCM).** `column_timevaryinginputs(::Vector{InMemoryColumnData},
...)` fills the field column by column on the host (`field2array(field)[:, c] .=
itp_c.(z[:, c])`, then `copyto!`) and wraps it in `TimeVaryingInput(Returns(field))`;
`surface_timevaryinginputs` fills a surface field with one constant per column. `read_cfsite`
reads the group's `lat`/`lon` into `site_location`.

**Initial condition by column index.** `center_initial_condition` is pointwise and sees
only the local geometry, so with a vector of profiles it needs the column. Add to
`Setups`: `column_indices(space::ColumnSpace)`, an `Int` field with `field2array(indices) .=
(1:ncolumns)'` (`nothing` on other spaces); `initial_condition_field(f, space)` broadcasts
`f.(local_geometry, columns)`; `initial_state` uses `center_ic(lg, column)`; a generic
`center_initial_condition(setup, local_geometry, column, params) =
center_initial_condition(setup, local_geometry, params)` so every other setup is untouched;
`ForcingFromFile` extends the four-argument method with `setup.profiles[column]`. The one
other caller of `initial_condition_field` (`constrain_state.jl:151-157`, Shipway-Hill
closures) gains an ignored second argument. Matching by coordinates is not an option: the
columns may share coordinates (Part 2).

**Config surface.** A dataset option may be a list with one entry per column (checked
against the number of columns) or a scalar for every column:

| option | case | vector meaning |
|---|---|---|
| `site_latitude`, `site_longitude` | `ReanalysisTimeVarying`, `ReanalysisMonthlyAveragedDiurnal` | one ERA5 site per column; `era5_datasets` generates or locates one file per distinct site (`Dict` by site, so duplicates share) |
| `external_forcing_file` | `ForcingFromFile`, `ARMVARANAL` | one file per column (ARM: each converted with `to_climacolumn`) |
| `cfsite_number` | `GCM` | one group per column of the one `external_forcing_file` |

One helper does the dispatch: `per_column(f, parsed_args, option)` applies `f` to each entry
of a list after checking its length, or to the scalar. It must not touch
`parsed_args["config"]` for scalars (`get_setup_type` is called in tests with minimal
`Dict`s). Both ERA5 vectors must be lists of the same length.

**ARM surface pieces.** `FileHeatFluxes(::Vector{ColumnDataset}, start_date)` keeps one
interpolant pair per column and returns `HeatFluxes(shf::Vector, lhf::Vector)`;
`resolve_flux_scheme(p, t, FT, surface_space)` turns vector fluxes into a
`DataLayout` of `MoninObukhov` parameterizations (a surface `Field` per component, then
`Fields.field_values(@. MoninObukhov(z0m, z0b, HeatFluxes(shf, lhf), ustar))`), the same
wrapping `boundary_overrides_wrapper` does for per-cell overrides; scalar fluxes are
unchanged. `TimeVaryingInsolation(; latitude::Vector, longitude::Vector)`:
`insolation_cache` adds surface fields of the two vectors and `set_insolation_variables!`
broadcasts `Insolation.insolation` over them. Alternatively (Part 2, option C), place the
columns at their sites and use the grid coordinates, which needs no vectors.

**Not needed / out of scope.** Streaming (files are small); `column_latitudes` /
`column_longitudes` as sites (Part 2); a coordinate check between files and columns
(the ClimaUtilities design decision Q6-Q8: pairing is positional).

### 1.4 Steps, each independently reviewable

1. `ColumnDatasets`: `DataSource`-based `column_timevaryinginputs` /
   `surface_timevaryinginputs` for `ColumnDataset` (single and vector), `ColumnData`, the
   vector methods, `_format`, `data_source`; drop `extrapolation_bc`. Tests: the existing
   surface-series test must evaluate on the surface space (a per-column input cannot be
   broadcast into a 3-D field like the old 0-D one); a new "one file per column" test
   with two files of different values and time axes on a three-column grid.
2. `Setups`: `column_indices`, the column argument in `initial_condition_field` /
   `initial_state`, the generic four-argument `center_initial_condition`;
   `ForcingFromFile` vector profiles indexed by column; `constrain_state.jl` closures.
   Test: per-column initial condition through `initial_condition_field`.
3. `InMemoryColumnData` per column and `read_cfsite` site location. Test: two cfsite
   groups (`write_test_cfsite_file` needs a `mode = "a"` to add a second group, and
   different `nt` gives different time means).
4. Config: `per_column`, the four `get_setup_type` branches, `era5_datasets` with site
   vectors, `warn_if_run_exceeds_forcing` unchanged (minimum span). Tests: a YAML list
   survives `AtmosConfig` (checked 2026-09-15: `Vector{String}`), wrong lengths error.
5. ARM: `FileHeatFluxes` vector, `resolve_flux_scheme` with the surface space,
   `TimeVaryingInsolation` vectors (or option C of Part 2 instead).
6. Docs and help texts (`default_config.yml`, `configuration.md`,
   `column_datasets_reference.md`, `api.md`: `ColumnData` in, `extrapolation_bc` out),
   NEWS (user).
7. Dependencies: Atmos then needs a release of `kp/ragged-data-intp`; released 0.1.32 has
   no `DataSource`, so every file-forced CI job fails at construction until then. #251 is
   no longer needed.

### 1.5 Verification, done and to redo

Done on 2026-09-15 with the reverted implementation (scripts in `multi_column_dev/`,
gitignored; details in `MULTI_COLUMN_CHECKLIST.md` §11):

- `test/column_datasets_tests.jl` 100/100 including the new testset.
- Real ERA5 ClimaColumn file, 200-level column, 139 times: surface series bitwise equal to
  the old path; profiles at the Float32 rounding of the file data (1.1); a shared file on
  three columns bitwise equal to the single column, one segment in memory.
- Full runs, ERA5 EDMF config (Float32, `ForcingFromFile`), three columns at the equator
  reading files [A, B, A] with B a warmer two-hourly copy of A, against single columns on A
  and B: after 10 minutes and after 3 hours (crossing file nodes) the state of every
  column is bitwise identical to its own single-column run; 69 of 70 hourly NetCDF
  diagnostics bitwise per column; the exception `clt` is RRTMGP's McICA cloud cover,
  sampled from the global random stream shared by the solver's columns (Part 2, 2.1).
- `era5_datasets` for two sites from the raw artifact: 74 s, shared object for the
  repeated site.

To redo after re-implementation: the same three checks, plus GCM with
`cfsite_number: [site23, site17, site23]` against single columns on each group, and ARM
with `[file, file]` for the vector surface path.

### 1.6 Risks and open questions

- The Float32 rounding in `linear_interpolation` (1.1): fix in ClimaUtilities or accept
  1e-7 changes in the Float64 file-forced references.
- `AtmosModel` holds the dataset(s); a vector is no more kernel-friendly than the single
  `ColumnDataset` that already breaks file-forced GPU runs (pre-existing, checklist §8).
- The run is bounded by the shortest file; `PeriodicCalendar` requires equal periods.
- Column order is a data-layout fact (`field2array` order = point order); ClimaUtilities
  keeps the mapping in `column_segment` so a layout change would not silently reorder.
- `cfsite` groups with different `lev` counts are fine (regridded at construction);
  different calendars are not mixed within one run (one `initial_condition` kind).

## Part 2. `LatLongZPoint` columns versus `ZPoint` columns

### 2.1 Where the coordinates are read (`HEAD`, `src/`)

"Single column" below is `config: column`: a `FiniteDifferenceSpace` with `ZPoint`
coordinates, which takes every flat-space branch. "Reached" says whether the file-driven
cases (ERA5, `ForcingFromFile`, ARM, GCM) execute the reader.

| # | Reader | What the sphere branch does | What the single column does | Guarded today | Reached by file-driven cases | Same at lat = 0? |
|---|---|---|---|---|---|---|
| 1 | Coriolis, `cache.jl:318-338` | `2Ω sin(lat)` (deep: also `f¹²`) | f-plane `f_plane_coriolis_frequency` (default 0) | yes, `iscolumn` | yes | n/a (guard) |
| 2 | `IdealizedInsolation`, `callbacks.jl:200-215` | `(1 + 0.3(1 - 3 sin²lat))/2` | same formula with lat = 0 | no | no (`ExternalTVInsolation`) | yes |
| 3 | `TimeVaryingInsolation` without explicit site, `callbacks.jl:224-272` | `Insolation.insolation(date, lat, long)` per column | `insolation(date, 0, 0)` | no | no (ARM passes its site) | yes |
| 4 | Gray radiation optical thickness, `radiation.jl:153-161` | `7.2 + (1.8 - 7.2) sin²lat` | lat = 0 | no | no (`allskywithclear`) | yes |
| 5 | Latitude passed to the all-sky RRTMGP solver, `radiation.jl:198-247` → RRTMGP `as.lat`, used in the column dry-air amount (`Optics.jl:138`, gravity correction) | per-column latitude | zeros | no | **yes** (`allsky*`) | yes |
| 6 | Default surface temperature `zonally_symmetric_temperature`, `Setups.jl:203-216`, and the slab-ocean initial `T`, `prognostic_variables.jl:318-330` | `271 + 29 exp(-lat²/1352) - 6.5e-3 z` | `300` | no (dispatch on point type) | no (`ExternalTemperature`); LARCFORM1's slab ocean: yes | yes (z = 0 at the surface) |
| 7 | Slab-ocean Q-flux, `surface_temp.jl:117` | `cos`/`sin` of lat | not applicable | `q_flux = false` default | no | yes |
| 8 | NOGW / OGW / Held-Suarez / analytic topography | latitude | not reached | – | no | – |
| 9 | Setups dispatching on the point type: `DecayingProfile._temperature_perturbation`, `RCEMIPIIProfile` | `LatLongZPoint` and `XYZPoint`/`XZPoint` methods | `ZPoint` method or none | – | no (analytic cases) | case by case |
| 10 | NetCDF `lat`/`lon` per column (ClimaDiagnostics `netcdf_writer_coordinates.jl:746-800, 860-870`, `netcdf_writer.jl:395-399`) | writes the grid coordinates | no horizontal coordinates written | – | yes (metadata) | writes 0, 0 |
| 11 | RRTMGP McICA cloud mask `Random.rand()` (`RRTMGP/src/optics/cloud_optics.jl:237-245`) | one global random stream shared by the solver's columns | one column consumes the stream | – | yes (`clt`, `cltl` diagnostics; fluxes were unaffected in the test) | independent of coordinates; differs by column count |

Row 5 is the one that matters for "identical": with true site latitudes, all-sky RRTMGP
radiation of a column at 17°N is not the radiation of the Cartesian single column, which
RRTMGP sees at latitude 0. The "same at lat = 0" column is why the (0, 0) placement
reproduces the single column without any conditional: every flat-space branch in Atmos is,
by convention, the equator (`# flat space is on Equator`, `callbacks.jl:206`,
`radiation.jl:156, 201`), and that was confirmed bitwise for every column configuration on
the unit-metric grid (checklist §7, §9).

### 2.2 The three options, evaluated against "identical to the single column"

**A. `XYZPoint` columns** (Cartesian `MultiPointGrid`). Tried on 2026-09-16 without
changing ClimaCore (`multi_column_dev/cartesian_columns.jl`): the `MultiPointGrid` struct
is generic in its geometry, only its constructor demands `LatLongPoint`s. The extruded
grid gets `CartesianGlobalGeometry`, the space is a `MultiColumnFiniteDifferenceSpace` with
`XYZPoint` coordinates, `level`/`column`/`field2array`/`bycolumn`/`iscolumn` work, and a
hydrostatic-balance run built its state, cache, and integrator completely. It failed only
in the ClimaDiagnostics writer (`target_coordinates` reads `coords.lat`). Identity: rows
2-7 take the flat branch exactly as the single column, by type, not by value, so no
equator coincidence is involved; row 9 differs where a setup has an `XYZPoint` method but
the single column uses the `ZPoint` fallback (analytic cases only). Cost: a generic
constructor and `Meshes.domain` in ClimaCore (~20 lines), the writer generalized to
whatever horizontal components exist or to none (~20-30 lines, four sites), Atmos
`MultiColumnGrid`/`get_grid` taking a column count. The written `x`/`y` would be zeros,
which is the "wrong diagnostics" objection; writing no horizontal coordinate at all (a
column index only, as the single column has no horizontal coordinate either) avoids it.

**B. `LatLongZPoint` at (0, 0).** Identical to the single column by the equator
convention (2.1), no conditional needed beyond the existing Coriolis guard (which is only
needed for a nonzero f-plane frequency; at the equator `2Ω sin 0 = 0` anyway). Costs: the
output `lat`/`lon` are 0 for a column whose data is from 17°N, and per-site
`TimeVaryingInsolation` needs explicit vectors because the grid cannot supply the site.
The identity rests on a convention that a future latitude reader could break, but such a
reader would break single-column-versus-sphere consistency in the same way, so this is not
a new class of risk.

**C. `LatLongZPoint` at the true sites.** Rows 2-5 differ from the single column unless
guarded; row 5 is reached by every all-sky file-driven case. Guards would go into
`callbacks.jl` (two sites), `radiation.jl` (two sites), `Setups.jl`/`prognostic_variables.jl`
(two sites), `surface_temp.jl` (one), i.e. seven more copies of the Coriolis conditional,
or one accessor `column_latitude(space)` returning the equator on `ColumnSpace` that all
seven call. That accessor also changes what the sphere code looks like at every reader.
And the guard makes the true coordinates physically inert, so the coordinates would exist
only for the NetCDF metadata.

### 2.3 A design that keeps the identity and the metadata: geometry at the equator, sites as labels

The coordinates serve two roles that the options above conflate: the geometry the physics
reads, and the site the data came from. Separate them.

- **Geometry**: every column at (0, 0) on the unit-metric `MultiPointGrid` (option B), or
  the Cartesian grid of option A. Physics is identical to the single column with no new
  conditional; both were verified.
- **Site metadata**: the datasets know their sites (`site_location`); write them into the
  output as the per-column coordinate variables. ClimaDiagnostics' `NetCDFWriter` already
  has the hook: the `horizontal_pts` keyword, which for a `MultiPointSpace` currently
  errors ("horizontal_pts is not supported for spaces of multiple columns",
  `netcdf_writer.jl:211`). Letting it *label* the columns instead (use the given points for
  the `lat`/`lon` variables when there is one per column; the data are still written as
  is) is a few lines in the user's PR #188 branch. Atmos passes
  `horizontal_pts = LatLongPoint.(site_location(datasets)...)` where it builds the writers
  (`AtmosSimulations.jl:91, 155`), falling back to the grid coordinates when the setup has
  no sites. `column_latitudes`/`column_longitudes` then only matter for analytic cases and
  default to zeros.

Result: no conditional anywhere, the state is bitwise the single column's, and the NetCDF
files carry the true sites. `TimeVaryingInsolation` per site still needs the explicit
vectors (Part 1, ARM), since the grid deliberately does not know the sites.

### 2.4 The design that removes the question: `ZPoint` columns in ClimaCore

The cleanest statement of "N single columns" is a multi-column space whose coordinates are
`ZPoint`, exactly as the single column's. ClimaCore cannot express it today: an extruded
space's coordinates are `product_coordinates(horizontal point, ZPoint)`
(`Geometry/coordinates.jl:120-132`), defined for `XPoint`, `XYPoint`, `LatLongPoint`, and
there is no zero-dimensional horizontal point. Adding one (a degenerate point type with
`coordinate_axis = ()`, its `LocalGeometryType`, an empty horizontal metric in
`product_geometry`, `MultiPointGrid` built from it) would make every Atmos reader in 2.1
take the `ZPoint` path by type, including row 9, and the diagnostics would write no
horizontal coordinate, as for the single column. It is the right end state but a ClimaCore
`Geometry` change with GPU and operator implications, not something to do inside this PR.
Option A with an index-only coordinate output is the closest approximation available now.

### 2.5 The grid follows the data (true sites, minimal guards)

The variant the user finds reasonable: keep `LatLongZPoint`, but take the column
positions from the datasets instead of from `column_latitudes`/`column_longitudes`, so the
grid and the output carry the true sites and nothing is duplicated in the config.

- **Where the points come from.** `get_simulation` builds the setup before the grid, so
  `get_grid(parsed_args, params, context; points)` can take the points from
  `site_location(setup.dataset)`: ClimaColumn files carry `site_latitude`/`site_longitude`
  (ERA5 writes the 0.25° grid point, which is where the data is); `read_cfsite` would read
  the group's `lat`/`lon` scalars (today `site_location = nothing`); ARM has them after
  conversion. `column_latitudes`/`column_longitudes` remain for analytic setups (default
  zeros) and are ignored, or must agree, when the datasets provide sites.
- **Pairing** stays positional: dataset `i` gives point `i`, forcing `i`, and initial
  condition `i`, so the column index is still what the initial condition needs, and two
  datasets at the same site are allowed.
- **What must be guarded for the file-driven cases to stay identical to the single
  column**: only row 5 of 2.1, the latitude passed to the all-sky RRTMGP solver (the dry-air
  amount's gravity correction); the guard is one line in `rrtmgp_solver_kwargs`, next to
  the existing `# flat space is on Equator` fallback, using `iscolumn` exactly as Coriolis
  does. Rows 2-4 and 6-7 are not reached by these cases. Alternatively, do not guard row 5
  and accept that all-sky radiation of a column at its true latitude differs from the
  Cartesian single column's, which would then have to be documented as intended physics
  rather than a bug; identity checks against the single column would need `rad: gray` or
  no radiation.
- **Analytic setups** on this grid see their latitude in rows 2-4, 6-7 whenever they are
  placed away from the equator; identity with the single column then requires the equator
  placement or the seven guards of option C. Since analytic cases have no site, the
  zeros default gives identity for free.
- **Per-site `TimeVaryingInsolation`** comes for free from the grid coordinates
  (`TimeVaryingInsolation(; start_date)` without an explicit site).

Compared with 2.3 this needs one physics guard (row 5) and a `get_grid` keyword instead
of the writer label, and the coordinates are real rather than labels.

### 2.6 Other potential solutions

- **Run every single column as a one-point multi-column grid.** If `config: column` were
  built as `MultiColumnGrid` with one point at (0, 0), single and multi-column runs would
  share one code path by construction and could not drift apart; every latitude reader
  would affect both identically. The cost is a behaviour change for single columns
  (`ZPoint` to `LatLongZPoint` at the equator, identical values by 2.1, but the NetCDF
  output gains `lat`/`lon` and a `column` dimension of length one) and the loss of the
  `FiniteDifferenceSpace` fast paths. Worth considering once the multi-column space is
  the established way to run columns; too large a change for this PR.
- **Sites as global attributes.** Instead of coordinate variables, Atmos writes
  `site_latitudes`/`site_longitudes` through the writer's existing `global_attribs`
  keyword. No ClimaDiagnostics change at all; the `lat`/`lon` variables stay at the grid's
  values (zeros in option B). Honest, but readers of the files must know to look at the
  attributes.
- **One accessor for the latitude the physics sees.** `column_latitude(space)` (or a
  `horizontal_position(space)` trait) returning the equator on a `ColumnSpace`, called by
  every reader in 2.1 instead of the inline `eltype(...) <: LatLongZPoint` tests. This
  centralizes option C's seven conditionals into one definition, and incidentally removes
  the seven inline branches that exist today, but it still touches every reader once.
- **`ZPoint` columns in ClimaCore** (2.4): the end state; a `Geometry` change outside this
  PR.

### 2.7 Recommendation

Two defensible choices, both verified to keep the state bitwise identical to the single
column for the file-driven cases:

- **2.3** (geometry at the equator, sites as writer labels): zero physics conditionals,
  true sites in the output, `TimeVaryingInsolation` needs explicit vectors.
- **2.5** (grid follows the data, the user's preference): true coordinates, one guard in
  `rrtmgp_solver_kwargs` (row 5) beside the existing Coriolis guard, and the rule that
  analytic setups stay at the equator; per-site insolation from the grid for free.

Either way, C in its full form (true sites plus seven guards) is not needed, and 2.4 is the
ClimaCore follow-up that would make the question disappear.

## Part 3. Implementation plan for 2.5 (decided 2026-09-16)

Written for an implementer who has not followed the discussion. Facts and line numbers
refer to `HEAD` = `1be4b99cd` on `kp/multi-col`. Ground rules from the user: minimal code,
fewer lines over more, one atomic commit per step (the user commits; prepare the changes
and report), format with the pinned JuliaFormatter (`.dev/format`, version 2.10.1), run
Julia only inside a tmux REPL started with `--history-file=no` on the `.buildkite`
environment (Julia 1.12.5; it develops ClimaCore `multi-col-extras`, ClimaUtilities
`ragged-data-intp`, ClimaDiagnostics `multi-cols`), and never attribute a numerical
difference to rounding without a reproducing experiment.

### 3.1 Decisions (user, 2026-09-16)

1. A file-driven `multicolumn` run has as many columns as its dataset list has entries.
   The datasets' sites place the columns and override `column_latitudes`/
   `column_longitudes`. A single shared dataset places every column at its one site (the
   "same site everywhere" run); the configured coordinates remain the placement for
   analytic setups and for datasets without a site.
2. ERA5 columns sit at the rounded 0.25° site written in the file's attributes.
3. A dataset without a site is not an error: its column falls back to the configured
   coordinates (3.2). ClimaUtilities cannot catch this, it never sees the placement;
   ClimaColumn files always carry `site_latitude`/`site_longitude` (`validate` requires
   them), cfsite groups carry `lat`/`lon`, only a runscript-built `InMemoryColumnData`
   may lack one. *Implemented (2026-09-16) for a scalar dataset only:* in a list, a
   dataset without a site is an error at `ForcingFromFile` construction ("records no
   site; pass `sites`"), because the initial condition of a list is matched by
   coordinates and the setup does not know the configured coordinates; a runscript
   passes the `sites` keyword instead.
4. The RRTMGP latitude guard applies to every `ColumnSpace`, single column included.
5. The initial condition of a column is paired with its dataset by matching the point's
   coordinates against the datasets' sites (3.5), not by column index.
6. Initial profiles keep `main`'s rule: the file time closest to `start_date`.
7. The Float32 rounding in ClimaUtilities `_regrid_block` (1.1) is deferred; Atmos accepts
   the resulting 1e-7 change of Float64 file-forced results.
8. Config names are unchanged: `site_latitude`/`site_longitude`, `external_forcing_file`,
   `cfsite_number` accept lists.
9. ARM per-column fluxes are implemented (3.10).

### 3.2 Semantics in one place

- **Scalar option** (a single path, site, or group) on a `multicolumn` grid: one dataset
  read by every column, as today, and the column count comes from the configured
  coordinate lists as today; every column is placed at the dataset's site (the same site
  repeated), so the "same site everywhere" run carries the true site in its output. A
  scalar dataset without a site keeps the configured coordinates. With the Coriolis and
  RRTMGP guards the placement has no physical effect for the file-driven cases.
- **List option**: one dataset per column, in list order. All list-valued options of a
  case must have the same length (`site_latitude` and `site_longitude`). The grid is built
  with one point per dataset at that dataset's site; a dataset without a site takes the
  configured coordinate of that column (`column_latitudes[i]`, or the single configured
  value repeated when the lists have length one).
- **Pairing**: forcing is positional (source `i` feeds column `i`, ClimaUtilities pairs
  by `field2array` column order, which is the order of the grid points); the initial
  condition is matched by coordinates (3.5). Two entries at one site must read the same
  file; equal entries share one segment in memory.
- **Time coverage**: the run must end before the shortest file (`file_time_span` is the
  minimum); `warn_if_run_exceeds_forcing` is unchanged.

### 3.3 Commit 1: ragged inputs for one dataset (`src/column_datasets/ColumnDatasets.jl`)

Replace the body of `column_timevaryinginputs(cd::ColumnDataset, names, target_space,
start_date; method = time_interpolation_method(cd.format))` (currently lines 495-525,
which build `TimeVaryingInput(cd.path, varname, target_space; start_date,
regridder_kwargs = (; extrapolation_bc, horizontally_uniform), file_reader_kwargs, method)`)
with one call per variable:

```julia
TimeVaryingInput(
    data_source(cd, name), target_space;
    start_date, method, preprocess_func = preprocess(cd.format, name),
)
```

with `data_source(cd::ColumnDataset, name::Symbol) = DataSource(cd.path,
format_variable_name(cd.format, name))` and `import ClimaUtilities.FileReaders: DataSource`.
`surface_timevaryinginputs(cd::ColumnDataset, ...)` (lines 558-571, today in-memory 0-D
inputs from `read_surface_series`) becomes the same call on its `target_space`, which the
caller (`external_forcing_cache`, `external_forcing.jl:520-530`) already passes as the
surface space `axes(Fields.level(Y.f.u₃, half))`; a `(time,)` variable becomes a one-level
segment. Delete the format hook `extrapolation_bc` (lines 193-199, its mention in the
module docstring line 19, and `ClimaAtmos.ColumnDatasets.extrapolation_bc` in
`docs/src/api.md`); keep `read_surface_series` (used by `FileHeatFluxes`) and the
`Interpolations` import (used by `_interp_column`). `start_date` is a `DateTime`
everywhere it is passed (`AtmosSimulation` parses it); the ragged constructor requires it.

Test (`test/column_datasets_tests.jl`, testset "ClimaColumn files onto a column space"):
the surface-series check evaluates `surface_fields.ts` into a center field `dest`; a
per-column input must be evaluated on the space it was built for, so build the inputs on
`ClimaCore.Spaces.level(center_space, 1)` and evaluate into
`ClimaCore.Fields.zeros(surface_space)`. Everything else in the file is unchanged.

Behaviour: identical interface for the forcing terms; the file is read once into memory
instead of two slices at a time; results are bitwise equal for surface series and within
one Float32 rounding of the file data for profiles on Float64 columns (Section 1.1). Check
with `multi_column_dev/ragged_inputs_check.jl` and `ragged_precision_check.jl`
(reference numbers in `MULTI_COLUMN_CHECKLIST.md` §11).

### 3.4 Commit 2: a vector of datasets

In `ColumnDatasets.jl`, after the `ColumnDataset` struct:

```julia
const ColumnData = Union{AbstractColumnData, AbstractVector{<:AbstractColumnData}}
```

with a docstring ("one source read by every column, or one dataset per column in column
order"), and the vector methods, one line each, typed
`AbstractVector{<:AbstractColumnData}` unless noted:

- `require_forcing_variables(data, column_vars, surface_vars) = foreach(...)`
- `time_interpolation_method(data) = only(unique(map(time_interpolation_method, data)))`
- `file_time_span(data, start_date) = minimum(cd -> file_time_span(cd, start_date), data)`
- `read_initial_profiles(data, start_date) = map(...)`
- `site_location(data) = (; latitude = [...], longitude = [...])` (a NamedTuple of vectors,
  so `(; latitude, longitude) = site_location(data)` works for scalar and vector)
- `surface_vars(d::AbstractColumnData) = d.surface_vars`,
  `surface_vars(data::AbstractVector) = intersect(map(surface_vars, data)...)`
- `data_source(data::AbstractVector{<:ColumnDataset}, name) = map(...)`
- `_format(cd::ColumnDataset) = cd.format`, `_format(data::AbstractVector{<:ColumnDataset})
  = only(unique(map(_format, data)))` (the files of a run share their format; used for
  `preprocess`).

Widen `column_timevaryinginputs`/`surface_timevaryinginputs` of commit 1 to
`data::Union{ColumnDataset, AbstractVector{<:ColumnDataset}}` (default `method =
time_interpolation_method(data)`, `preprocess(_format(data), name)`). In `src/types.jl`
change `ExternalDrivenTVForcing{CD <: ColumnDatasets.AbstractColumnData, ...}` (line 1247)
and its constructor argument (line 1257) to `ColumnDatasets.ColumnData`, and the docstring
line "`dataset`: The `ColumnDatasets.ColumnDataset` handle" accordingly. Add
`ClimaAtmos.ColumnDatasets.ColumnData` to the `@docs` block of `docs/src/api.md`.

Test: new testset "One file per column" in `test/column_datasets_tests.jl`: write two
ClimaColumn files with `CD.ClimaColumnFiles.write_column_forcing_file` whose `ta` is
linear in height and time and differs by an offset, on an hourly axis (0-8 h) and a
two-hourly axis (0-6 h); build `CA.MultiColumnGrid(FT; points, radius, z_elem = 10, z_max
= 6000.0, z_stretch = false)` with three points and `datasets = [a, b, a]`; evaluate
`column_timevaryinginputs(datasets, (:ta,), center_space, start_date).ta` at 3.5 h and
check each column against `280 + offset + 1e-3 clamp(z, 100, 5000) + 3.5`;
`surface_timevaryinginputs(datasets, (:ts,), Spaces.level(face_space, half), ...)`;
`file_time_span(datasets, start_date) == 6 * 3600`; `require_forcing_variables` errors for a
variable one file lacks. (This testset existed on 2026-09-15 and passed; rewrite it, the
version with `sites` is gone.)

### 3.5 Commit 3: initial condition by site (`src/setups/ForcingFromFile.jl`)

Today (`ForcingFromFile.jl:37-84`): `struct ForcingFromFile{CD <: AbstractColumnData, F, FS,
ST, I, P <: ColumnProfiles}` with `profiles::P` built from `read_initial_profiles(dataset,
start_date)`, and `center_initial_condition(setup, local_geometry, params) =
column_profiles_ic(setup.profiles, local_geometry)`.

Change: `CD <: ColumnDatasets.ColumnData`; `P` unconstrained (a `ColumnProfiles` or a
vector of them); a new field `sites` holding the resolved sites as a NamedTuple of vectors
(one entry per dataset of a list, one entry for a scalar dataset, `NaN` where a dataset has
no site), used by 3.6 to place the columns and by the match below. Builders:

```julia
column_profiles(dataset::ColumnDatasets.AbstractColumnData, start_date) = ColumnProfiles(values(read_initial_profiles(dataset, start_date))...)  # keep the explicit z, ta, ua, va, hus, rho order as today
column_profiles(datasets::AbstractVector, start_date) = map(d -> column_profiles(d, start_date), datasets)
```

Selection, pointwise:

```julia
site_profiles(profiles::ColumnProfiles, sites, coords) = profiles
function site_profiles(profiles::AbstractVector, sites, coords)
    i = findfirst(j -> sites.latitude[j] == coords.lat && sites.longitude[j] == coords.long, eachindex(profiles))
    isnothing(i) && error("No forcing dataset at ($(coords.lat), $(coords.long))")
    return profiles[i]
end
center_initial_condition(setup::ForcingFromFile, local_geometry, params) =
    column_profiles_ic(site_profiles(setup.profiles, setup.sites, local_geometry.coordinates), local_geometry)
```

Equality is exact because the grid points are built (3.7) from the same `Float64`
attributes with the same `FT(...)` conversion; compare `FT(sites.latitude[j]) == coords.lat`
with `FT = typeof(coords.lat)` to be explicit. Constructor rule: if two datasets have the
same site and different `path`s, `error("Columns at one site must read the same file")`.
Datasets without a site in a vector: `site_location` of an `InMemoryColumnData` without a
location errors today (`ColumnDatasets.jl:629-632`); for decision 3.1.3 the vector method
must return `NaN` for a missing location rather than error (`(; latitude = NaN, longitude =
NaN)` from a `nothing`), and 3.7 substitutes the configured coordinate; the match in
`site_profiles` then uses the substituted values, so `ForcingFromFile` must receive the
final sites, not compute them: build them in `get_setup_type` (3.7) and pass `sites` as a
keyword (default `site_location(dataset)`; the GCM/ARM/ERA5 branches pass the resolved
list). The docstring gains the vector case and the same-site rule.

Test: extend "One file per column": `setup = CA.Setups.ForcingFromFile(datasets,
"20000506")` with points equal to the files' sites; `CA.Setups.initial_condition_field(lg
-> CA.Setups.center_initial_condition(setup, lg, params), center_space)` and check
`Fields.field2array(ic.T)[:, c]` per column; `ForcingFromFile([a, b], date)` with equal
sites errors.

### 3.6 Commit 4: config lists (`src/config/type_getters.jl`, `era5_observations_to_forcing_file.jl`)

Helper next to `get_setup_type`:

```julia
per_column(f, option) = option isa AbstractVector ? map(f, option) : f(option)
```

(no length check against the config coordinates: the list defines the column count,
3.1.1). Branches of `get_setup_type` (lines 182-319):

- `ReanalysisTimeVarying` (line 245): `era5_datasets(parsed_args, FT)`.
- `ForcingFromFile` (line 251): `per_column(ColumnDatasets.ColumnDataset,
  parsed_args["external_forcing_file"])`.
- `ARMVARANAL` (line 199): `data = per_column(varanal_file) do file; ColumnDataset(
  to_climacolumn(file; thermo_params, dir = BUILDKITE ? mktempdir() : dirname(file))); end`;
  `flux_scheme` when `issubset((:hfls, :hfss), ColumnDatasets.surface_vars(data))`
  (commit 7 makes `FileHeatFluxes` accept the vector); insolation:
  `TimeVaryingInsolation(; start_date)` for a vector (the grid carries the sites),
  unchanged explicit `latitude`/`longitude` from `site_location(data)` for a scalar (a
  single column has no coordinates and would otherwise fall back to the equator).
- `GCM` (line 193): `per_column(n -> read_cfsite(parsed_args["external_forcing_file"], n;
  thermo_params), parsed_args["cfsite_number"])`.
- In `era5_observations_to_forcing_file.jl` after `era5_dataset` (line 697):

```julia
function era5_datasets(parsed_args, ::Type{FT}; monthly = false) where {FT}
    lats, lons = parsed_args["site_latitude"], parsed_args["site_longitude"]
    lats isa AbstractVector || lons isa AbstractVector || return era5_dataset(parsed_args, FT; monthly)
    lats isa AbstractVector && lons isa AbstractVector && length(lats) == length(lons) ||
        error("`site_latitude` and `site_longitude` must both be lists of the same length")
    sites = collect(zip(lats, lons))
    # Columns at one site share its file, located or generated once
    dataset(site) = era5_dataset(merge(parsed_args, Dict("site_latitude" => site[1], "site_longitude" => site[2])), FT; monthly)
    by_site = Dict(site => dataset(site) for site in unique(sites))
    return [by_site[site] for site in sites]
end
```

  and `model_getters.jl:872` (`ReanalysisMonthlyAveragedDiurnal`) calls `era5_datasets(...;
  monthly = true)`. `era5_dataset` and the path helpers keep reading scalar
  `site_latitude`/`site_longitude` from the merged dictionary.

Grid from the datasets. `get_simulation` (`type_getters.jl:717-720`) builds `setup` before
`grid = get_grid(pa, params, config.comms_ctx)`. Add `Setups.column_sites(setup) = nothing`
and `column_sites(setup::ForcingFromFile) = setup.sites`, the NamedTuple of vectors from
3.5 (one entry per dataset for a list, one entry for a scalar dataset, `NaN` where a
dataset has no site); call `get_grid(pa, params, ctx; sites = Setups.column_sites(setup))`
and in the `multicolumn` branch of `get_grid` (line 538-547) build the points as

```julia
lats, lons = parsed_args["column_latitudes"], parsed_args["column_longitudes"]
if !isnothing(sites)
    # The datasets place the columns: one site per dataset, or one site repeated for a
    # shared dataset; a dataset without a site keeps the configured coordinate
    n = max(length(sites.latitude), length(lats))
    at(v, i) = v[min(i, length(v))]
    lats = [isnan(at(sites.latitude, i)) ? at(lats, i) : at(sites.latitude, i) for i in 1:n]
    lons = [isnan(at(sites.longitude, i)) ? at(lons, i) : at(sites.longitude, i) for i in 1:n]
end
points = Geometry.LatLongPoint{FT}.(lats, lons)
```

so a list of datasets sets the column count (a shorter configured list is repeated), a
scalar dataset takes the configured column count with its site repeated, and a scalar
dataset without a site reproduces today's placement. `ForcingFromFile` must hold the same
resolved `sites` (3.5) so the initial-condition match sees the final coordinates; for a
scalar dataset no match is needed (`profiles` is a single `ColumnProfiles`). `check_case_consistency` still checks the configured
lists are non-empty and of equal length; nothing else changes. `read_cfsite`
(`GCMColumnData.jl:115-131`) passes `site_location = (; latitude = Float64(site["lat"][]),
longitude = Float64(site["lon"][]))` (the cfsite groups hold `lat`/`lon` scalars; read
them with `haskey` guards and `nothing` when absent).

Help texts in `config/default_configs/default_config.yml` (`external_forcing_file`,
`site_latitude`, `site_longitude`, `cfsite_number`, `column_latitudes`,
`column_longitudes`): each dataset option may be a list with one entry per column of a
`multicolumn` configuration; the datasets' sites place the columns.

Tests: `test/config/model_from_config.jl` gets a case with lists for
`external_forcing_file` on the "One file per column" files, checking
`get_setup_type` returns a `ForcingFromFile` with a 3-vector and that `get_grid` with its
`column_points` places the columns at the files' sites; a scalar path on
`config: multicolumn` keeps the configured coordinates.

### 3.7 Commit 5: GCM per-column steady fields (`ColumnDatasets.jl`)

Factor the interpolant out of `_interp_column` (lines 645-658): `_interpolant(z_src,
vals_src)` returns the `Intp.extrapolate(...)` object. Add

```julia
function column_timevaryinginputs(data::AbstractVector{<:InMemoryColumnData}, names, target_space, start_date; method = nothing)
    ᶜz = ClimaCore.Fields.coordinate_field(target_space).z
    z = Array(ClimaCore.Fields.field2array(ᶜz))
    inputs = map(Tuple(names)) do name
        values = similar(z)
        for (c, d) in enumerate(data)
            values[:, c] .= _interpolant(d.z, d.column[name]).(z[:, c])
        end
        field = similar(ᶜz)
        copyto!(ClimaCore.Fields.field2array(field), values)
        TimeVaryingInput(Returns(field))
    end
    return NamedTuple{Tuple(names)}(inputs)
end
function surface_timevaryinginputs(data::AbstractVector{<:InMemoryColumnData}, names, target_space, start_date; method = nothing)
    FT = ClimaCore.Spaces.undertype(target_space)
    inputs = map(Tuple(names)) do name
        field = ClimaCore.Fields.zeros(target_space)
        copyto!(ClimaCore.Fields.field2array(field), [FT(d.surface[name]) for d in data])
        TimeVaryingInput(Returns(field))
    end
    return NamedTuple{Tuple(names)}(inputs)
end
```

(host computation, one `copyto!` to the device; `field2array` of a level field on a
multi-column space is a vector of length `ncolumns` in column order). The scalar
`InMemoryColumnData` methods are unchanged.

Test: extend "In-memory column data onto a column space": a second cfsite group with a
different `nt` (different time means) in the same file (`write_test_cfsite_file` needs a
`mode = "a"` keyword to add a group), `datasets = [d23, d17, d23]` on a three-column grid,
per-column profiles against each dataset's interpolant, surface constants per column, and
`get_setup_type` with `"cfsite_number" => ["site23", "site17", "site23"]`.

### 3.8 Commit 6: RRTMGP latitude guard (`src/parameterized_tendencies/radiation/radiation.jl`)

Two sites, lines 153 and 198:

```julia
latitude = if eltype(bottom_coords) <: Geometry.LatLongZPoint && !iscolumn(ᶜspace)
```

with the same comment style as `compute_coriolis` (`cache.jl:319-320`): columns have no
horizontal position, so RRTMGP sees them at the equator like the Cartesian column. This
makes the all-sky dry-air amount (RRTMGP `Optics.jl:138`, `as.lat`) and the gray optical
thickness independent of the column's placement.

Verification (the run that justifies the commit): the ERA5 EDMF config
(`prognostic_edmfx_tv_era5driven_column.yml`, `allskywithclear`) as `ForcingFromFile` on
the real site file, single column against a three-column run whose columns sit at
17°N/-149°E (the file's site) and at two other latitudes, 10 minutes: every column's state
must be bitwise identical to the single column (before this commit the columns away from
the equator differ through the RRTMGP dry-air amount; measure that first so the commit
message can state it). Script to adapt: `multi_column_dev/multisite_case.jl` (three columns
reading `[A, B, A]` against single columns on A and B) and
`examples/multi_column/comparison_utils.jl` (`field_diffs`, `report_diffs`,
`diagnostic_diffs`, `CACHE_IGNORE`).

### 3.9 Commit 7: ARM per-column prescribed fluxes

Why it is needed: ARM's `flux_scheme` is `MoninObukhov(; z0, ustar, fluxes =
FileHeatFluxes(data, start_date))` (`type_getters.jl:222-226`). `FileHeatFluxes`
(`surface_setups.jl:26-100`) holds two `Interpolations` interpolants over the file's
`hfls`/`hfss` series and is a callable `(t, FT) -> HeatFluxes(; shf, lhf)`; `resolve_flux_scheme`
(`surface_conditions.jl:85-89`) evaluates it once per surface update and rebuilds
`MoninObukhov(z0m, z0b, fluxes, ustar)`, which `update_surface_conditions!` passes as a
scalar argument into the per-cell broadcast `surface_state_to_conditions(...)`
(`surface_conditions.jl:52-70`). One flux pair therefore reaches every column. With one
dataset per column each column needs its own pair.

Change 1, `src/surface_conditions/surface_setups.jl`:

```julia
# One dataset per column: one interpolant pair per column
function FileHeatFluxes(data::AbstractVector{<:ColumnDatasets.ColumnDataset}, start_date; nan_to_zero = true)
    parts = map(d -> FileHeatFluxes(d, start_date; nan_to_zero), data)
    return FileHeatFluxes(getfield.(parts, :lhf_interp), getfield.(parts, :shf_interp), nan_to_zero)
end
```

and in the callable `(f::FileHeatFluxes)(t, ::Type{FT})` evaluate each interpolant:
`_at(interp, t) = interp(t)`, `_at(interps::AbstractVector, t) = map(i -> i(t), interps)`;
the NaN replacement becomes `ifelse.(isnan.(lhf), 0.0, lhf)` (works for a scalar and a
vector); return `HeatFluxes(; shf = FT.(shf), lhf = FT.(lhf))`. `HeatFluxes{FT, FTN}`
(`surface_state.jl:77`) has no constraint on `FT`, so `HeatFluxes{Vector{Float32},
Vector{Float32}}` is a valid value of the callable. Docstring: "or one dataset per column,
giving one flux pair per column".

Change 2, `src/surface_conditions/surface_conditions.jl`: give `resolve_flux_scheme` the
surface space and let it return a per-cell layout when the fluxes are per column:

```julia
flux_scheme = resolve_flux_scheme(atmos.surface.flux_scheme, t, eltype(params), axes(sfc_conditions))   # line 50
```

```julia
# Custom schemes keep the three-argument method documented in surface_conditions_internals.md
resolve_flux_scheme(p, t, ::Type{FT}, _) where {FT} = resolve_flux_scheme(p, t, FT)
function resolve_flux_scheme(p::MoninObukhov, t, ::Type{FT}, surface_space) where {FT}
    p.fluxes isa Function || return p
    fluxes = p.fluxes(t, FT)
    fluxes isa HeatFluxes && fluxes.shf isa AbstractVector ||
        return MoninObukhov(p.z0m, p.z0b, fluxes, p.ustar)
    # One flux pair per column: one parameterization per surface point
    column_field(values) = (field = Fields.zeros(surface_space); copyto!(Fields.field2array(field), values); field)
    shf = column_field(fluxes.shf)
    lhf = isnothing(fluxes.lhf) ? nothing : column_field(fluxes.lhf)
    return Fields.field_values(@. MoninObukhov(p.z0m, p.z0b, HeatFluxes(shf, lhf), p.ustar))
end
```

The broadcast `@. sfc_conditions_values = surface_state_to_conditions(overrides,
flux_scheme, T_sfc_values, ...)` already mixes scalars and `DataLayout`s (that is what
`boundary_overrides_wrapper` does for per-cell overrides and what `T_sfc_values` is), so a
`DataLayout` of `MoninObukhov{FT, HeatFluxes{FT, FT}, FT}` in place of the scalar needs no
change in the kernel: `parameterization isa MoninObukhov` and `parameterization.fluxes
isa HeatFluxes` (`surface_conditions.jl:258-283`) hold per cell. `HeatFluxes(shf, lhf)` and
`MoninObukhov(z0m, z0b, fluxes, ustar)` are the positional default constructors
(`@kwdef` keeps them; `m_o` in `surface_state.jl:200-214` uses the latter). Cost: two
surface fields of `ncolumns` values allocated per surface update; acceptable, cache them
in `p` later if profiling asks. `field2array` of a surface field on a multi-column space is
a vector in column order, the order of the datasets.

Docs: `docs/src/surface_conditions_internals.md:26, 77, 152-157` describe the
three-argument `resolve_flux_scheme`; add a sentence that the built-in method receives the
surface space and returns a per-cell layout for column-valued fluxes.

Test (`test/column_datasets_tests.jl`, testset "ForcingFromFile surface and insolation
seams"): `fluxes = CA.SurfaceConditions.FileHeatFluxes([data, data], "20000506")`;
`hf = fluxes(0.0, FT)` has `hf.shf isa Vector{FT}` of length 2 equal to the scalar
version's value; `scheme = CA.SurfaceConditions.resolve_flux_scheme(CA.SurfaceConditions.MoninObukhov(;
z0 = 0.05, ustar = 0.28, fluxes), 0.0, FT, surface_space)` on the level space of a
two-column grid is a `DataLayout` whose two entries are `MoninObukhov`s with the scalar
`HeatFluxes`; the scalar `FileHeatFluxes(data, ...)` path is unchanged.

Run: ARM config (`prognostic_edmfx_armvaranal_column.yml`, 1M microphysics, `dt` 120 s)
as a two-column run with `external_forcing_file: [file, file]` against the single column,
10 minutes: bitwise identical state (both columns at the SGP site from the converted
file's attributes; the single column's `TimeVaryingInsolation` uses the explicit site,
the two-column run's the grid coordinates, which are the same numbers).

### 3.10 Commit 8: docs

`docs/src/configuration.md` "Multiple independent columns": the dataset options accept one
entry per column and the datasets' sites place the columns; a single value drives every
column with the same data. `docs/src/column_datasets_reference.md`: the list form of
`external_forcing_file` and `cfsite_number`. `docs/src/single_column.md` ERA5 section:
`site_latitude`/`site_longitude` lists. NEWS entry: the user writes it.

### 3.11 Verification protocol and definition of done

- Unit tests to run in the REPL after each commit (isolated module,
  `Core.eval(m, :(using Test))`, `include` with `path = Base.include($m, path)`):
  `test/column_datasets_tests.jl`, `test/config/model_from_config.jl`, `test/grids.jl`.
- Input-level checks: `multi_column_dev/ragged_inputs_check.jl` (old path against new on
  the real ERA5 file; expected: surface bitwise, profiles at Float32 rounding),
  `ragged_precision_check.jl`.
- Full runs. The template for every case is **two single-column runs and one
  multi-column run**: a single column on dataset A, a single column on dataset B, and a
  multi-column run whose dataset list is `[A, B, A]` (three columns; the repeated A checks
  the shared-segment path and that duplicated sites work) or `[A, B]` when a duplicate is
  not wanted. Pass criteria, all at `rtol = atol = 0` on the final state (`field_diffs`
  from `examples/multi_column/comparison_utils.jl`, `col2 = k` selecting the column):
  column 1 == single A, column 2 == single B, column 3 == single A and == column 1, and
  column 2 differs from column 1 in every state entry (the second dataset was really
  used). The A9 uninitialized `ᶜmp_tendency` scratch and the inputs' `range` bookkeeping
  are the only admissible cache differences, `clt`/`cltl` (McICA, 2.1 row 11) the only
  admissible NetCDF differences; run at least one case long enough to write NetCDF output
  and to cross a file node (the ERA5 file is hourly: 3 hours). Each run is 10 minutes
  unless noted:
  1. ERA5 EDMF config as `ForcingFromFile`: A the real site file, B a warmer two-hourly
     copy of A written by the script (`multi_column_dev/multisite_case.jl` does exactly
     this, columns at the files' sites); also the 3-hour variant.
  2. The same with the multi-column run's columns away from the equator (a site attribute
     changed in B's copy), before and after commit 6: before, the columns off the equator
     differ through the RRTMGP dry-air amount; after, all three match.
  3. GCM: A = `site23`, B = `site17` of the artifact file (51 groups),
     `cfsite_number: [site23, site17, site23]`.
  4. ARM: only one dataset exists, so A = B = the converted file and the multi-column run
     is `[file, file]`; both columns == the single column (commit 7's per-column fluxes
     and the grid-coordinate insolation are exercised, at the same site).
  5. ERA5 with lists: `site_latitude: [17.0, 17.0, 17.0]`, `site_longitude: [-149.0,
     -150.0, -149.0]`, single columns at (17, -149) and (17, -150), under
     `ENV["BUILDKITE"] = "true"` (files generated from the raw artifact, about 40 s per
     site); also check `era5_datasets` returns a 3-vector whose first and third entries
     are the same object.
- Done when every run above is bitwise per column, the tests pass, the touched files are
  formatted with the pinned formatter, and `MULTI_COLUMN_CHECKLIST.md` §11 records the
  numbers.

## Part 4. Implementation status (2026-09-16)

Part 3 was implemented as eight commits on `kp/multi-col` (one per step, 3.3 to 3.10; the
two follow-up fixes noted below were squashed into their steps), verified as in 3.11 on
the CPU and, for the ERA5 and GCM cases, on one A100. Full numbers are in
`MULTI_COLUMN_CHECKLIST.md` §12. In one line: on the CPU every column of every case
(ERA5 `ForcingFromFile` at 10 minutes and 3 hours, GCM, ARM, ERA5 site lists) is bitwise
its own single-column run in state, and so are the ERA5 and GCM cases on one A100; the only
cache differences are the known uninitialized `ᶜmp_tendency` scratch and the ragged inputs'
`range` bookkeeping, the only NetCDF difference the McICA `clt`. The before/after runs of
the RRTMGP guard show the columns at 17 N differing in every state entry without it.

Deviations from the plan, all small:

- **A dataset without a site in a list is an error** at `ForcingFromFile` construction
  ("records no site; pass `sites`"), not a fallback to the configured coordinate
  (3.1.3): the list's initial condition is matched by coordinates and the setup does not
  see the configured coordinates. A scalar dataset without a site does fall back. Only a
  runscript-built `InMemoryColumnData` can lack a site; cfsite groups carry `lat`/`lon`,
  ClimaColumn files carry `site_latitude`/`site_longitude`.
- **The YAML loader needed one rule**, `coerce_to_default(::Type{T}, ::AbstractVector)`
  for a scalar number or string default (entry-wise coercion), because every value is
  coerced to the type of its default and `site_latitude`/`cfsite_number` defaults are
  scalars. `external_forcing_file` (default `nothing`) already passed lists through.
- **ARM conversion writes one directory per call** (`BUILDKITE`: one `mktempdir()`),
  so a file listed twice converts to one path and passes the same-site check; the check
  compares `source_name`s, i.e. paths, not contents.
- **The scalar `InMemoryColumnData` inputs go through the per-column builder**
  (`fill(d, ncolumns)`), which interpolates on the host and copies to the device; the
  former `_interp_column` broadcast an `Interpolations` object over the field, which
  does not compile on the GPU (a pre-existing gap of the GCM case on `main`).
- **`TimeVaryingInsolation` converts an explicit site to the model float type** before
  calling `Insolation.insolation` (commit 7): `solar_geometry` forms
  `FT(deg2rad(latitude))`, so a `Float64` site and the same site as a `Float32` grid
  coordinate could differ in the last bit; the ARM single column (explicit site) and its
  multi-column run (grid coordinates) now compute the same insolation. This moves the
  ARM single-column result on `main` at the Float32 rounding level.
- `column_points` (the placement rule of 3.6) lives next to `get_grid` in
  `type_getters.jl`; `get_grid` errors when a list of datasets meets `config: column`.

GPU notes (details in the checklist): Julia 1.12.5 still fails in ClimaCore's eager
stencil kernel, so the GPU runs used Julia 1.11.4 (`climacommon/2025_03_18`) with a scratch
environment built from `.buildkite/Manifest-v1.11.toml`. Two things outside this PR were
needed for file-forced runs on the GPU and are **not committed**: the surface-conditions
kernel must receive `atmos.microphysics_model` instead of the `AtmosModel` (checklist §8,
applied in the working tree), and the fused product inside the subsidence interpolation
stencil (`external_forcing.jl:229`, unchanged from `main`) no longer compiles with the
current ClimaCore worktree, worked around by a GPU-only method override
(`multi_column_dev/gpu_subsidence_override.jl`) that materializes the center vector field
first. Neither changes any result.
