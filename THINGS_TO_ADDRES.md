# Multi-column branch: adversarial review

Date: 2026-09-09. Scope: commit `12a1c8f7c` ("need to check", the only commit ahead of
`origin/main` on `kp/multi-col`) plus the untracked `MULTI_COLUMN_CHECKLIST.md` and
`examples/multi_column/`. Static analysis only; nothing was executed. Every file:line was
checked against the files on disk. ClimaCore refers to the dev'd worktree
`ClimaCore.jl/col-intp`; ClimaDiagnostics to
`ClimaDiagnostics.jl/multi-cols`.

Companion: `DIFF_SPACE.md` (site-by-site comparison of the single-column and multi-column
spaces). Items already listed there are referenced, not repeated, except where the commit
changes the picture.

Severity key: **[BLOCK]** breaks CI or the run as committed; **[BUG]** wrong result or
crash on a reachable path; **[RISK]** latent or fragile; **[NIT]** cosmetic or docs.

---

## 1. Things that break CI or the repository as committed

### 1.1 [BLOCK] `.buildkite/Manifest.toml` is committed with absolute local dev paths

- `git show HEAD:.buildkite/Manifest.toml` lines 382 and 413:
  `path = "ClimaCore.jl/col-intp"` and
  `path = "ClimaDiagnostics.jl/multi-cols"`.
- `main` does not track this file (`git ls-tree main .buildkite/Manifest.toml` is empty).
- `.buildkite/pipeline.yml:60` runs `Pkg.instantiate(; update_registry=false)` in that
  project, so every CI job fails on the missing paths.
- Root cause worth fixing separately: `.gitignore:28` reads
  `*/Manifest*.toml  # docs, test`. gitignore has no trailing-comment syntax, so the
  pattern literally includes `  # docs, test` and never matches. That is why the file
  showed as `??` and could be committed. Only the root `/Manifest*.toml` rule works.
- Action: `git rm --cached .buildkite/Manifest.toml`, fix the ignore rule, and record the
  two dev'd branches somewhere human-readable (the checklist already does) instead.

### 1.2 [BLOCK] Docs build fails on the new export

- `src/simulation/grids.jl:7` exports `MultiColumnGrid`; it has a docstring
  (`grids.jl:131-152`). `grids.jl` is included at module level
  (`src/ClimaAtmos.jl:191`), so the export is a module export.
- `docs/src/api.md:41-45` lists `SphereGrid`, `ColumnGrid`, `BoxGrid`, `PlaneGrid` only.
- `docs/make.jl:46` sets `checkdocs = :exports`, and `docs/make.jl:53-55` only downgrades
  `:linkcheck` and `:external_cross_references` to warnings. A missing docstring for an
  exported symbol is therefore a hard Documenter error.
- Action: add `ClimaAtmos.MultiColumnGrid` to the `@docs` block. While there, the
  `AtmosSimulation` docstring at `src/simulation/AtmosSimulations.jl:243-245` lists the
  grid constructors and omits the new one, and `docs/src/interfaces.md:33-35` has no
  multi-column row.

### 1.3 [BLOCK, probable] Formatter check

- `.JuliaFormatter.toml` sets `margin = 92` and does not exclude `examples/`. Lines over
  the margin in new files:
  - `examples/multi_column/comparison_utils.jl:164` (111 chars)
  - `examples/multi_column/comparison_utils.jl:166` (94 chars)
  - `examples/multi_column/single_vs_multi_column.jl:78` (96 chars)
- JuliaFormatter with `join_lines_based_on_source = true` may or may not rewrap each of
  these; I could not run it. The `prek` hook in CI will show a diff if it does.
- Action: run `prek run julia-formatter --all-files` before pushing.

### 1.4 [NIT] `DIFF_SPACE.md` is in the commit

Probably not intended for a PR. Same question for `MULTI_COLUMN_CHECKLIST.md` and
`THINGS_TO_ADDRES.md` once they are tracked.

---

## 2. Functional gaps in the committed Atmos code

### 2.1 [BUG] A default `config: multicolumn` run still crashes

- `config/default_configs/default_config.yml:183` `hyperdiff: "Hyperdiffusion"`.
- `src/prognostic_equations/hyperdiffusion.jl:22`
  `h = Spaces.node_horizontal_length_scale(Spaces.horizontal_space(axes(Y.c)))`.
  For the multi-column space the horizontal space is a `MultiPointSpace`; ClimaCore
  defines `node_horizontal_length_scale` only for `AbstractSpectralElementSpace`,
  `PointSpace`, and `Nothing` (`Spaces/spectralelement.jl:193,200`,
  `Spaces/pointspace.jl:70`). MethodError on the first call to
  `apply_hyperdiffusion_tendency!` (`hyperdiffusion.jl:255`) or
  `apply_tracer_hyperdiffusion_tendency!` (`:464`).
- The checklist (§4 C1) states every verified run used a REPL-only shim
  `node_horizontal_length_scale(::MultiPointSpace) = 1` that exists in neither repo. The
  commit therefore depends on code that is nowhere on disk.
- Same pattern, opt-in: `smagorinsky_lilly.jl:96-97`,
  `anisotropic_minimum_dissipation.jl:69-70, 210-211`.
- Action: either land the ClimaCore method first and note the dependency in the PR, or
  guard `ν₄` (and the LES closures) on `iscolumn`. Note that `horizontal_filter_scale`
  in `src/utils/utilities.jl:421-427` already encodes the right policy (`Inf` for
  columns); `ν₄` could reuse it.

### 2.2 [BUG] `MultiColumnGrid` accepts `context` and silently discards all but the device

- `src/simulation/grids.jl:153-171`: the kwarg is documented as "the ClimaComms
  communications context" but only `ClimaComms.device(context)` is forwarded.
- ClimaCore `_MultiPointGrid` (`Grids/multipoint.jl:84-88`) always builds a
  `SingletonCommsContext(device)`. Under MPI every rank would build and time-step all
  columns, with `ClimaComms.context(axes(Y.c))` (singleton) disagreeing with
  `config.comms_ctx` (MPI). `radiation.jl:383` uses the former, `get_callbacks.jl:404`
  and the output-path generator use the latter.
- `CommonGrids.ColumnGrid` asserts a singleton context for exactly this reason
  (`CommonGrids.jl:340`). The wrapper should do the same, or drop the kwarg.

### 2.3 [BUG] No validation of the column coordinates beyond equal length

- `src/config/type_getters.jl:538-548`.
- Empty lists: `MultiPointGrid` builds a zero-element `VIJFH` and the failure surfaces
  somewhere downstream with no mention of the config key.
- Longitude is never range-checked (ClimaCore checks `|lat| < 90` only,
  `Grids/multipoint.jl:96-98`).
- Scalars are fine: `yaml_helper.jl:130` `coerce_to_default(::Type{T}, v) = convert(T, v)`
  rejects `convert(Vector{Float64}, 30.0)` with a clear message. Integer lists convert.
- Nothing warns when `column_latitudes` / `column_longitudes` are set with a non
  `multicolumn` config; they are silently ignored (`yaml_helper.jl:155` only flags keys
  absent from the default config).
- Action: move the checks into `check_case_consistency` next to `valid_configs`
  (`model_getters.jl:1040`), add `isempty` and longitude range, and warn on the unused
  keys.

### 2.4 [BUG] `check_case_consistency` has no `multicolumn` rules

- `deep_atmosphere` defaults to `true` (`default_config.yml:439-441`). The multi-column
  grid is always shallow (`CommonGrids.jl:786-794` never forwards `deep`). RRTMGP still
  applies the `((z+R)/R)^-2` area scaling because `radiation.jl:149-150` passes
  `planet_radius` for any `AbstractSphericalGlobalGeometry` and the radiation mode carries
  `deep_atmosphere = true` (`model_getters.jl:613`). The checklist measured a 2% offset
  in `rsdt`/`rlut`/`ρe_tot`. See `DIFF_SPACE.md` §5 for the full table.
- `ReanalysisMonthlyAveragedDiurnal` asserts `config == "column"`
  (`model_getters.jl:853`), which now silently excludes `multicolumn` with a message that
  does not mention it.
- Action: at minimum error or warn on `config == "multicolumn" && deep_atmosphere`.

### 2.5 [RISK] Behavior change hidden in the `iscolumn` widening

- `src/utils/utilities.jl:405-408, 939`: `iscolumn` now matches
  `MultiColumnFiniteDifferenceSpace`.
- `non_orographic_gravity_wave.jl:140, 353`: multi-column runs now take the flat-column
  NOGW branch (height-based source, latitude-independent coefficients) even though they
  carry latitude. The checklist treats this as intended; the commit message and code do
  not say so, and no test pins it.
- The `elseif issphere(...)` fallback at the same sites would throw on a
  `MultiPointSpace` (`utilities.jl:947-949` goes through `Spaces.topology`). Ordering is
  the only thing keeping it unreachable. Fix `issphere` to use
  `Spaces.global_geometry` regardless (`DIFF_SPACE.md` E5).
- `edmfx_sgs_flux.jl:440` early return is also widened; harmless (would be a zero no-op).

### 2.6 [RISK] `do_dss(::ColumnSpace)` is load-bearing, not redundant

- ClimaCore has no space-level `quadrature_style` for `MultiPointSpace` (grep of
  `src/Spaces/*.jl` finds none; only `Grids.quadrature_style(::MultiPointGrid) = nothing`
  at `Grids/multipoint.jl:47`). The generic `do_dss` at `utilities.jl:626-629` would hit
  a MethodError on the multi-column space. Keep the override; the checklist's C2 is the
  upstream fix.
- Corollary: any other Atmos call of `Spaces.quadrature_style(Spaces.horizontal_space(…))`
  on the multi-column space fails the same way. `autodiff_utils.jl:100` is guarded by the
  new early return at `:98-99`; nothing else in `src/` calls it.

### 2.7 [NIT] Stale comment and looser error in touched code

- `src/prognostic_equations/implicit/autodiff_utils.jl:80` still says "The horizontal
  SpectralElementSpace of the fields in a FieldVector"; it can now return a
  `MultiPointSpace`.
- `src/simulation/grids.jl:411-412`: `get_spaces` no longer raises the explicit
  "Unsupported grid type" error; unsupported grids surface a ClimaCore MethodError. No
  test depended on the message (`grep` of `test/` is empty), so this is a judgment call.
- `src/config/model_getters.jl:1021-1022`: the docstring line break lands mid-sentence
  ("`\"plane\"`; that an ISDAC\nrun").

### 2.8 Verified correct in the commit

- `radiation.jl:442` `ncol = Spaces.ncolumns(axes(Y.c))`: for `SpectralElementSpace2D`
  `ncolumns` is `Nh·Nq²` (`Spaces/Spaces.jl:214-218`), identical to the old
  `length(all_nodes(...))` (`Spaces/spectralelement.jl:202-206`). Single column gives 1
  (`Spaces/finitedifference.jl:122`). Plane grids previously had no `all_nodes` method
  and now work.
- `autodiff_utils.jl:98-99`: `(1, 1, h)` tuples flow into `point` (`:152-153`), which
  calls `Fields.column(value, column_index...)`; ClimaCore defines `column` for the
  multi-column space (`Spaces/multicolumn.jl:131-132`). Consumers at
  `auto_dense_jacobian.jl:129-193` and `auto_sparse_jacobian.jl:92-524` only splat the
  tuple.
- `get_spaces` via `Spaces.space(grid, staggering)`: methods exist for
  `AbstractFiniteDifferenceGrid`, `ExtrudedFiniteDifferenceGrid`, and the more specific
  `ExtrudedMultiPointGrid` (`Spaces/finitedifference.jl:36`, `extruded.jl:29`,
  `multicolumn.jl:89`).
- `has_topography(::ColumnSpace) = false` is required for the single column (a bare
  `FiniteDifferenceGrid` has no `hypsography` field) and correct for the multi-column
  grid (always `Flat`).
- `horizontal_filter_scale(::ColumnSpace) = Inf`: no ambiguity with the
  `ExtrudedFiniteDifferenceSpace` method because the multi-column space is not a subtype.
- Name collision: `ClimaAtmos.ColumnSpace` (type union) vs
  `ClimaCore.CommonSpaces.ColumnSpace` (constructor). Nothing in `src/` imports
  `CommonSpaces`, so no clash today; tests use the qualified ClimaCore name.

---

## 3. `examples/multi_column/comparison_utils.jl`

### 3.1 [BUG] All vector fields are silently skipped, so velocities were never compared

- `comparison_utils.jl:59` `is_tensor_field(v) = v isa Fields.Field && eltype(v) <: Geometry.AxisTensor`
  runs first in `_field_diffs!` (`:85-86`) and returns before any comparison.
- In this ClimaCore, `Geometry/deprecated.jl:16-17`:
  `const AxisTensor{T, N, B, S} = Tensor{N, T, B, S}` (any rank) and
  `const AxisVector{T, A, S} = Tensor{1, T, Tuple{A}, S}`. Every `AxisVector` is an
  `AxisTensor`, so `Y.c.uₕ`, `Y.f.u₃`, `sgsʲs.u₃`, and every vector-valued cache entry
  (`ᶜu`, `ᶠu³`, `ᶜf³`, fluxes, gradients) are dropped from the comparison.
- Consequences: the "match within rtol 1e-9" claims in the checklist §7 hold for
  scalars only. `physical` (`:48-49`) and the `AxisVector` branch of `is_leaf_eltype`
  (`:57`) are dead code. The header comment ("vector fields compared in physical
  components") describes behavior that never runs.
- Fix: test rank explicitly, e.g. `eltype(v) <: Geometry.AxisTensor{<:Any, 2}` (or
  `Geometry.Axis2Tensor`), and keep vectors on the leaf path.

### 3.2 [BUG] Diagnostic variable name is derived by splitting the file name at the first `_`

- `comparison_utils.jl:139` `name = first(split(file, "_"))`.
- 25 diagnostics have underscores in their short names (`grep 'short_name = "[^"]*_'
  src/diagnostics`), e.g. `nogw_Q_conv`, `nogw_beres_active`, and the generated
  `"$(prefix)_$(field)"` family at `src/diagnostics/*:229`. For those the lookup misses,
  `FieldDiff(NaN, NaN, NaN)` is stored, `passes` returns `false` (`NaN <= rtol` is
  false), and the run is reported as failing for the wrong reason.
- Fix: take the variable name from the dataset (the single non-dimension variable) rather
  than the file name, or strip the known `_<period>_<reduction>` suffix.

### 3.3 [RISK] Errors and non-finite values are swallowed

- `_recurse!` (`:104-115`) wraps every property access in a bare `try/catch` and records
  `NaN`. A genuine exception (e.g. a `MethodError` from a type that cannot be
  columnized) is indistinguishable from a missing property.
- `mask_nonfinite` (`:42`) replaces `NaN`/`Inf` with 0 before comparing. A field that is
  `NaN` in one run and finite-but-tiny in the other compares equal. Uninitialized memory
  (checklist A9) is only caught when the garbage happens to be finite and large.
- `depth > 10 && return diffs` (`:84`) truncates silently.
- Fix: record the exception type in the path key; count non-finite entries and report
  the count; make the depth cutoff emit a warning.

### 3.4 [NIT] Minor

- `FieldDiff(x1, x2)` on length mismatch returns `NaN` (`:35`), which reads as a
  tolerance failure rather than a shape mismatch in the report.
- `mixed_error` `abs_floor = 100 * eps(eltype(x1))` (`:19`) is `1.2e-5` for Float32
  states, i.e. it treats anything below that as absolute; fine, but undocumented.
- `report_diffs` (`:164`) prints a 111-character line (see §1.3).

---

## 4. `examples/multi_column/single_vs_multi_column.jl`

- **[RISK] Both runs write to the same directory if the base config sets `output_dir`.**
  `common` (`:44`) sets `output_dir_style = "removepreexisting"` but not `output_dir`;
  `job_id` differs (`_single` / `_multi`), so the default `output/<job_id>` differs
  (`restart.jl:176`). If a user passes a config with an explicit `output_dir`, the second
  run deletes the first run's NetCDF files before `diagnostic_diffs` runs, and every
  diagnostic is reported as missing. None of the shipped column configs set `output_dir`,
  so this is a footgun rather than a live bug. Set `output_dir` per run in the overlay.
- **[NIT] Dead fallback.** `job_id = something(job_id, "single_vs_multi_column")` (`:42`):
  `commandline_kwargs` always supplies a non-`nothing` `job_id`
  (`cli_options.jl:23-24` default `job_id_from_config_file(default_config_file)`).
- **[NIT] Driver and checklist disagree.** The driver places all three columns at
  `(0, 0)` (`:53-54`); checklist §7 describes the third column at `(30, -50)`. One of
  them should change, or the checklist should say the latitude-30 runs used a different
  overlay.
- **[NIT] Test semantics.** The duplicate-column test uses `rtol = atol = 0` (bitwise) and
  the single-vs-multi test uses `passes = err <= rtol || rmse <= atol`, so a field with a
  large relative error but tiny RMSE passes. Intended for near-zero fields, but worth
  stating in the docstring.
- Verified fine: `CA.AtmosConfig(::Vector; job_id)` matches
  `atmos_config.jl:61,90`; `output_dir_style` is matched case-insensitively
  (`restart.jl:185`); `YAML`, `NCDatasets`, `Test` are in `.buildkite/Project.toml`;
  `AtmosSimulation` has `output_dir` and `integrator` fields
  (`AtmosSimulations.jl:29,33`); ClimaDiagnostics writes no `_FillValue`, so
  `Array(ds[name])` is not `Union{Missing,…}`; the multi-column NetCDF dimension order
  `(time, column, z)` assumed at `:150` matches the ClimaDiagnostics writer.

---

## 5. `MULTI_COLUMN_CHECKLIST.md`

- §2 header says the Atmos changes are "uncommitted"; they are in `12a1c8f7c`.
- §2 last bullet says the ClimaDiagnostics branch "is versioned 0.4.0" and asks for a
  compat bump. `multi-cols/Project.toml:3` says `0.3.10`; `Project.toml:57`
  `ClimaDiagnostics = "0.3.9"` already admits it. No bump needed until upstream actually
  goes to 0.4.
- §1 "columns match the single-column run within rtol 1e-9 in state, cache, and NetCDF
  output": qualify with §3.1 above (vector fields were excluded).
- §4 C1 is a hard prerequisite for §2.1 here, not a nice-to-have; say so.
- Environment line says `julia +1.12`; repo norms in `AGENTS.md` prefer 1.11 locally
  (CI runs 1.10 and 1.11).

---

## 6. Suggested order

1. Un-commit `.buildkite/Manifest.toml`, fix `.gitignore:28`, drop `DIFF_SPACE.md` from
   the PR. (§1.1, §1.4)
2. Add `MultiColumnGrid` to `docs/src/api.md`, `interfaces.md`, and the
   `AtmosSimulation` docstring. (§1.2)
3. Run the formatter. (§1.3)
4. Decide the hyperdiffusion fix: ClimaCore C1 or an Atmos `iscolumn` guard; do not ship
   a default-config crash. (§2.1)
5. Assert a singleton context in `MultiColumnGrid`. (§2.2)
6. Add `multicolumn` rules to `check_case_consistency`: coordinate list validation,
   `deep_atmosphere` warning/error, unused-key warning. (§2.3, §2.4)
7. Fix `is_tensor_field` and the diagnostic name lookup, then re-run the verification
   ladder so the velocity comparisons are real. (§3.1, §3.2)
8. Make `issphere` geometry-based. (§2.5)
9. Update the checklist. (§5)
10. Add a `config: multicolumn` model config, a CI step running the driver, a
    `MultiColumnGrid` fixture in `test/diagnostics/unit_diagnostics.jl:154-155`, and
    `"multicolumn"` in `test/restart.jl:190` (`DIFF_SPACE.md` §7).
