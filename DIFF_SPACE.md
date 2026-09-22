# Single column vs multi-column space in ClimaAtmos: static analysis

Date: 2026-09-09. Branch: `kp/multi-col` (ClimaAtmos), with uncommitted working-tree
changes included. ClimaCore is the dev'd worktree
`ClimaCore.jl/col-intp` (branch `kp/col-intp`).
ClimaDiagnostics resolves (via `.buildkite/Manifest.toml`) to the dev'd worktree
`ClimaDiagnostics.jl/multi-cols` (branch
`kp/multi-cols`), which already has multi-column writer methods; the registered
release does not.

All line numbers were verified against the files on disk at the time of writing.
"SC" = single column, "MC" = multi-column.

---

## 1. How the two spaces differ in ClimaCore

| | SC: `Spaces.FiniteDifferenceSpace` | MC: `Spaces.MultiColumnFiniteDifferenceSpace` |
|---|---|---|
| Grid | bare `FiniteDifferenceGrid` from a 1-D `IntervalTopology` (`Grids/finitedifference.jl:51-80`) | `ExtrudedFiniteDifferenceGrid{MultiPointGrid, FiniteDifferenceGrid}` (`Grids/extruded.jl:47-97`, `Grids/multipoint.jl`) |
| Global geometry | `CartesianGlobalGeometry()`, hard-coded | `ShallowSphericalGlobalGeometry(radius)`, always shallow: `CommonGrids.MultiColumnGrid` never forwards `deep` (`CommonGrids.jl:775-794`) |
| Coordinates | `ZPoint` | `LatLongZPoint` |
| Local geometry axes | `I = (3,)`; horizontal metric is identity padding | `I = (1,2,3)`; horizontal `∂x∂ξ = diag(R·π/180, R·cosd(lat)·π/180)`, `J = J_h · J_v` |
| Horizontal space | `Spaces.horizontal_space` returns `level(space, 1)`, a `PointSpace` | `MultiPointSpace`; `topology` **errors**, `quadrature_style` is `nothing`, no DSS, no `node_horizontal_length_scale`, no `all_nodes` |
| Data layout | `VIJFH{…, Nv, 1, 1, 1}` | `VIJFH{…, Nv, 1, 1, N}` |
| `Spaces.column(space, i)` | itself | a `FiniteDifferenceSpace` wrapping a `ColumnGrid` view; keeps spherical global geometry, lat/lon/z coords, `I = (1,2,3)`, and the `R²·cosd(lat)` factor in `J` |
| Supertype | `AbstractSpace` | `AbstractSpace` only. **Not** `<: FiniteDifferenceSpace`, **not** `<: ExtrudedFiniteDifferenceSpace`. Method signatures written for those two types miss MC. |

Can they be made identical without ClimaCore changes? No. The inner constructors are
open (`MultiPointGrid(context, CartesianGlobalGeometry(), lg)` and the four-argument
`ExtrudedFiniteDifferenceGrid(h, v, Flat(), global_geometry)`), so a Cartesian MC grid
can be hand-built, but `product_geometry` always yields `I = (1,2,3)` and 3-D
coordinates, `Meshes.domain(::MultiPointGrid)` and the HDF5 writer unconditionally read
`.radius`, and cache keys differ. In practice this does not matter for vertical
numerics: under shallow geometry `J_h` is constant along a column and cancels in every
vertical FD operator.

Key ClimaCore facts that drive the findings below:

- `Operators.horizontal_dims(arg)` (`Operators/spectralelement.jl:193-197`) returns `()`
  when `nquadpoints == 1`. Both SC and MC layouts have `Ni = Nj = 1`, so horizontal
  spectral operators (`divₕ`, `gradₕ`, `curlₕ`, `wdivₕ`, `wgradₕ`, `wcurlₕ`) are **zero
  no-ops on both**, and the derivative matrix (which would need a quadrature) is never
  requested. This is why existing column configs run with hyperdiffusion left on.
- `node_horizontal_length_scale` has methods for `AbstractSpectralElementSpace`,
  `PointSpace` (returns 1), and `Nothing` (returns 1). None for `MultiPointSpace`.
- `all_nodes` has methods for `PointSpace` and `SpectralElementSpace2D` only.
- `has_vertical(::MultiColumnFiniteDifferenceSpace) = true`, so `Fields.field2array`
  returns `(nlevels, ncolumns)` for MC; on a `MultiPointSpace` level field it returns
  `vec(parent(field))`. Fine.
- `InputOutput` writers/readers handle `MultiPointGrid` (`writers.jl:444-453`,
  `readers.jl:505-513, 644-650`). Restart works.
- `MultiPointGrid` is single-process only (`SingletonCommsContext`). MC + MPI is
  unsupported.

---

## 2. Working-tree changes on this branch (context)

- `config/default_configs/default_config.yml`: `config` accepts `"multicolumn"`; new
  `column_latitudes` / `column_longitudes` (default `[0.0]`).
- `src/config/type_getters.jl:509-549`: topography kwargs skipped for
  `config ∉ ("column", "multicolumn")`; `"multicolumn"` builds `MultiColumnGrid` with
  `radius = CAP.planet_radius(params)`; length check on the two coordinate lists.
- `src/config/model_getters.jl:1040`: `"multicolumn"` added to `valid_configs`.
- `src/simulation/grids.jl:132-171`: new `MultiColumnGrid(FT; points, radius, ...)`
  wrapper (no `deep_atmosphere` kwarg); `get_spaces` now generic via `Spaces.space`.
- `src/utils/utilities.jl`: new
  `const ColumnSpace = Union{FiniteDifferenceSpace, MultiColumnFiniteDifferenceSpace}`;
  `iscolumn`, `has_topography`, `do_dss`, `horizontal_filter_scale` widened to it.
- `src/parameterized_tendencies/sponge/viscous_sponge.jl:38`: `uₕ` sponge guard now
  `iscolumn(...)`.
- `src/prognostic_equations/implicit/autodiff_utils.jl:98-99`: `column_index_iterator`
  handles `MultiPointSpace` as `(1, 1, h)`.

---

## 3. Hard errors on MC that SC survives

| # | Site | SC | MC | Fix |
|---|---|---|---|---|
| E1 | `src/parameterized_tendencies/radiation/radiation.jl:442` `ncol = length(Spaces.all_nodes(axes(Spaces.level(Y.c, 1))))` | `PointSpace` → `(1,)` → 1 | `MultiPointSpace` → **MethodError**. Blocks all RRTMGP radiation. | `Spaces.ncolumns(axes(Y.c))` |
| E2 | `src/prognostic_equations/hyperdiffusion.jl:22` `h = Spaces.node_horizontal_length_scale(Spaces.horizontal_space(axes(Y.c)))` in `ν₄` | `PointSpace` → 1 m placeholder; operators are zero, so hyperdiffusion is a silent no-op | **MethodError**. `hyperdiff` defaults to `"Hyperdiffusion"` and most `*_column.yml` configs leave it on, so a default MC run crashes here. Reached from `apply_hyperdiffusion_tendency!` (:255) and `apply_tracer_hyperdiffusion_tendency!` (:464). | route through `horizontal_filter_scale`, or `iscolumn` guard / `ColumnSpace` method for `ν₄` |
| E3 | `src/parameterized_tendencies/les_sgs_models/smagorinsky_lilly.jl:96-97`; `anisotropic_minimum_dissipation.jl:69-70, 210-211` | 1 | **MethodError** (same pattern as E2). Opt-in closures. | same as E2 |
| E4 | `src/config/model_getters.jl:853` `@assert parsed_args["config"] == "column"` for `ReanalysisMonthlyAveragedDiurnal` | allowed | **AssertionError**. Only hard config-string gate that excludes `"multicolumn"`. | decide; also note `site_latitude`/`site_longitude` are scalars, no per-column forcing exists |
| E5 | `src/utils/utilities.jl:947-949` `issphere`: `Meshes.domain(Spaces.topology(Spaces.horizontal_space(space)))` | errors too (`topology(PointSpace)` has no method) | **errors** (`MultiPointGrid has no topology`). Latent: only unreachable because `iscolumn` is tested first at `non_orographic_gravity_wave.jl:140,353`. Also errors on a `bycolumn` slice of MC. | use `Spaces.global_geometry(space) isa AbstractSphericalGlobalGeometry` |
| E6 | `src/diagnostics/core_diagnostics.jl:164` `compute_rv`: unconditional `Spaces.weighted_dss!(vort)` (no `do_dss` guard) | MethodError on `VF` layout | matches the `VIJFH` `weighted_dss!` method, then dies inside on `topology(::MultiPointGrid)`. Deeper failure. Not a default diagnostic. | guard with `do_dss` |
| E7 | `src/parameterized_tendencies/gravity_wave_drag/orographic_gravity_wave.jl:70-72, 93` `Spaces.topology(Spaces.horizontal_space(axes(Y.c))).mesh.domain.radius`; `:1226` `n_elements_per_panel_direction`; `orographic_gravity_wave_helper.jl:649`; `preprocess_topography.jl:106` | errors | errors. OGW is sphere-only. | read radius from `Spaces.global_geometry` if MC support is wanted |
| E8 | `src/utils/utilities.jl:793-800` `horizontal_integral_at_boundary` asserts `FaceExtrudedFiniteDifferenceSpace` and `SpectralElementSpace2D`. Callers: `callbacks.jl:42,46` (`check_conservation` + radiation), `solve.jl:225-248`, `conservation_diagnostics.jl:61,83` | AssertionError | AssertionError. Parity; pre-existing. | keep `check_conservation` off for both |

Order a default MC run hits them: E2 (unless `hyperdiff: ~`) → E1 (if radiation on).
NetCDF output is **not** a blocker in this environment: the dev'd ClimaDiagnostics
worktree defines `default_num_points`, `NetCDFWriter`, `target_coordinates`, etc. for
`MultiColumnFiniteDifferenceSpace` / `MultiPointSpace` (`netcdf_writer_coordinates.jl:746-1036`,
`netcdf_writer.jl:158-689`). With the registered ClimaDiagnostics release it would fail
at `src/simulation/AtmosSimulations.jl:82,91`.

---

## 4. Physics silently enabled/disabled differently

### 4.1 Dispatch on coordinate type (`eltype(coords) <: LatLongZPoint`)

| # | Site | SC (`ZPoint`) | MC (`LatLongZPoint`) |
|---|---|---|---|
| P1 | `src/cache/cache.jl:317-343` `compute_coriolis` | f-plane: `ᶜf³ = f_plane_coriolis_frequency(params)` (ClimaParams default **0**), `ᶠf¹² = nothing` | `2Ω sin(lat)` per column, `ᶠf¹² = nothing` (shallow branch, because global geometry is Shallow, see §5). Equal only at lat = 0. |
| P2 | `src/callbacks/callbacks.jl:203-206` `IdealizedInsolation`; `:246-273` `TimeVaryingInsolation`; `src/parameterized_tendencies/radiation/radiation.jl:153-157, 198-202` (also `optical_thickness_parameter` at :160) | latitude forced to 0 ("flat space is on Equator"); `TimeVaryingInsolation` uses `(0, 0)` unless `tvi.latitude/longitude` set | real per-column lat/lon. Caveat: an explicit `tvi.latitude/longitude` (set for `ForcingFromFile` / ARM VARANAL in `src/config/type_getters.jl:239-243`) is applied with `Ref(...)`, i.e. **one site value broadcast to every column**, overriding the grid coordinates. `src/types.jl:452` doc ("flat-space columns") is now only true for `config: column`. |
| P3 | `src/setups/Setups.jl:204-215` `zonally_symmetric_temperature` (default SST) | generic method: 300 K | `LatLongZPoint` method: `271 + 29·exp(-lat²/(2·26²)) - 6.5e-3·z` |
| P4 | `src/setups/common/prognostic_variables.jl:324-329` slab-ocean initial T (`:lat in propertynames`) | 300 K | Gaussian in latitude |
| P5 | `src/setups/DecayingProfile.jl:39-47` `_temperature_perturbation` | generic: 0 | `0.1·sind(long)·(z < 5000)`: columns perturbed differently by longitude |
| P6 | `src/setups/RCEMIPIIProfile.jl:70-88` SST | no `ZPoint` method → **MethodError** | latitude branch works |
| P7 | `src/parameterized_tendencies/radiation/held_suarez.jl:200` `coordinate_field(ᶜspace).lat` | errors (no `.lat`) | works |
| P8 | `src/prognostic_equations/surface_temp.jl:117` slab Q-flux `coordinate_field(Y.f).lat` (only if `q_flux_enabled`) | errors | works |
| P9 | `src/setups/DryBaroclinicWave.jl:161-175`, `MoistBaroclinicWave.jl:80-100` `(; z, lat, long) = coords` | errors | works; uses `CAP.planet_radius(params)`, not the grid radius |
| P10 | `src/prognostic_equations/advection.jl:227-231` and `hyperdiffusion.jl:316` `point_type <: Abstract3DPoint` → `wcurlₕ(...)` else zero | zero branch | `LatLongZPoint` is 3-D → `wcurlₕ` branch, which evaluates to **zero** (no-op, §1). Same result, wasted work; not an error. |
| P11 | `src/cosp/subcol.jl:256-272` `_coord_hash` | same seed everywhere | per-column seeds. Fine. |

### 4.2 Dispatch on `iscolumn` (now the `ColumnSpace` union)

| # | Site | Effect |
|---|---|---|
| I1 | `src/parameterized_tendencies/gravity_wave_drag/non_orographic_gravity_wave.jl:140, 353` `if iscolumn(...) ... elseif issphere(...)` | MC now takes the **column** branch: source level from `gw_source_height`, uniform `Bw/Bn/cw/flag`, latitude-dependent tropical tuning at `:193-200` skipped, even though `.lat` is available. Fragile: reordering the branches would hit E5. |
| I2 | `src/prognostic_equations/edmfx_sgs_flux.jl:440` horizontal SGS diffusive flux | disabled for MC too (would have been a zero no-op anyway). Fine. |
| I3 | `src/parameterized_tendencies/sponge/viscous_sponge.jl:38` `uₕ` sponge | `NullBroadcasted` for MC. Fine. The `u₃`/`ρe_tot`/tracer sponges at `:64, :79, :101, :116` are not guarded and silently evaluate to zero on both. Inconsistent but harmless. |
| I4 | `src/utils/utilities.jl:427` `horizontal_filter_scale = Inf`; consumers `eddy_diffusion_closures.jl:846`, `edmfx_diagnostics.jl:918`, `utilities.jl:454` | same for SC and an N-column MC. Intended for the ensemble use case? |
| I5 | `src/utils/utilities.jl:511` `has_topography = false` | required for SC (bare grid has no `hypsography`); MC generic path would also work. Feeds `AtmosSimulations.jl:112` so neither emits `orog`. |

### 4.3 Horizontal metric assumptions

ClimaCore injects the local geometry into one-argument axis-tensor conversions inside
field broadcasts (`Fields/broadcast.jl:386-395`), so most `C12(UVVector(...))` patterns
are metric-aware. On SC the horizontal metric is identity, so a missing local geometry
is invisible; on MC the components differ from physical by ~R·π/180 (~1e5).

- `src/prognostic_equations/scm_coriolis.jl:45`
  `ᶜuₕ_g = @. lazy(C12(Geometry.UVVector(prof_ug(ᶜz), prof_vg(ᶜz))))`: relies on
  injection surviving `lazy`. Recommend explicit `C12(UVVector(...), ᶜlg)` as in
  `external_forcing.jl:180, 630`. Also uses a scalar `coriolis_param`, not `2Ω sin(lat)`.
- `non_orographic_gravity_wave.jl:931-932`, `orographic_gravity_wave.jl:314-315`:
  `Covariant12Vector.(UVVector.(...))` without explicit lg; relies on injection. Fine
  but worth a comment.
- `setups/common/prognostic_variables.jl:46`, `external_forcing.jl:180, 630`,
  `surface_conditions.jl:368-372`: explicit local geometry. Fine.
- `src/utils/utilities.jl:482-503` `g³ʰ` errors for `I = (3,)`; only caller
  (`manual_sparse_jacobian.jl:745-748`) is gated by `has_topography = false`. Fine, fragile.

### 4.4 Columns extracted from MC (`Spaces.column` / `bycolumn`)

They are `FiniteDifferenceSpace`-typed, so `iscolumn`, `has_topography`, `do_dss`,
`horizontal_filter_scale`, and the viscous-sponge / EDMF guards all treat them like SC.
But their coordinates are `LatLongZPoint`, so every §4.1 site treats them like the
sphere. `autodiff_utils.jl:83` (`space isa FiniteDifferenceSpace ? nothing : ...`) is
the only site assuming "FiniteDifferenceSpace ⇒ no lat/lon"; harmless there. `issphere`
(E5) errors on such a slice.

---

## 5. Deep vs shallow atmosphere

**The grid and the flag are decoupled.**

| Mechanism | Site | Reads |
|---|---|---|
| Grid construction, sphere | `src/simulation/grids.jl:53, 67-71` | `deep_atmosphere` kwarg → Deep/Shallow global geometry |
| Grid construction, multi-column | `src/simulation/grids.jl:153-171`, `type_getters.jl:537-549` | **no `deep_atmosphere` kwarg**; ClimaCore `CommonGrids.MultiColumnGrid` never forwards `deep` → always `ShallowSphericalGlobalGeometry` |
| Grid construction, column | `src/simulation/grids.jl:101-123` | Cartesian; neither |
| Coriolis / dynamics | `src/cache/cache.jl:322` → `advection.jl:262` `isnothing(ᶠf¹²)` | global geometry **type**; MC and SC both shallow |
| Radiation flux scaling | `src/config/model_getters.jl:613-651` → `RRTMGPInterface.jl:399, 598-605` | `parsed_args["deep_atmosphere"]` (Bool on the radiation mode) **and** `:planet_radius in keys(dict)` |
| Radiation diagnostics | `src/diagnostics/radiation_diagnostics.jl:22, 52-85` | `radiation_mode.deep_atmosphere` with `CAP.planet_radius(params)` (parameters, not grid) |
| Baroclinic-wave IC | `src/config/type_getters.jl:276-279` | `parsed_args["deep_atmosphere"]` → `DryBaroclinicWave(deep_atmosphere=...)` |
| Geopotential | `src/utils/utilities.jl:84` `geopotential(grav, z) = grav * z` | nothing; shallow for everyone (pre-existing inconsistency with a deep `SphereGrid`) |

Default: `config/default_configs/default_config.yml:439-441` `deep_atmosphere: true`.

Consequences with the default config on MC:

- D1. Geometry is silently shallow; no warning.
- D2. **Radiation gets deep scaling on shallow dynamics.** `radiation.jl:149-150, 163-168,
  195-196, 327-334` pass `planet_radius` for any `AbstractSphericalGlobalGeometry`,
  which Shallow satisfies, and the mode has `deep_atmosphere = true`, so RRTMGP applies
  `((z + R)/R)⁻²`. SC is Cartesian, never passes a radius, so the flag is inert there.
  This is a genuine SC-vs-MC physics difference under identical config.
- D3. `radiation_diagnostics.jl` mirrors D2 using the parameter radius.
- D4. Baroclinic-wave setups initialize a deep analytic state onto a shallow metric.
  Setup defaults (`DryBaroclinicWave.jl:15`, `MoistBaroclinicWave.jl:23,43`) are
  `deep_atmosphere = false`, disagreeing with the config default.

Suggested fix: in `check_case_consistency` (`model_getters.jl:1037-1043`), reject or
warn on `config == "multicolumn" && parsed_args["deep_atmosphere"]`, or add a `deep`
path through `MultiColumnGrid` (requires forwarding `deep` in ClimaCore's
`CommonGrids.MultiColumnGrid`).

---

## 6. Verified as the same / fine

- Horizontal spectral operators: zero on both (`horizontal_dims == ()`), including
  `viscous_sponge.jl:47-116`, `advection.jl:49-91`, `hyperdiffusion.jl:141-152`,
  `constant_horizontal_diffusion.jl:36-41`, LES closures (modulo E3).
- `do_dss = false` on both; all `weighted_dss!` / `create_dss_buffer` calls are
  guarded except E6.
- `autodiff_utils.jl:98-99` MC column iterator; `auto_dense_jacobian.jl:89`
  `Fields.ncolumns`; `jacobian.jl:110` `first_column_view`.
- Restart write/read (`callbacks.jl:290-313`, `restart.jl:29-31`).
- `Spaces.level`, `Fields.field2array`, `Fields.array2field`, `Spaces.z_max`,
  `center_space`/`face_space`, `nlevels`, `undertype`: all have MC methods.
- `core_diagnostics.jl:724-727` `orog` on flat grids: works on MC (`zeros(MultiPointSpace)`),
  errors on SC; not a default diagnostic for either.
- `src/column_datasets/ColumnDatasets.jl:639-651, 679-681`: broadcasts one forcing
  profile / scalar onto all columns. Mechanically fine; semantically single-site.
  `external_forcing.jl:319, 414, 438` `TimeVaryingInput(path, name, target_space)` on an
  MC space is untested (regridder picked from coordinate type).

---

## 7. Coverage gaps

- No `config: multicolumn` yaml, buildkite job, unit test, or reproducibility test
  anywhere. Only hits outside `src/` are the `default_config.yml` help strings.
- `test/restart.jl:190` `configurations = ["sphere", "box", "column"]`;
  `test/restart_AtmosSimulation.jl:316, 375-376` grid list has no `MultiColumnGrid`
  (and would label one `"column"`).
- `test/diagnostics/unit_diagnostics.jl:154-155` grid fixtures: natural place for a
  `MultiColumnGrid` fixture that would catch E1/E2.
- No test of `check_case_consistency`'s `valid_configs`.
- `docs/src/api.md:41-45` lists the grid constructors without `MultiColumnGrid`;
  `docs/src/interfaces.md:33-35` has no multi-column row.
- Name collision to consider: ClimaCore `CommonSpaces.ColumnSpace` (constructor) vs new
  `ClimaAtmos.ColumnSpace` (type union). No actual clash in `src/` (only
  `CommonGrids.*` is used qualified), but confusable.

---

## 8. Priority list

1. E1 `radiation.jl:442` → `Spaces.ncolumns`.
2. E2/E3 `node_horizontal_length_scale ∘ horizontal_space` → `horizontal_filter_scale`
   or `iscolumn` guard.
3. D1–D4: consistency check for `deep_atmosphere` with `multicolumn`.
4. E4 `model_getters.jl:853`: decide whether `multicolumn` is allowed.
5. I1: decide whether MC should take the column or sphere NOGW branch; fix E5 `issphere`
   to use global geometry regardless.
6. P2 caveat: site lat/lon override vs per-column coordinates.
7. Add a multicolumn config, CI job, restart test entry, and API docs.
