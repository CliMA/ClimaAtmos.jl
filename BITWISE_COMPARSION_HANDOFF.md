# Handoff: why single-column and multi-column runs are not bitwise identical

Session of 2026-09-09 on the `clima` cluster, branch `kp/multi-col`, on top of the work
described in `MULTI_COLUMN_HANDOFF.md` and `MULTI_COLUMN_CHECKLIST.md`. The question:
take the simplest column configuration, compare it against the same column run on the
multi-column grid, and explain why the two are only equal to rounding, not bitwise.

## 1. Short answer

The state at t = 0 is bitwise identical. The difference is created by the vertical metric
terms of the multi-column grid, which are built differently from the single column's:

- The single column is a bare `Grids.FiniteDifferenceGrid` (ClimaCore
  `CommonGrids.ColumnGrid`). Its `LocalGeometry` has `J = Δz`, `WJ = J`, and a 1×1
  `∂x∂ξ = [Δz]` that the `LocalGeometry` constructor pads to `diag(1, 1, Δz)`.
- The multi-column grid is an `ExtrudedFiniteDifferenceGrid` over a `MultiPointGrid`
  (ClimaCore worktree `~/worktree/ClimaCore.jl/col-intp`, `src/Grids/multipoint.jl`).
  Its horizontal `LocalGeometry` carries the sphere metric at each point:
  `∂x∂ξ_h = diag(R·π/180, R·cosd(lat)·π/180)`, `J_h = R²·cosd(lat)·(π/180)²`.
  `Geometry.product_geometry` (`src/Geometry/globalgeometry.jl`) then forms
  `J = J_h · J_v`, `WJ = WJ_h · WJ_v`, and `∂x∂ξ = diag(s_lon, s_lat, Δz)`.
- The `LocalGeometry` constructor (`src/Geometry/localgeometry.jl`) computes
  `∂ξ∂x = inv(∂x∂ξ)` and `gⁱʲ = ∂ξ∂x · ∂ξ∂xᵀ` with StaticArrays' generic 3×3 inverse
  (`StaticArrays/src/inv.jl`, `_inv(::Size{(3,3)}, A)`: cross products divided by the
  determinant). For `diag(a, b, c)` its `[3,3]` entry is `(a / (a·b·c)) · b`, which is
  exactly `1/c` when `a = b = 1` (single column) but not when `a = b = R·π/180 ≈ 1.1e5`
  (multi-column). `∂ξ∂x` is also recomputed on every access (`getproperty(lg, :∂ξ∂x)`).

Measured on the hydrostatic-balance column (63 levels, `deep_atmosphere: false`,
Float64, column at lat = 0, lon = 0), single column vs column 1 of the multi-column grid:

| Metric term | Differs | Size |
|---|---|---|
| `z` (centers and faces) | no | |
| `∂x∂ξ[3,3]` (= Δz) | no | |
| `J`, `WJ` | every level | ratio `J_multi / J_single = 12364311711.4888..12364311711.488802` (= `J_h`, itself not constant to the last bit) |
| `∂ξ∂x[3,3]` | 28 of 63 centers, 31 of 64 faces | up to 2 ulp |
| `gⁱʲ[3,3]` (g³³) | 28 of 63 centers, 31 of 64 faces | up to 3 ulp |
| single: `∂ξ∂x[3,3]` vs `1/∂x∂ξ[3,3]` | no | exact |
| multi: `∂ξ∂x[3,3]` vs `1/∂x∂ξ[3,3]` | 28 / 31 levels | up to 2 ulp |

The multi-column `∂x∂ξ` at every level is
`[111194.92664455874 0 0; 0 111194.92664455874 0; 0 0 Δz]`.

Every vertical operator that uses the metric therefore rounds differently:

- Covariant → contravariant (`CT3(u₃) = g³³ u₃`, used for vertical advection and the
  kinetic energy) through `g³³`;
- Covariant → physical (`WVector(u₃) = ∂ξ∂x[3,3] u₃`) through `∂ξ∂x[3,3]`;
- `DivergenceF2C` through `Jcontravariant3 = J · u³` at the faces divided by the center
  `J` (`Operators/finitedifference.jl`): the factor `J_h` cancels mathematically but not in
  floating point, because `J_h · J_v` is rounded before the subtraction and division;
- the implicit Jacobian blocks (`MatrixFields`) that use `g³³` and `J`.

Operators that carry no metric (`GradientC2F` returning covariant components,
`InterpolateF2C`/`InterpolateC2F`) are expected to be bitwise identical (stage 3 of the
script below tests this; it has not run yet). Because the initial state has `u₃ = 0` and
`uₕ = 0`, all metric-dependent quantities vanish at t = 0, which is why the earlier
session found the t = 0 cache and tendencies identical to 1e-13 and the difference only
appears inside the first step. For configurations with a nonzero `uₕ` (Bomex, GABLS)
there is an additional t = 0 rounding in `Y.c.uₕ`: covariant components are
`∂x∂ξ_hᵀ · (u, v)`, i.e. the physical wind multiplied by `R·π/180`, and the round trip
back to physical components is not exact.

## 2. Environment and how to reproduce

- tmux session `julia-hb` (new; the older `julia-mc*` sessions belong to other work and
  must not be reused). Started with `module load climacommon` and then plain
  `julia --project=.buildkite --history-file=no` (the module provides Julia 1.12.5;
  do not pass `+1.12.5`, juliaup is bypassed). The pane is piped to
  `multi_column_dev/logs/julia-hb.log`.
- `include("multi_column_dev/common.jl")` (loads ClimaAtmos, `build_simulation`,
  `compare.jl`), then `include("multi_column_dev/hb_bitwise.jl")`.
- Poll the log from the last `include` line rather than the pane; the pane scrollback
  holds an older shell `ERROR:` line that breaks a naive `grep "^ERROR"` loop:
  `tail -n +$(grep -n 'hb_bitwise.jl' LOG | tail -1 | cut -d: -f1) LOG`.

Files added in this session (all untracked, none formatted):

- `multi_column_dev/hb_1day.yml`: overlay for `single_column_hydrostatic_balance_ft64.yml`
  (`t_end: 1days`, `deep_atmosphere: false`, no diagnostics, no reproducibility test).
- `multi_column_dev/hb_bitwise.jl`: the diagnostic. Stages, each printing
  `HB STAGE n DONE`:
  1. build `s1` (single) and `sn` (multi-column, one column at (0, 0)); bitwise report on
     `Y` at t = 0. Result: 4 entries, 0 differ.
  2. metric terms, table above. Result: as in §1.
  3. vertical operators on identical synthetic inputs (`CT3`, `WVector`, `DivergenceF2C`
     of `u₃` and of `u³`, `GradientC2F`, both interpolations, and
     `ᶠinterp(q) · CT3(u₃)`), comparing raw components. Not run yet: the first attempt
     failed to parse `exp(-ᶜz1 / 8000)` (Julia does not accept a unary minus before a
     modifier-letter identifier); fixed to `exp(-(ᶜz1 / 8000))`, re-include the file.
  4. cache after `set_precomputed_quantities!`, `implicit_tendency!`, and
     `remaining_tendency!` on the identical initial state, bitwise. Not run yet.
  5. one `SciMLBase.step!` on each integrator, bitwise report of `Y` and `p`; then
     `solve_atmos!` to 1 day, bitwise report and the rtol 1e-9 report. Not run yet.
  Helpers: `bitdiff` (count of non-identical entries, max abs, max ulp distance),
  `bit_diffs`/`report_bits` (walk `Y` or `p` like `field_diffs`), `raw(field, col)`
  (raw components, no conversion to physical components).

## 3. What is left

1. Re-include `hb_bitwise.jl` and record stages 3-5: which operators are bitwise
   identical, which fields differ first after one step, and how the ulp count grows over
   8 steps (expected: a few ulp after one step, growing slowly; the earlier session saw
   rtol 1e-9 agreement after 10 days).
2. Confirm the attribution by construction (REPL only, no source edits): build a
   `Grids.MultiPointGrid(context, global_geometry, local_geometry)` whose horizontal
   `LocalGeometry` is the identity (`J = WJ = 1`, `∂x∂ξ = I₂`), extrude it with
   `Grids.ExtrudedFiniteDifferenceGrid(h_grid, z_grid; deep = false)`, and run the same
   one step. With `a = b = 1` the padded inverse and `J = 1 · J_v` are exact, so the
   prediction is a bitwise-identical run. The simplest way to inject the grid is to
   redefine `CA.MultiColumnGrid(::Type{FT}; ...)` in the REPL to return this grid.
3. Decide with the user whether bitwise identity is a goal. Options, in order of
   preference:
   - Make the horizontal metric of `MultiPointGrid` the identity (as in item 2). Columns
     have no horizontal operators, so the horizontal metric only enters (a) the
     `uₕ` covariant/physical round trip, which then becomes exact, (b) horizontal sums
     (`Fields.sum` weights by `WJ`; with `WJ = 1` each column has unit weight, which is
     arguably right for independent columns), and (c)
     `Spaces.node_horizontal_length_scale` for hyperdiffusion and LES closures, which
     would need its own definition instead of one derived from `J`. This is a ClimaCore
     change in `kp/col-intp` and a design decision.
   - Alternatively keep the sphere metric but make `LocalGeometry` compute `∂ξ∂x` and
     `gⁱʲ` blockwise for block-diagonal `∂x∂ξ`, so `∂ξ∂x[3,3] = 1/Δz` exactly. This
     removes the `g³³` and `∂ξ∂x` differences but not the `J = J_h · J_v` rounding in
     `DivergenceF2C`, so it cannot give bitwise identity on its own.
   - Accept rounding-level agreement and keep the rtol 1e-9 comparison as the CI check
     (what `examples/multi_column/single_vs_multi_column.jl` does now). For EDMF cases the
     single column's own 1-ulp sensitivity is the right yardstick
     (`MULTI_COLUMN_CHECKLIST.md`, "B1 resolved").
4. If the user wants the diagnostic kept, format `hb_bitwise.jl` with the pinned
   JuliaFormatter (`.dev/format`) and consider folding `bitdiff`/`report_bits` into
   `examples/multi_column/comparison_utils.jl`.
