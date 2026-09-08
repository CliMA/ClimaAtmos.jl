# Parent-Budget Ledger: Coverage Registry

Every path that can change an authoritative parent field, with the disposition
it is expected to have and the evidence that would establish it. The
[contract](contract.md) fixes what the dispositions mean, the
[architecture](architecture.md) fixes how a row is collected, and the
[implementation plan](plan.md) says which stack step owns each row.

This table is a **precursor to an executable registry**, not a permanent
artefact. A hand-maintained inventory of dispatches drifts from the code
silently, and a coverage claim resting on a stale table is worth nothing. Stack
step 4 introduces `src/parent_budget/coverage_registry.jl` as the single source
of truth; from that point this page is either generated from the registry or
tested for exact agreement with it, and the event identifiers below are the
registry keys.

The registry is also where a run's schema comes from. For a given configuration
the schema selects the rows whose guard holds and declares them as expectations
before collection begins, which is what lets a row that was expected and never
recorded appear as blocked instead of vanishing. Every column below is therefore
a declaration about the code, not a summary of what some run happened to report.

No supported configuration may claim closure while a row that writes an
authoritative parent field is still `open`.

## How to read a row

Two different things are called a status and they are kept in separate columns.
A row can be mathematically zero and still be uncollected, and reading one as
the other is how an unmeasured term becomes an assumed zero.

**Disposition** — what the implemented equation does to the parent variable.
This is a proof obligation about the code, not a claim about the ledger.

| Disposition | Meaning                                                                  |
|:----------- |:------------------------------------------------------------------------ |
| `measured`  | Not provably zero, so the ledger has to measure it                       |
| `zero`      | Invariant zero, with the proof named in the row's note                   |
| `n/a`       | The path does not write this parent field in any supported configuration |
| `open`      | Not yet established from the code; blocks the affected claim             |

**Collection state** — whether the ledger reads the row at runtime, and how far
that has been checked.

| State       | Meaning                                                             |
|:----------- |:------------------------------------------------------------------- |
| `none`      | Not collected at all                                                |
| `envelope`  | Covered only by the enclosing channel envelope, with no attribution |
| `collected` | Collected as its own contribution                                   |
| `verified`  | Collected, and its evidence has been established by a passing test  |

A `zero` disposition is never by itself evidence that anything was checked at
runtime. Until a row reaches `collected` or `verified`, it is uncollected
whatever its disposition says, and the claim it feeds stays blocked.

**Collection level** — which part of the nested reconciliation the row belongs
to: `envelope`, `decomposition`, `final map`, or `transfer`. An envelope row is
the reference its decomposition rows are reconciled against and is never summed
alongside them. A `final map` row is a direct term in the parent identity and is
not an attribution channel, so recording one creates no requirement for a channel
envelope of its own.

**Topology** — for a transfer row, how its two sides relate. The schema derives
this from the configuration before collection begins. It is never inferred from
whichever legs arrived.

| Topology   | Meaning                                                                |
|:---------- |:---------------------------------------------------------------------- |
| `internal` | Both sides are modeled reservoirs within one control volume            |
| `coupled`  | Both sides are modeled reservoirs, in different control volumes        |
| `exterior` | One modeled side, and a counterparty the model does not carry as state |

An `internal` or `coupled` row requires every declared leg, and its signed sum is
tested for cancellation. An `exterior` row records its modeled leg alone: no
numerical counter-leg is fabricated, no cancellation is tested, and the signed
crossing is reported as a boundary source or sink. A row whose topology depends on
the configuration names both cases, and the schema picks one.

The reservoirs column lists modeled reservoirs only. An exterior counterparty is
not a reservoir, has no state to integrate, and is named in its own column.

Disposition columns describe the effect on the **atmosphere** unless the row's
reservoir column says otherwise.

## Channel envelopes

The primary identity reconciles endpoint change against these envelopes plus the
final maps. Each row here is the complete update one integrator channel applied,
taken from the applied increment and never from endpoint subtraction.

The final maps are **not** in this table. They are terms of the same identity but
they are not channels: a map has no envelope for a decomposition to explain, so
it produces no attribution result and demands none. Their rows are in the **Final
accepted-state maps** section below, and the identity sums this table and that
one, never an aggregate of either alongside its own rows.

| Event id               | Dispatch                         | Guard  | Channel  | Reservoirs | Parent fields                 | Disposition M·W·E              | Proof obligation                                                     | Level    | State | Evidence required                                                     | Test                | Step |
|:---------------------- |:-------------------------------- |:------ |:-------- |:---------- |:----------------------------- |:------------------------------ |:-------------------------------------------------------------------- |:-------- |:----- |:--------------------------------------------------------------------- |:------------------- |:---- |
| `env.explicit_main`    | accepted increment from `Yₜ`     | always | `Yₜ`     | atmosphere | `ρ`, `ρq_tot`, `ρe_tot`       | measured · measured · measured | applied increment equals tableau-weighted stage sum                  | envelope | none  | accepted explicit weights from the pinned tableau                     | `envelope_tests.jl` | 3    |
| `env.explicit_limited` | accepted increment from `Yₜ_lim` | always | `Yₜ_lim` | atmosphere | `ρq_tot`, categories, tracers | measured · measured · measured | limited channel integrated through the limiter, separately from `Yₜ` | envelope | none  | accepted limited-channel increment                                    | `envelope_tests.jl` | 3    |
| `env.implicit`         | accepted increment from `T_imp!` | always | `T_imp!` | atmosphere | `ρ`, `ρq_tot`, `ρe_tot`       | measured · measured · measured | effective implicit increment as the pinned solver forms it           | envelope | none  | stage weights and hook-folding established against the pinned version | `envelope_tests.jl` | 3    |

The algebraic solve defect is **not** a separate envelope. It is part of what the
implicit channel applied, so it appears in `env.implicit` and shows up again as a
term of that envelope's decomposition. An implicit attribution residual can only
close with the defect included.

Neither is the post-implicit correction. `ClimaTimeSteppers` applies
`T_post_imp!` to the Newton-solved stage state and then forms the stored implicit
stage tendency by differencing, so the correction is already inside
`env.implicit`. Booking it as an envelope of its own would count it twice. It is
a decomposition row of the implicit channel, `impl.post_implicit_correction`,
below.

## Explicit limited channel decomposition

| Event id                               | Dispatch                                | Guard                       | Channel  | Reservoirs | Parent fields                 | Disposition M·W·E  | Proof obligation                                                              | Level         | State | Evidence required                         | Test                            | Step |
|:-------------------------------------- |:--------------------------------------- |:--------------------------- |:-------- |:---------- |:----------------------------- |:------------------ |:----------------------------------------------------------------------------- |:------------- |:----- |:----------------------------------------- |:------------------------------- |:---- |
| `lim_chan.horizontal_tracer_advection` | `horizontal_tracer_advection_tendency!` | moist or tracers configured | `Yₜ_lim` | atmosphere | `ρq_tot`, categories, tracers | zero · zero · zero | conservative horizontal divergence, global sum zero                           | decomposition | none  | operator global-zero test on a real state | `explicit_attribution_tests.jl` | 4    |
| `lim_chan.tracer_hyperdiffusion`       | `apply_tracer_hyperdiffusion_tendency!` | `hyperdiff` configured      | `Yₜ_lim` | atmosphere | `ρq_tot`, categories, tracers | zero · zero · zero | conservative, DSS of the `∇²` cache happens inside `hyperdiffusion_tendency!` | decomposition | none  | operator global-zero test on a real state | `explicit_attribution_tests.jl` | 4    |

## Explicit main channel decomposition

| Event id                           | Dispatch                                                    | Guard                                                    | Channel | Reservoirs | Parent fields                          | Disposition M·W·E              | Proof obligation                                                                                      | Level         | State | Evidence required                            | Test                            | Step |
|:---------------------------------- |:----------------------------------------------------------- |:-------------------------------------------------------- |:------- |:---------- |:-------------------------------------- |:------------------------------ |:----------------------------------------------------------------------------------------------------- |:------------- |:----- |:-------------------------------------------- |:------------------------------- |:---- |
| `expl.horizontal_dynamics`         | `horizontal_dynamics_tendency!`                             | always                                                   | `Yₜ`    | atmosphere | `ρ`, `ρe_tot`, `uₕ`                    | zero · n/a · zero              | conservative transport                                                                                | decomposition | none  | operator global-zero test                    | `explicit_attribution_tests.jl` | 4    |
| `expl.explicit_vertical_advection` | `explicit_vertical_advection_tendency!`                     | `diff_mode` and advection config                         | `Yₜ`    | atmosphere | `ρ`, `ρe_tot`, tracers, `u₃`           | zero · zero · zero             | conservative transport, closed vertical boundaries                                                    | decomposition | none  | operator global-zero test                    | `explicit_attribution_tests.jl` | 4    |
| `expl.hyperdiffusion`              | `apply_hyperdiffusion_tendency!`                            | `hyperdiff` configured                                   | `Yₜ`    | atmosphere | `ρe_tot`, `uₕ`, `ρtke`                 | n/a · n/a · zero               | conservative                                                                                          | decomposition | none  | operator global-zero test                    | `explicit_attribution_tests.jl` | 4    |
| `expl.viscous_sponge`              | `viscous_sponge_tendency_*`                                 | `viscous_sponge` configured                              | `Yₜ`    | atmosphere | `uₕ`, `u₃`, `ρe_tot`, tracers, `ρ`     | measured · measured · measured | interior numerical source; the `ρ` leg is added only for the `ρq_tot` tracer                          | decomposition | none  | applied increment per field                  | `explicit_attribution_tests.jl` | 4    |
| `expl.rayleigh_sponge`             | `rayleigh_sponge_tendency_uₕ`                               | `rayleigh_sponge` configured                             | `Yₜ`    | atmosphere | `uₕ`                                   | zero · zero · zero             | momentum only, `ρe_tot` prognostic and untouched                                                      | decomposition | none  | field-write inventory                        | `explicit_attribution_tests.jl` | 4    |
| `expl.held_suarez_drag`            | `held_suarez_forcing_tendency_uₕ`                           | `HeldSuarezForcing`                                      | `Yₜ`    | atmosphere | `uₕ`                                   | zero · zero · zero             | momentum only                                                                                         | decomposition | none  | field-write inventory                        | `explicit_attribution_tests.jl` | 4    |
| `expl.held_suarez_heating`         | `held_suarez_forcing_tendency_ρe_tot`                       | `HeldSuarezForcing`                                      | `Yₜ`    | atmosphere | `ρe_tot`                               | zero · zero · measured         | idealized external heating, no mass or water term                                                     | decomposition | none  | applied increment                            | `explicit_attribution_tests.jl` | 4    |
| `expl.scm_coriolis`                | `scm_coriolis_tendency_uₕ`                                  | single column with SCM Coriolis                          | `Yₜ`    | atmosphere | `uₕ`                                   | zero · zero · zero             | momentum only                                                                                         | decomposition | none  | field-write inventory                        | `explicit_attribution_tests.jl` | 4    |
| `expl.subsidence`                  | `subsidence_tendency!`                                      | `LargeScaleSubsidence`                                   | `Yₜ`    | atmosphere | `ρe_tot`, `ρq_tot`, `ρq_lcl`, `ρq_icl` | zero · measured · measured     | writes no `ρ` term, so the mass contribution is invariant zero                                        | decomposition | none  | applied increment plus field-write inventory | `explicit_attribution_tests.jl` | 4    |
| `expl.large_scale_advection`       | `large_scale_advection_tendency_*`                          | large-scale advection configured                         | `Yₜ`    | atmosphere | `ρe_tot`, `ρq_tot`                     | zero · measured · measured     | writes no `ρ` term                                                                                    | decomposition | none  | applied increment plus field-write inventory | `explicit_attribution_tests.jl` | 4    |
| `expl.external_forcing`            | `external_forcing_tendency!`, `apply_Tq_forcing!`           | external forcing configured                              | `Yₜ`    | atmosphere | `ρe_tot`, `ρq_tot`, `uₕ`               | zero · measured · measured     | writes no `ρ` term                                                                                    | decomposition | none  | applied increment plus field-write inventory | `explicit_attribution_tests.jl` | 4    |
| `expl.vertical_diffusion`          | `vertical_diffusion_boundary_layer_tendency!`               | `diff_mode == Explicit()`                                | `Yₜ`    | atmosphere | `ρe_tot`, tracers                      | measured · measured · measured | interior operator with zero flux at top and bottom faces                                              | decomposition | none  | applied increment                            | `explicit_attribution_tests.jl` | 4    |
| `expl.smagorinsky_lilly`           | `horizontal_/vertical_smagorinsky_lilly_tendency!`          | SGS diffusion configured                                 | `Yₜ`    | atmosphere | `ρ`, `ρe_tot`, tracers                 | measured · measured · measured | diffusive; global zero only if the discrete operator has it                                           | decomposition | none  | applied increment                            | `explicit_attribution_tests.jl` | 4    |
| `expl.amd`                         | `horizontal_/vertical_amd_tendency!`                        | AMD configured                                           | `Yₜ`    | atmosphere | `ρ`, `ρe_tot`, tracers                 | measured · measured · measured | as above                                                                                              | decomposition | none  | applied increment                            | `explicit_attribution_tests.jl` | 4    |
| `expl.constant_diffusion`          | `horizontal_constant_diffusion_tendency!`                   | constant diffusion configured                            | `Yₜ`    | atmosphere | tracers                                | measured · measured · measured | as above                                                                                              | decomposition | none  | applied increment                            | `explicit_attribution_tests.jl` | 4    |
| `expl.microphysics_formation`      | `microphysics_tendency!`, 1M and non-equilibrium            | `NonEquilibriumMicrophysics1M` and explicit microphysics | `Yₜ`    | atmosphere | `ρq_lcl`, `ρq_icl`, `ρq_rai`, `ρq_sno` | zero · zero · zero             | formation redistributes categories inside `ρq_tot` and applies no source to `ρq_tot`, `ρ` or `ρe_tot` | decomposition | none  | field-write inventory                        | `explicit_attribution_tests.jl` | 4    |
| `expl.tracer_nonnegativity_vapor`  | `tracer_nonnegativity_vapor_tendency!`                      | moist                                                    | `Yₜ`    | atmosphere | `ρq_lcl`, `ρq_icl`, `ρq_rai`, `ρq_sno` | zero · zero · zero             | category-only; writes no parent field                                                                 | decomposition | none  | field-write inventory                        | `explicit_attribution_tests.jl` | 4    |
| `expl.non_orographic_gravity_wave` | `non_orographic_gravity_wave_apply_tendency!`               | configured                                               | `Yₜ`    | atmosphere | `uₕ`                                   | zero · zero · zero             | momentum only                                                                                         | decomposition | none  | field-write inventory                        | `explicit_attribution_tests.jl` | 4    |
| `expl.orographic_gravity_wave`     | `orographic_gravity_wave_apply_tendency!`                   | configured                                               | `Yₜ`    | atmosphere | `uₕ`                                   | zero · zero · zero             | momentum only                                                                                         | decomposition | none  | field-write inventory                        | `explicit_attribution_tests.jl` | 4    |
| `expl.zero_velocity`               | `zero_velocity_tendency!`                                   | advection tests                                          | `Yₜ`    | atmosphere | `uₕ`, `u₃`                             | zero · zero · zero             | momentum only                                                                                         | decomposition | none  | field-write inventory                        | `explicit_attribution_tests.jl` | 4    |
| `expl.out_of_scope`                | `edmfx_*`, `pressure_work_tendency!`, `chemistry_tendency!` | out-of-scope configurations                              | `Yₜ`    | —          | —                                      | n/a · n/a · n/a                | excluded by the contract's scope                                                                      | decomposition | none  | configuration refused at setup               | `scope_tests.jl`                | 3    |

## Implicit channel decomposition

| Event id                        | Dispatch                                                     | Guard                                                    | Channel  | Reservoirs | Parent fields                | Disposition M·W·E              | Proof obligation                                                                                                                                                                      | Level         | State | Evidence required                                                           | Test                            | Step |
|:------------------------------- |:------------------------------------------------------------ |:-------------------------------------------------------- |:-------- |:---------- |:---------------------------- |:------------------------------ |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |:------------- |:----- |:--------------------------------------------------------------------------- |:------------------------------- |:---- |
| `impl.vertical_advection`       | `implicit_vertical_advection_tendency!`                      | always                                                   | `T_imp!` | atmosphere | `ρ`, `ρe_tot`, tracers, `u₃` | zero · zero · zero             | conservative transport with closed vertical boundaries; precipitation leaves through a different operator                                                                             | decomposition | none  | operator global-zero test plus accepted implicit weight                     | `implicit_attribution_tests.jl` | 5    |
| `impl.water_fallout`            | `vertical_advection_of_water_tendency!`                      | `NonEquilibriumMicrophysics1M`                           | `T_imp!` | atmosphere | `ρ`, `ρq_tot`, `ρe_tot`      | measured · measured · measured | `ᶜprecipdivᵥ` of the category flux with free outflow at the lower boundary                                                                                                            | transfer      | none  | applied increment with accepted implicit weight                             | `transfer_tests.jl`             | 6    |
| `impl.microphysics_removal_0m`  | `microphysics_tendency!`, 0-moment                           | `EquilibriumMicrophysics0M` and implicit microphysics    | `T_imp!` | atmosphere | `ρq_tot`, `ρ`, `ρe_tot`      | measured · measured · measured | removal straight out of the column, no receiving reservoir                                                                                                                            | transfer      | none  | applied increment with accepted implicit weight                             | `transfer_tests.jl`             | 6    |
| `impl.microphysics_formation`   | `microphysics_tendency!`, 1M                                 | `NonEquilibriumMicrophysics1M` and implicit microphysics | `T_imp!` | atmosphere | categories                   | zero · zero · zero             | redistributes inside `ρq_tot`                                                                                                                                                         | decomposition | none  | field-write inventory                                                       | `implicit_attribution_tests.jl` | 5    |
| `impl.vertical_diffusion`       | `vertical_diffusion_boundary_layer_tendency!`                | `diff_mode == Implicit()`                                | `T_imp!` | atmosphere | `ρe_tot`, tracers            | measured · measured · measured | interior operator, zero flux at top and bottom faces                                                                                                                                  | decomposition | none  | applied increment with accepted implicit weight                             | `implicit_attribution_tests.jl` | 5    |
| `impl.solve_defect`             | Newton stage residual                                        | implicit configurations                                  | `T_imp!` | atmosphere | `ρ`, `ρq_tot`, `ρe_tot`      | measured · measured · measured | leading order at `max_iters = 1`; sign and accepted weight verified                                                                                                                   | decomposition | none  | independent projection of the algebraic residual                            | `solve_defect_tests.jl`         | 5    |
| `impl.post_implicit_correction` | `correct_implicit_advection_tendency!` through `T_post_imp!` | `energy_q_tot_upwinding != Val(:none)`                   | `T_imp!` | atmosphere | `ρe_tot`, `ρq_tot`           | open · open · open             | folded into the effective implicit increment by the stepper; the difference of two vertical divergences with zero boundary flux, so a global zero is expected and not yet established | decomposition | none  | accepted implicit weight `b_imp[i]/γ` on the stage change, one booking only | `implicit_attribution_tests.jl` | 5    |
| `impl.zero_velocity`            | `zero_velocity_tendency!`                                    | advection tests                                          | `T_imp!` | atmosphere | momentum                     | zero · zero · zero             | momentum only                                                                                                                                                                         | decomposition | none  | field-write inventory                                                       | `implicit_attribution_tests.jl` | 5    |
| `impl.out_of_scope`             | `edmfx_*`, `sgs_*`, `pressure_work_tendency!`                | out-of-scope configurations                              | `T_imp!` | —          | —                            | n/a · n/a · n/a                | excluded by the contract's scope                                                                                                                                                      | decomposition | none  | configuration refused at setup                                              | `scope_tests.jl`                | 3    |

Every implicit row's **accepted weight** stays `open` until stack step 5 measures
it. The pinned stepper's coefficients are known: it stores the implicit stage
tendency as `(U − U_start)/dtγ` and the accepted update takes `dt · b_imp[i]` of
it, so a change folded into stage `i` enters with weight `b_imp[i]/γ`, which for
`ARS343` is 2.7726, −1.4784 and 1 at stages 2, 3 and 4. The disposition columns
describe the tendency, not yet the accepted increment.

## Final accepted-state maps

A final map contributes its raw before/after difference on the accepted state.
The same dispatch running on an intermediate stage array is a stage observation
instead and never enters the identity at its raw value.

| Event id                               | Dispatch                                                                   | Guard                           | Channel            | Reservoirs       | Parent fields                                         | Disposition M·W·E              | Proof obligation                                                                                                                             | Level     | State | Evidence required                                               | Test                      | Step |
|:-------------------------------------- |:-------------------------------------------------------------------------- |:------------------------------- |:------------------ |:---------------- |:----------------------------------------------------- |:------------------------------ |:-------------------------------------------------------------------------------------------------------------------------------------------- |:--------- |:----- |:--------------------------------------------------------------- |:------------------------- |:---- |
| `map.quasimonotone_limiter`            | `limiters_func!`, SEM quasimonotone limiter                                | limiter configured              | `lim!`             | atmosphere       | `ρq_tot`, categories, tracers                         | measured · measured · measured | numerical correction, not a physical tendency                                                                                                | final map | none  | ordered before/after pair on the accepted state                 | `final_map_tests.jl`      | 7    |
| `map.rescale_water_tags`               | `limiters_func!`, `rescale_water_tags!`                                    | water tags configured           | `lim!`             | atmosphere       | tag fields only                                       | zero · zero · zero             | tag-only, writes no parent field                                                                                                             | final map | none  | field-write inventory                                           | `final_map_tests.jl`      | 7    |
| `map.mass_energy_consistency`          | `limiters_func!`, `enforce_mass_energy_consistency!`                       | moist                           | `lim!`             | atmosphere       | `ρ`, `ρe_tot`                                         | measured · zero · measured     | moves `ρ` by `Δρq_tot` and `ρe_tot` by `Δρq_tot·(uᵥ(T)+Φ)`                                                                                   | final map | none  | ordered before/after pair                                       | `final_map_tests.jl`      | 7    |
| `map.vertical_mass_borrowing`          | `limiters_func!`, vertical mass borrowing limiter                          | configured                      | `lim!`             | atmosphere       | `ρq_tot`, categories                                  | measured · measured · measured | numerical correction                                                                                                                         | final map | none  | ordered before/after pair                                       | `final_map_tests.jl`      | 7    |
| `map.dss`                              | `dss!`                                                                     | distributed or spectral element | `dss!`             | atmosphere       | `ρ`, `ρq_tot`, `ρe_tot`                               | measured · measured · measured | conservative in exact arithmetic on a closed sphere, so the amount is expected at reduction level and larger is a finding                    | final map | none  | ordered before/after pair, compared against the reduction scale | `final_map_tests.jl`      | 7    |
| `map.prescribe_flow`                   | `constrain_state!`, `prescribe_flow!`                                      | `prescribed_flow`               | `constrain_state!` | atmosphere       | `ρ`, `ρe_tot`, momentum                               | n/a · n/a · n/a                | prescribed overwrite, out of scope, never a physical tendency                                                                                | final map | none  | configuration refused at setup                                  | `scope_tests.jl`          | 7    |
| `map.tracer_nonneg_element_categories` | `constrain_state!`, `tracer_nonnegativity_constraint!` element variant     | `constrain_qtot = false`        | `constrain_state!` | atmosphere       | categories                                            | zero · zero · zero             | the loop skips `ρq_tot`, so no parent field is written                                                                                       | final map | none  | field-write inventory                                           | `final_map_tests.jl`      | 7    |
| `map.tracer_nonneg_element_qtot`       | `constrain_state!`, `tracer_nonnegativity_constraint!` element variant     | `constrain_qtot = true`         | `constrain_state!` | atmosphere       | `ρq_tot`, `ρ`, `ρe_tot`                               | measured · measured · measured | clips `ρq_tot` and hands the increment to `enforce_mass_energy_consistency!`                                                                 | final map | none  | ordered before/after pair                                       | `final_map_tests.jl`      | 7    |
| `map.tracer_nonneg_vapor`              | `constrain_state!`, `tracer_nonnegativity_constraint!` vapour variant      | moist                           | `constrain_state!` | atmosphere       | categories, and `ρq_tot` when `constrain_qtot = true` | zero · measured · zero         | at `constrain_qtot = true` it clips `ρq_tot` without calling `enforce_mass_energy_consistency!`, so water moves while stored energy does not | final map | none  | ordered before/after pair plus a physical-inconsistency note    | `final_map_tests.jl`      | 7    |
| `map.physical_constraints`             | `constrain_state!`, `enforce_physical_constraints!` non-EDMF branch        | moist                           | `constrain_state!` | atmosphere       | categories                                            | zero · zero · zero             | clamps and rescales the condensate fields, reads `ρq_tot` and writes none of `ρ`, `ρq_tot`, `ρe_tot`                                         | final map | none  | field-write inventory                                           | `final_map_tests.jl`      | 7    |
| `map.repair_water_tag_partition`       | `constrain_state!`, `repair_water_tag_partition!`                          | water tags configured           | `constrain_state!` | atmosphere       | tag fields only                                       | zero · zero · zero             | tag-only, sum preserved by construction                                                                                                      | final map | none  | field-write inventory                                           | `final_map_tests.jl`      | 7    |
| `map.restart_transition`               | `handle_restart`                                                           | restart                         | initialization     | atmosphere, slab | `ρ`, `ρq_tot`, `ρe_tot`, `sfc.*`                      | measured · measured · measured | a zero-duration transition of its own, never charged to the next step                                                                        | final map | none  | endpoint pair across the restart boundary                       | `restart_ledger_tests.jl` | 7    |
| `map.initial_state`                    | `Setups.initial_state`, `overwrite_initial_state!`, `overwrite_from_file!` | initialization                  | initialization     | atmosphere, slab | all                                                   | n/a · n/a · n/a                | sets `B⁰`, outside every transaction                                                                                                         | final map | none  | first endpoint recorded as the initial one                      | `restart_ledger_tests.jl` | 7    |

`update_constrain_state_every` decides how often the `constrain_state!` rows
fire. At `"step"` there is one firing per transaction and it is a final map. At
`"stage"` and `"dss"` the same dispatch also fires on intermediate stage arrays,
where it is a stage observation and enters the identity only with its accepted
weight.

## Transfer events

Each of these either moves a quantity between two modeled reservoirs or carries it
out of the modeled system entirely, and the topology column says which. When both
sides are modeled, every declared leg is collected **independently**, from its own
quadrature, never by negating the other, and the signed sum is tested for
cancellation. When the far side is not modeled, the modeled leg is collected, the
counterparty is named, and no counter-leg is invented to make a sum vanish.

| Event id                      | Topology                                    | Modeled legs                                                                              | Exterior counterparty                               | Guard                                | Reservoirs                           | Parent fields                          | Disposition M·W·E              | Proof obligation                                                                    | Level    | State | Evidence required                                            | Test                | Step |
|:----------------------------- |:------------------------------------------- |:----------------------------------------------------------------------------------------- |:--------------------------------------------------- |:------------------------------------ |:------------------------------------ |:-------------------------------------- |:------------------------------ |:----------------------------------------------------------------------------------- |:-------- |:----- |:------------------------------------------------------------ |:------------------- |:---- |
| `xfer.surface_turbulent_flux` | `coupled` with a slab, `exterior` otherwise | `surface_flux_tendency!`, and `surface_temp_tendency!` for a slab                         | unmodeled surface store, when no slab is configured | always                               | atmosphere, and slab when configured | `ρe_tot`, `ρq_tot`, `ρ`, `uₕ`, `sfc.T` | measured · measured · measured | `Yₜ.c.ρ -= btt` is the mass leg; boundary crossing in the atmosphere-only view      | transfer | none  | every declared leg measured separately                       | `transfer_tests.jl` | 6    |
| `xfer.radiation_toa`          | `exterior`                                  | `radiation_tendency!` at the model top                                                    | space above the model top                           | radiation configured                 | atmosphere                           | `ρe_tot`                               | zero · zero · measured         | boundary crossing with no receiving reservoir                                       | transfer | none  | atmospheric leg, cross-checked against the reported TOA flux | `transfer_tests.jl` | 6    |
| `xfer.radiation_surface`      | `coupled` with a slab, `exterior` otherwise | `radiation_tendency!` at the surface, and `surface_temp_tendency!` for a slab             | unmodeled surface store, when no slab is configured | radiation configured                 | atmosphere, and slab when configured | `ρe_tot`, `sfc.T`                      | zero · zero · measured         | separate from the TOA leg, because only one of them has a reservoir on the far side | transfer | none  | every declared leg measured separately                       | `transfer_tests.jl` | 6    |
| `xfer.precipitation_0m`       | `exterior`                                  | `microphysics_tendency!`, 0-moment                                                        | unmodeled surface store                             | `EquilibriumMicrophysics0M`          | atmosphere                           | `ρq_tot`, `ρ`, `ρe_tot`                | measured · measured · measured | removal with no receiving reservoir, so no cancellation is expected in any view     | transfer | none  | atmospheric leg only, exterior counterparty declared         | `transfer_tests.jl` | 6    |
| `xfer.precipitation_1m`       | `coupled` with a slab, `exterior` otherwise | `vertical_advection_of_water_tendency!`, and `surface_precipitation_tendency!` for a slab | unmodeled surface store, when no slab is configured | `NonEquilibriumMicrophysics1M`       | atmosphere, and slab when configured | `ρq_tot`, `ρ`, `ρe_tot`, `sfc.water`   | measured · measured · measured | two quadratures of one physical flux, so the pair is measured and any mismatch kept | transfer | none  | every declared leg measured separately                       | `transfer_tests.jl` | 6    |
| `xfer.slab_qflux`             | `exterior`                                  | `surface_temp_tendency!` Q-flux term                                                      | prescribed ocean heat transport                     | `SlabOceanTemperature` with a Q-flux | slab                                 | `sfc.T`                                | n/a · n/a · measured           | prescribed exterior source into the slab                                            | transfer | none  | slab leg only, exterior counterparty declared                | `transfer_tests.jl` | 6    |

A row that reads `coupled` with a slab and `exterior` otherwise is two different
expectations, not one flexible one. The schema resolves it from the configuration,
and the resolved topology decides whether a cancellation test applies at all.

## Non-authoritative paths

Listed so that no future reader has to rediscover that they were considered.
None of them writes a parent field, so none is ever booked.

| Event id                           | Dispatch                                                                   | Hook               | Why it is not booked                                                                         |
|:---------------------------------- |:-------------------------------------------------------------------------- |:------------------ |:-------------------------------------------------------------------------------------------- |
| `cache.precomputed`                | `set_precomputed_quantities!`                                              | `cache!`           | cache, and the velocity filter on `Y.f.u₃` at the two boundary faces, which is momentum only |
| `cache.implicit_precomputed`       | `set_implicit_precomputed_quantities!`                                     | `cache_imp!`       | cache, and the velocity filter on `Y.f.u₃` at the two boundary faces, which is momentum only |
| `cache.implicit_stage_setup`       | `initialize_implicit_stage_problem!`                                       | `initialize_imp!`  | stage setup                                                                                  |
| `cb.flux_accumulation`             | `flux_accumulation!`                                                       | discrete callback  | mutates `Ref`s in `p` only                                                                   |
| `cb.external_driven_single_column` | `external_driven_single_column!`                                           | discrete callback  | refreshes forcing caches only                                                                |
| `cb.rrtmgp_solver`                 | `rrtmgp_solver_callback!`                                                  | discrete callback  | fills the radiation cache; the state effect arrives through `radiation_tendency!`            |
| `cb.read_only`                     | `nan_checking_callback`, `checkpoint_callback`, `gc_callback`, diagnostics | discrete callbacks | read-only with respect to `Y`                                                                |

A custom callback outside this list is unsupported unless it declares itself
read-only or supplies its own accounting, and an undeclared one fails the
configuration at setup.

## Proof obligations that need more than a cell

**Momentum-only rows.** Every `zero` in the energy column of a drag or sponge
row is exact by construction: `ρe_tot` is prognostic, and a tendency that writes
only `uₕ` or `u₃` leaves it untouched, so the total-energy contribution is
exactly zero and the diagnosed kinetic-energy change is balanced by an equal and
opposite diagnosed internal-energy change. That is a statement about the
implemented equations. Where a scheme is meant to deposit frictional heat there
is no implemented term doing it, and that belongs in the contract's limitations
register, never in a numerical residual.

**Forcing rows.** `expl.subsidence`, `expl.large_scale_advection` and
`expl.external_forcing` write `ρq_tot` and `ρe_tot` and no `ρ` term, so their
mass disposition is an invariant zero and the dry-air budget of a forced run is
open by construction. A mass contribution is never manufactured from the water
tendency.

**Interior diffusion against the surface flux.**
`vertical_diffusion_boundary_layer_tendency!` is an **interior** operator:
`ᶜdiffdivᵥ` sets zero flux at the top and bottom faces, so it carries no
boundary condition. `surface_flux_tendency!` carries the surface boundary flux.
The diffusion row is `measured` because the discrete interior integral is not
exactly zero, not because anything crosses a boundary.

**One-moment fallout against one-moment formation.** Formation
(`microphysics_tendency!` in the 1M and non-equilibrium branches) only
redistributes categories inside `ρq_tot` and applies no source to `ρq_tot`, `ρ`
or `ρe_tot`, so it is invariant zero for all three parents. Fallout is a
different path: `vertical_advection_of_water_tendency!` is called from
`implicit_tendency!`, so it is always on the implicit channel and carries
implicit accepted-stage weighting, and its lower boundary is open, which is where
1M water leaves the atmosphere.

**Zero-moment removal.** `microphysics_tendency!` in the 0M branch is a direct
sink out of the column with no receiving reservoir. Which channel it is on
follows `microphysics_tendency_timestepping`, which defaults to implicit, so the
row appears under the implicit channel and the explicit variant is the
configured alternative rather than a second event.

**Category-only constraints.** A constraint that writes only `ρq_lcl`, `ρq_icl`,
`ρq_rai` and `ρq_sno` is invariant zero for `ρ`, `ρq_tot` and `ρe_tot`. The
vapour it draws on is implicit in `q_tot` minus the categories, not a field it
writes.

**The accepted aggregate is an envelope.** The rows under *Channel envelopes*
are what the decomposition rows are reconciled against. They are never summed
alongside their own decomposition, and an envelope may stand in for attribution
that does not exist yet.

**What the stored implicit tendency folds in.** For `ClimaTimeSteppers` 0.10.6 an
implicit stage records `U_start` before `initialize_imp!`, then runs
`initialize_imp!`, a DSS, the `WithDSS` constraint firing, the Newton solve,
`T_post_imp!`, a DSS and the `EndOfStage` constraint firing, and only then stores
`T_imp[i] = (U − U_start)/dtγ`. Every one of those changes is inside the
effective implicit increment and none may be booked again on its own. The DSS
and constraint that run on the assembled stage value before `U_start` is taken,
and the stage-level `lim!`, are not inside it: they reach the endpoint only
through the tableau and stay stage observations.

**The velocity filter in the cache hooks.** `set_implicit_precomputed_quantities!`,
which `set_precomputed_quantities!` also calls, rewrites `Y.f.u₃` at the bottom
and top faces so that the contravariant vertical velocity vanishes there
(`set_velocity_at_surface!`, `set_velocity_at_top!`). It runs at every `cache!`
and `cache_imp!` call, including at initialization before `B⁰` is read and inside
every implicit stage, where the stepper folds it into the effective implicit
increment. It writes momentum and nothing else, so its mass, water and energy
contributions are exactly zero by construction. It is on record so that a future
change which made it touch `ρ` is seen to need a row.

## Open gaps and what they block

| Gap                                                                                         | Blocks                                                                      | Cleared by   |
|:------------------------------------------------------------------------------------------- |:--------------------------------------------------------------------------- |:------------ |
| `Yₜ_lim` unread by any adapter                                                              | water closure, and energy closure wherever a limited tracer carries energy  | stack step 3 |
| Accepted implicit stage weights unestablished                                               | claim level 2 for implicit terms                                            | stack step 5 |
| Accepted weights of the hooks folded into the effective implicit increment not yet measured | claim level 3 for the implicit channel, and `impl.post_implicit_correction` | stack step 5 |
| Coupled surface legs unmeasured                                                             | claim level 4 in the coupled view                                           | stack step 6 |
| Energy leg of `map.tracer_nonneg_vapor` at `constrain_qtot = true`                          | energy closure wherever that variant is configured                          | stack step 7 |
| This table is not generated from an executable registry                                     | any claim that coverage is complete                                         | stack step 4 |

None of these may become `zero` by assumption. Each is turned into `measured` or
`zero` by the stack step that owns it, with the evidence its row names.
