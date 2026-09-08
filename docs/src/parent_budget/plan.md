# Parent-Budget Ledger: Implementation Plan

The sequence of work that builds the parent-budget ledger. The
[contract](contract.md) fixes what may be claimed, the
[architecture](architecture.md) fixes the shapes, and the
[coverage registry](coverage.md) inventories the paths. This page says in what
order they are built and what each step establishes.

Work is organized into **stack steps**, numbered 1 to 8. Each step is one pull
request and each carries the actual PR number once it is opened. A step
establishes exactly one claim, has its own tests, and is finished only when its
definition of done holds.

| Stack step | Pull request                                                           | Title                                                                   | Claim it establishes                                                            |
|:---------- |:---------------------------------------------------------------------- |:----------------------------------------------------------------------- |:------------------------------------------------------------------------------- |
| 1          | [#48](https://github.com/johannespletzer/ClimaAtmosResiDyn.jl/pull/48) | Specify the parent-budget closure contract and coverage model           | none; it defines the claims                                                     |
| 2          | [#49](https://github.com/johannespletzer/ClimaAtmosResiDyn.jl/pull/49) | Add the internal parent-budget journal and endpoint-reconciliation core | none in a simulation; the core's own invariants hold                            |
| 3          | not yet opened                                                         | Capture accepted parent-budget update envelopes                         | accepted-state reconciliation                                                   |
| 4          | not yet opened                                                         | Attribute explicit parent-budget contributions                          | process attribution, explicit channels                                          |
| 5          | not yet opened                                                         | Attribute implicit and post-implicit parent-budget contributions        | implemented-update accounting, and process attribution for the implicit channel |
| 6          | not yet opened                                                         | Account for boundary fluxes and reservoir transfers                     | transfer consistency                                                            |
| 7          | not yet opened                                                         | Account for final maps, restarts, and callbacks                         | accepted-state reconciliation across finalization and restart                   |
| 8          | not yet opened                                                         | Add parent-budget reporting and closure certification                   | the published claim certificate                                                 |

**Tests accompany every step.** The reporting step is where results are
published, not where realistic tests begin. A step that cannot test the claim it
makes does not make that claim.

## What this work is not

  - Not a fixer. The ledger never changes the trajectory, and that is tested
    bitwise rather than asserted.
  - Not a claim about configurations outside the contract's supported list.
  - Not source-tag closure. `ρq_tag_*`, `ρe_src_*` and `ρe_tag_*` are excluded
    from the parent identity, and their sum-to-parent properties are a separate
    audit. A finished parent ledger may later help explain a tag residual; tag
    closure may never be used to certify the parent.
  - Not a claim of physical completeness. `D = M − W` is a derived dry-air
    diagnostic, not a conservation invariant, and a forced run has an open
    dry-air budget by construction.
  - Not a component-energy, column, or local budget.

## What is reused from the process-record work

  - The prognostic, untransported `prc_e_<process>` and `prc_q_<process>` fields
    and their restart persistence, as a model for how a diagnostic field
    survives a checkpoint.
  - The snapshot-and-difference bracket pattern around explicit process blocks.
  - The timestepper integration of a record's tendency, which is what makes a
    record a time integral rather than a sum that scales with `dt`.

What is **not** reused, and must not be implied: a process record is not a
closed budget and is never a closure leg. It is a partial signed history of
configured, bracketed processes, and its bracket set is not the ledger's
coverage set. Brackets exist only where `snapshot_tags!` and `attribute_tags!`
are called, which is the explicit path and one implicit water block. The
`e_prc_*` and `q_prc_*` diagnostics also divide by current density; only the raw
`prc_e_*` and `prc_q_*` state is extensive.

## Stack step 1 — Contract and coverage model

**Claim.** None. This step defines the claims the later steps make.

**Scope.** The [contract](contract.md), the [architecture](architecture.md) and
the [coverage registry](coverage.md). Documentation and design only.

**Non-goals.** No instrumentation, no runtime accounting, no change to any
simulation.

**Primary files.** `docs/src/parent_budget/`, `docs/make.jl`.

**Tests.** The documentation build, and the link and formatting checks.

**Definition of done.** One contradiction-free normative contract; a coverage
row for every path that writes an authoritative parent field, each with a
disposition, an intended collection level, the evidence that would establish it,
the test that would provide that evidence, an owning stack step, and, for a
transfer event, a declared topology.

## Stack step 2 — Journal and endpoint-reconciliation core

**Claim.** None in a simulation. The core's own invariants hold: expectations
come from a declared schema and never from records, only a measured component
carries a nonzero amount, an unknown blocks, a not-applicable is not a measured
zero, an unset packet slot is not a not-applicable one, a leg is recorded once,
an aggregate is never summed with its decomposition, and a commit is atomic.

**Scope.** The declaration layer that says what a configuration is expected to
produce, the internal types that record what it did, and the arithmetic that
turns endpoints and legs into the three residuals, with the packet layout that
makes one collective per step possible.

**Non-goals.** No runtime wiring, no process attribution, no surface-transfer
certification, no user-facing configuration key, and no claim that any
simulation closes.

**Primary files.** `src/parent_budget/`, under
`ClimaAtmos.Internals.ParentBudget`: `integrals.jl`, `schema.jl`,
`reduction.jl`, `journal.jl`, `transaction.jl`.

**Tests.** `test/parent_budget/` — endpoint integrals against real ClimaCore
state across the supported model and surface combinations and both state float
types; schema declaration, and its refusal of undeclared, duplicate and missing
records; journal invariants including deliberate faults; packet slot states,
layout and reduction assembly.

**Definition of done.** The core is internal, exports nothing, and every rule in
the contract that can be enforced by construction is enforced rather than
documented. An expected channel or event that is never recorded blocks the claim
instead of disappearing from it.

## Stack step 3 — Accepted update envelopes

**Claim.** Accepted-state reconciliation. The endpoint change agrees with the
recorded channel envelopes and final maps within tolerance.

**Scope.** A timestepper adapter that captures the complete accepted explicit,
limited and implicit envelopes and the final maps, and the packed per-step
reduction that makes collecting them affordable.

**Non-goals.** No per-process attribution. An envelope stands alone at this step
and its decomposition is deliberately absent.

**Primary files.** `src/simulation/integrator.jl`, `src/simulation/solve.jl`,
`src/prognostic_equations/limited_tendencies.jl`,
`src/prognostic_equations/implicit/implicit_tendency.jl`, and a new timestepper
adapter under `src/parent_budget/`.

**Requirements.**

  - Record the exact `ClimaTimeSteppers` version and behavior being adapted; the
    trace test is the pin.
  - Add an executable trace test for stage construction and hook order.
  - Obtain every envelope from stage coefficients and applied increments, never
    from endpoint subtraction.
  - Accumulate and reduce in at least `Float64`, converting before accumulation.
  - Use one fixed-layout packed global collective per accepted step.
  - Reuse the previous closing endpoint as the next opening endpoint.
  - Where an accepted implicit envelope cannot be obtained independently, leave
    the affected claim blocked rather than inferring it.

**Tests.** Trace test for stage and hook order; envelope reconciliation on dry,
moist and implicit configurations; a bitwise trajectory-invariance test with
accounting on and off; a reduction-count test.

**Definition of done.** Primary reconciliation passes or is explicitly blocked
with named blockers, in every supported configuration, with one collective per
step and no trajectory change.

## Stack step 4 — Explicit attribution

**Claim.** Process attribution for the explicit channels: classified
contributions reproduce `env.explicit_main` and `env.explicit_limited`.

**Scope.** Decomposition of `Yₜ` and `Yₜ_lim` separately, with exact accepted
tableau weights, and the conversion of the coverage table into an executable
registry.

**Non-goals.** The implicit channel, which is step 5. No process-change record
is used as a closure leg.

**Primary files.** `src/prognostic_equations/remaining_tendency.jl`,
`src/prognostic_equations/limited_tendencies.jl`, the explicit forcing, sponge,
diffusion, radiation and microphysics files the registry lists, and a new
`src/parent_budget/coverage_registry.jl`.

**Requirements.**

  - Decompose the two explicit channels separately. An adapter reading `Yₜ`
    alone silently loses horizontal tracer advection and tracer hyperdiffusion.
  - Compare classified totals against their channel envelope, never against the
    endpoint change.
  - Keep forcing paths that do not write `ρ` at a zero mass contribution.
  - Build the run's schema from the registry, so the expected set of events is
    fixed by the configuration before the first step.
  - Generate the documentation table from the registry, or test exact agreement.
  - Book net amounts. Under `parent_budget_attribution: gross` carry each row's
    positive and negative parts as two further slots, with the identities still
    on the net.

**Tests.** Attribution residual per explicit channel; deliberate missing,
duplicated and sign-reversed event tests; registry-to-documentation agreement.

**Definition of done.** Both explicit channels attribute to within tolerance or
name what blocks them, and the registry is the single source of truth for
process classification.

## Stack step 5 — Implicit and post-implicit attribution

**Claim.** Implemented-update accounting, and process attribution for the
implicit channel.

**Scope.** The effective implicit increment as the pinned timestepper forms it,
the ownership of post-Newton and post-implicit hooks, and the algebraic solve
defect.

**Non-goals.** No inference of a tableau the pinned version does not expose. If
a decomposition cannot be proven, aggregate closure is reported and attribution
is left blocked.

**Primary files.** `src/prognostic_equations/implicit/implicit_tendency.jl`,
`src/prognostic_equations/implicit/initialize_implicit_problem.jl`,
`src/prognostic_equations/implicit/jacobian.jl`,
`src/simulation/integrator.jl`, and the post-implicit correction code.

**Requirements.**

  - Derive the effective implicit increment from the pinned implementation.
  - Identify which post-Newton, post-implicit, DSS, limiter or constraint
    changes are already folded into it, and never book an aggregate together
    with its decomposition.
  - Define the solve defect with a verified sign and accepted weight.
  - Book the post-implicit correction as a decomposition row of the implicit
    channel, never as an envelope.

**Tests.** Converged and deliberately under-converged solves, sweeping
`max_iters` and `approximate_solve_iters`; all three
`update_constrain_state_every` cadences, `"step"`, `"stage"` and `"dss"`.

**Definition of done.** The implicit envelope's decomposition closes with the
defect included, and every hook is booked exactly once.

## Stack step 6 — Boundaries and reservoir transfers

**Claim.** Transfer consistency: independently measured legs of a modeled
internal exchange cancel within tolerance, and a boundary crossing is reported
as such rather than as a failed cancellation.

**Scope.** Atmospheric, slab and exterior legs of every transfer event in the
registry, collected independently.

**Non-goals.** No synthesized exterior counter-leg, and no forced cancellation.

**Primary files.** `src/prognostic_equations/surface_flux.jl`,
`src/prognostic_equations/surface_temp.jl`,
`src/prognostic_equations/water_advection.jl`,
`src/parameterized_tendencies/microphysics/tendency.jl`,
`src/parameterized_tendencies/radiation/radiation.jl`.

**Requirements.**

  - Take each event's topology from the schema, never from which legs a run
    happened to record, and resolve a configuration-dependent topology once at
    setup.
  - Handle 0-moment removal separately from 1-moment sedimentation and fallout.
  - Treat `vertical_advection_of_water_tendency!` as implicit.
  - Include mass, water and the appropriate energy carrier independently.
  - Separate TOA radiation, surface radiation, latent and sensible exchange,
    precipitation, and slab heating.
  - Report atmosphere-only boundary flux, coupled change, and transfer mismatch
    separately.
  - Record the physical limitations of `Y.sfc.water` in the register.

**Tests.** Per-event transfer residual in both control volumes; a mismatch
injected deliberately and detected; dry and moist slab and prescribed-surface
configurations.

**Definition of done.** Every transfer event has both legs measured from their
own quadrature, and any mismatch is preserved and reported rather than removed.

## Stack step 7 — Final maps, restarts, and callbacks

**Claim.** Accepted-state reconciliation holds across finalization and restart.

**Scope.** The final accepted-state limiter, DSS, consistency-repair and
constraint maps, the restart transition, and the callback mutation rules.

**Non-goals.** No intermediate-stage map is booked at its raw value; those stay
stage observations unless their accepted weights are proven.

**Primary files.** `src/prognostic_equations/limited_tendencies.jl`,
`src/prognostic_equations/constrain_state.jl`, `src/simulation/restart.jl`,
`src/callbacks/`, `test/restart.jl`, `test/restart_AtmosSimulation.jl`.

**Requirements.**

  - Segment the report at a restart, persist the last closing endpoints in the
    checkpoint, and check the restored state against them exactly before the
    first transaction opens.
  - Require custom callbacks to declare read-only behavior or provide
    accounting, and fail closed for an unknown state-mutating callback.
  - Verify no duplicate charge across step, restart and callback boundaries.

**Tests.** Each final map measured on the accepted state; all three
`constrain_state!` cadences; restart round trip with the ledger enabled; a
deliberately mutating callback rejected at setup.

**Definition of done.** Every authoritative accepted-state mutation has exactly
one disposition and exactly one booking, across ordinary steps and restarts.

## Stack step 8 — Reporting and closure certification

**Claim.** The published claim certificate: exactly which claim levels were
established, for which quantities and control volumes, under which
configuration.

**Scope.** Configuration, report generation, the claim certificate, integrated
verification, and the performance gates.

**Non-goals.** No new accounting. If a claim was not established by steps 3 to 7,
this step reports that rather than closing it.

**Primary files.** the configuration schema and getters, the diagnostics and
reporting modules, `docs/src/parent_budget/`, and the integration and
performance jobs.

**Requirements.**

  - Provide `off`, `summary` and `audit` modes, with `off` the default until
    overhead and correctness are established, and
    `parent_budget_attribution: net | gross`, with `net` the default.
  - Emit versioned machine-readable output, and a concise human-readable
    summary.
  - Report each quantity and control volume as `pass`, `fail`, `blocked` or
    `not_applicable`, and a crossing as `reported`.
  - Include the configuration, backend and rank count, state and accounting
    precision, timestepper algorithm and adapter version, supported-scope
    classification, tolerances and scales, the parent, attribution and transfer
    residuals, the physical-completeness limitations, and any restart
    segmentation.
  - Calibrate `κ` by the contract's protocol and commit the calibration table.
  - Verify bounded storage and no per-step allocation growth in summary mode,
    against an explicit acceptable runtime and memory overhead.

**Tests.** CPU, GPU, MPI, restart, fault injection, and performance; the
energy-reference covariance audit, algebraic and physical kept apart; comparison
against `check_conservation` as an independent cross-check only.

**Definition of done.** A run emits a certificate that a reader can act on, and
no claim level appears in it that its own tests did not establish.

## Acceptance criteria for the whole sequence

  - Every authoritative accepted-state mutation in each supported configuration
    has exactly one disposition and exactly one booking.
  - Intermediate-stage changes are distinguished from accepted-step
    contributions, and nothing is booked twice.
  - Every component is measured, invariant zero, not applicable, or an explicit
    unknown, and an unknown blocks the claim it affects.
  - Parent, attribution and transfer residuals are computed and reported
    separately.
  - No envelope is reconstructed from the endpoint subtraction it explains.
  - Contributions use the accepted stages, weights, and split order.
  - Expected channels, events and reservoirs are declared from the configuration
    before collection begins, and a declared term that is never recorded blocks
    the claim rather than disappearing from the report.
  - Reservoir legs are collected independently and never negated into existence.
  - Internal cancellations are measured, never imposed. An exterior crossing is
    reported as a signed boundary source or sink, with no fabricated counter-leg
    and no cancellation test.
  - A packet slot is unset, measured, or explicitly not applicable, and reduction
    is refused while a required slot is unset.
  - Residuals are reported per quantity as stepwise maximum, cumulative
    absolute, and signed drift, never normalized by a signed total.
  - Accounting precision matches the documented precision at the point of
    accumulation, not after the fact.
  - One packed global collective per accepted step.
  - The ledger changes neither the trajectory nor solver convergence, tested
    bitwise.
  - Unsupported configurations and undeclared state-mutating callbacks fail at
    setup.
  - The contract, the registry, the code, and the published report describe the
    same scope.
