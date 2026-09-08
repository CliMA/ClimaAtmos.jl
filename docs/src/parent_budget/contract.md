# Parent-Budget Ledger: Closure Contract

This page is the normative contract for the parent-budget ledger. It fixes the
parent quantities, the reservoirs and control volumes they live in, the
identities that reconcile them, the vocabulary every report uses, and the
tolerance a residual is judged against. Everything else in the parent-budget
work is measured against this page: the
[architecture](architecture.md) says how the pieces fit together, the
[coverage registry](coverage.md) inventories every path that has to be
dispositioned, and the [implementation plan](plan.md) sequences the work.

The contract governs three separate budgets that share one journal. It does not
by itself establish that any of them closes in a running simulation. What has
been established at any moment is a property of the implementation, and each
stack step in the plan names the one claim it adds.

The same terms are explained in plain language, for model users, on the
[vocabulary page](vocabulary.md). This page governs where the two differ.

## Claim levels

Six claims are kept apart. Establishing one does **not** establish the next, and
a report may never present a lower level as evidence for a higher one.

 1. **Accepted-state reconciliation.** The endpoint change of each parent
    quantity over an accepted step agrees, within tolerance, with the recorded
    accepted channel envelopes and final maps.
 2. **Implemented-update accounting.** The ledger represents what the supported
    discrete integrator actually applied, including the stage weights it used
    and the algebraic defect of an unconverged solve.
 3. **Process attribution.** Classified process contributions reproduce their
    channel envelope, so a named process can be held responsible for its share.
 4. **Transfer consistency.** Independently measured legs of a modeled internal
    exchange cancel within tolerance.
 5. **Physical completeness.** The model equations contain every reservoir,
    carrier, work term and exchange the physics intends.
 6. **Provenance attribution.** Material or energy can be assigned to an origin.

Levels 1 to 4 are what this work targets. Level 5 is tracked in the
[limitations register](#Limitations-register) and is never a numerical residual:
incomplete equations still close against themselves, so a physically missing
term can leave no residual at all. Level 6 belongs to the source-tag work and is
excluded here.

**Discrete closure, within a declared tolerance.** The phrase "exact closure" is
not used. What the ledger can establish is agreement between an endpoint change
and a sum of recorded amounts to within the tolerance defined below, for the
discrete system the model actually integrates. That is a weaker and more useful
statement than exactness, and it is the only one the arithmetic supports.

## Ledger glossary

One vocabulary, used in this contract, in the code, in the registry, and in
every report. The heading is qualified because `docs/src/glossary.md` is the
model's own glossary, and two headings with the same text make every reference
to either of them ambiguous.

| Term                      | Meaning                                                                                                                                            |
|:------------------------- |:-------------------------------------------------------------------------------------------------------------------------------------------------- |
| Parent quantity           | One of `M`, `W`, `E`. The three budgets the ledger reconciles.                                                                                     |
| Reservoir                 | A place the model owns state in, which can gain or lose a parent quantity.                                                                         |
| Control volume            | A named set of reservoirs a budget is projected onto.                                                                                              |
| Budget schema             | The configuration-derived declaration of everything the ledger expects to collect, built before collection begins.                                 |
| Internal transfer         | An exchange whose participating reservoirs are all modeled. Its legs are expected to cancel.                                                       |
| Exterior crossing         | An exchange whose counterparty is not modeled. It has one modeled leg and no cancellation test.                                                    |
| Exterior counterparty     | The unmodeled far side of a crossing. Named in the topology; never given a numerical leg.                                                          |
| Endpoint                  | The authoritative integral of one parent quantity in one reservoir at one instant.                                                                 |
| Accepted channel envelope | The complete update one integrator channel applied to the accepted state, obtained from the applied increment rather than by endpoint subtraction. |
| Event                     | One physical or numerical occurrence, shared by every leg that belongs to it.                                                                      |
| Reservoir leg             | The signed amount one event contributed to one reservoir.                                                                                          |
| Final map                 | A map applied to the accepted state itself, whose raw before/after difference is part of the endpoint change.                                      |
| Stage observation         | A raw before/after difference taken on an intermediate stage array. Evidence, never accounting.                                                    |
| Parent residual           | `R_parent`, from the primary identity below.                                                                                                       |
| Attribution residual      | `R_attribution`, from the process-attribution identity below.                                                                                      |
| Transfer residual         | `R_transfer`, from the transfer identity below.                                                                                                    |
| Measured                  | The amount was taken from the implemented update.                                                                                                  |
| Invariant zero            | The amount is provably zero, and the proof is named.                                                                                               |
| Not applicable            | The quantity does not exist for this path or reservoir in this configuration.                                                                      |
| Unknown                   | Not established. Blocks the claim it belongs to.                                                                                                   |
| Blocked                   | A claim that cannot be evaluated because a required component is unknown, open, or missing.                                                        |
| Open                      | A disposition not yet established from the code. Demands nothing of a record and blocks every claim it feeds.                                      |
| Reported                  | The status of a crossing: a signed boundary flux with no verdict, because nothing exists to cancel against.                                        |

## The three identities

Three residuals answer three different questions. They are computed separately,
reported separately, and never added together.

### Primary reconciliation

For each parent quantity `q`, control volume, and accepted step:

```
R_parent(q) = ΔB(q) − Σ_c Q_envelope(q, c) − Σ_m Q_final_map(q, m)
```

`ΔB(q)` is the endpoint change over the accepted step. `c` runs over the
integrator's channels and `m` over the final accepted-state maps.

**The envelope must come from the update the integrator applied.** It is the
increment the channel contributed, read from stage coefficients and applied
increments. Reconstructing it by subtracting endpoints makes the identity
`R_parent = 0` by construction: the same measurement would appear on both sides
and the residual would test nothing. Endpoint subtraction is the left-hand side
of this identity and may never also serve as its right-hand side.

Answers: *did the accounting reproduce the accepted state transition?*

### Process attribution

For each parent quantity and each channel:

```
R_attribution(q, c) = Q_envelope(q, c) − Σ_{e ∈ c} Q(q, e)
```

where `e` runs over the classified events assigned to that channel.

Answers: *did the classified paths explain the whole of the accepted channel?*
A channel can reconcile perfectly at level 1 while its attribution is entirely
unexplained, which is exactly why the two residuals are separate.

### Final maps are not attribution channels

The two identities take different terms, and the distinction is easy to lose.

An **accepted channel** is a term of the primary identity *and* an attribution
target: its envelope is the amount its classified processes have to add up to. A
**final accepted-state map** is a term of the primary identity *only*. A
limiter, a DSS correction, a consistency repair, or a constraint applied to the
accepted state contributes its raw before/after difference to `ΔB`, and it has
no envelope, no decomposition, and no attribution residual.

Recording a final map therefore never creates a requirement for a channel
envelope. An operation acquires that requirement only by being declared an
accepted integrator channel in the schema, which is a statement about what the
integrator applied rather than about where in the step the operation runs.

Raw intermediate-stage maps are neither. They stay observations until their
exact accepted contribution has been established.

### Transfer consistency

An event's topology is **declared**, never inferred from whichever legs happened
to be recorded. For each event the schema says which modeled reservoirs take
part and whether the counterparty is another modeled reservoir or an unmodeled
exterior. Two different tests follow, and applying the wrong one is the mistake
this separation exists to prevent.

**Internal or coupled transfer.** Every participant is a modeled reservoir.
Every leg the schema declares is required, and their signed sum over a control
volume containing all of them is tested for cancellation:

```
R_transfer(q, e) = Σ_{r ∈ modeled(e)} Q(q, e, r)
```

A nonzero result is a finding, naming lagged coupling, clipping, inconsistent
quadrature, or a reservoir the model does not represent. A declared leg that was
not recorded **blocks** the event; it is never read as a zero.

**Exterior crossing.** One side is a modeled reservoir and the other is an
unmodeled exterior: the top of the atmosphere, a prescribed forcing, the
destination of 0-moment removal, or an externally imposed surface flux whose
store the model does not carry. Then

  - the modeled reservoir's leg is measured independently and is required;
  - the exterior counterparty is named in the event's topology as metadata;
  - **no numerical exterior counter-leg is created.** A synthesized counterparty
    guarantees cancellation and therefore measures nothing;
  - the zero-sum cancellation test does not apply, and its inapplicability is
    reported rather than passed off as a cancellation that happened to hold;
  - the signed crossing is reported as a boundary source or sink, and it
    contributes to the accepted channel and the parent reconciliation that carry
    it.

One physical event can be internal in one control volume and a crossing out of
another: precipitation reaching a slab is internal to the coupled volume and
leaves the atmosphere-only volume. That is a property of the view, declared per
view, and never a property of what happened to be recorded.

Answers: *did independently measured legs of one exchange agree, where both
sides are modeled?*

### What none of them proves

None of the three establishes physical completeness or provenance. A model that
omits a reservoir entirely closes all three identities perfectly.

### Aggregates are envelopes, never extra legs

An accepted aggregate — the whole increment one channel applied — is an envelope
or a fallback for attribution that does not exist yet. It is **never summed**
alongside its own decomposition, because that counts the same update twice.

The rule is about sums, not about recording. An envelope and its decomposition
are meant to be recorded together: comparing them *is* the attribution identity.
What keeps them apart is the collection level, which decides which identity a
leg belongs to, rather than care exercised at every call site.

## Declared expectations

The ledger does not discover what it should have collected by looking at what it
did collect. Expectations come from a schema derived from the selected model
configuration, and the recorded data are checked against that schema.

> Expectations are constructed from the selected model configuration before
> collection begins. Recorded data are checked against those expectations;
> recorded data must never define which channels, events, or reservoirs were
> expected.

The schema is the executable form of the [coverage registry](coverage.md), and
it declares:

  - the enabled budget quantities;
  - the supported control volumes;
  - the expected accepted-update channels;
  - the expected final-state maps;
  - the expected transfer events;
  - for each event, whether it is internal, coupled, or exterior;
  - the modeled reservoirs taking part in each event;
  - the exterior counterparties, which have no numerical state leg;
  - the numerical legs each event requires;
  - the applicability of each quantity in each reservoir, and which components
    carry a proven-zero obligation;
  - whether an accepted channel envelope is required;
  - the roster of process rows each channel's decomposition must record, so an
    omitted process is a named missing row and not an exact cancellation among
    the rows that arrived.

Three consequences follow, and each refuses a failure mode that an
observation-driven report has by construction.

**A missing expectation is visible.** An expected channel or event that recorded
nothing at all produces a `blocked` or `fail` result naming it. It does not
disappear from the report, which is what happens whenever a report is assembled
out of whatever was recorded.

**An unexpected record is refused.** A channel, event, or reservoir the schema
does not declare is an error rather than a new row, and so is a second record
under an identity already used.

**Applicability is declared, never sniffed.** Whether a reservoir owns a
quantity comes from the configuration, not from whether a field happens to exist
in the state. `Y.sfc.water` exists in a dry run and holds a permanent zero, so
reading its presence as ownership is precisely the confusion between a measured
zero and an inapplicable quantity that the status vocabulary exists to prevent.

## Supported scope

### Parent quantities

Mass `M`, total water `W`, and total energy `E`. Each has its own definition,
applicability rule, collection path, residual, tolerance, and result. They share
a journal so that a coupled exchange stays coordinated across the three. They
are never interchangeable and a component of one is never inferred from another.

### Configurations supported

Concrete types, not families.

  - Microphysics: `DryModel`, `EquilibriumMicrophysics0M`, and
    `NonEquilibriumMicrophysics1M`, on the sphere and in a single column.
  - `diff_mode` explicit and implicit.
  - `microphysics_tendency_timestepping` explicit and implicit.
  - Surface: every prescribed or diagnosed surface temperature, and
    `SurfaceConditions.SlabOceanTemperature`.
  - Radiation off, `HeldSuarezForcing`, and RRTMGP.
  - Forcing: `LargeScaleSubsidence`, large-scale advection, and the external
    forcing that reaches `apply_Tq_forcing!`. Supported, with the open dry-air
    budget recorded in the limitations register.
  - Callbacks: the default set only, which the coverage registry inventories as
    read-only with respect to `Y`.

### Configurations out of scope

  - `PrognosticEDMFX` and `DiagnosticEDMFX`. The updraft subdomains carry their
    own `ρa`, `mse`, and `q_tot`, and whether those sit inside the atmospheric
    control volume or are a decomposition of it is a modelling question, not a
    bookkeeping one.
  - `prescribed_flow` runs. `fully_explicit_tendency!` overwrites state from a
    prescribed field, so mass and energy are imposed rather than evolved. The
    path is booked as a prescribed overwrite and no closure is claimed for it.
  - Chemistry (`GasPhaseChem`), which changes tracer composition through an
    external solver.
  - `NonEquilibriumMicrophysics2M` and `NonEquilibriumMicrophysics2MP3`, whose
    number concentrations carry paths that have not been audited.
  - Every local, column, or component-energy budget.

### Custom callbacks

`AtmosSimulation` accepts a `callbacks` keyword and appends whatever it is
given, so a caller can install a callback that writes `Y`.

A custom state-mutating callback is **unsupported** unless it either declares
itself read-only with respect to `Y` or supplies its own ledger accounting.
Without one of the two, the configuration fails closed at setup rather than
producing a closure claim that silently omits whatever the callback did.

### Timestepping methods supported

Every `CTS.IMEXAlgorithm` built with the `Unconstrained` constraint, which is the
family one IMEX-ARK stepper runs. It includes `ARS343`, the default, `ARS222`,
which most configurations in this repository select, and the
`CTS.ExplicitAlgorithm` tableaus, which are the same algorithm with identical
explicit and implicit tableaus and no Newton solver. The adapter reads the
tableau from the integrator at setup and never assumes a particular one.
`SSP`-constrained tableaus and Rosenbrock algorithms run through different
steppers and are refused at setup. Decided on 2026-09-08.

All of these are fixed-step. `args_integrator` passes a fixed `dt` and no
controller, so there is no embedded error estimator, no step rejection, and no
retry. The journal still commits only at step end, and an assertion fails if a
rejection ever occurs, so the property is guarded rather than assumed.

The behavior being adapted belongs to a specific `ClimaTimeSteppers` version.
The adapter records `pkgversion(ClimaTimeSteppers)` in every certificate, and its
trace test fixes the stage construction and hook order it assumes, so a change
in the dependency fails a test instead of silently changing the meaning of every
implicit leg. The compat bound is deliberately not pinned to one patch version:
the upstream syncs move it, and the trace test is the pin.

## Parent quantities and authoritative integrals

For the atmosphere, over the global domain:

```
M = ∫ ρ            kg
W = ∫ ρq_tot       kg
E = ∫ ρe_tot       J
```

For the slab surface, as a horizontal integral at the boundary:

```
M_sfc = ∫_sfc sfc.water                                  kg
W_sfc = ∫_sfc sfc.water                                  kg
E_sfc = ∫_sfc sfc.T · ρ_ocean · cp_ocean · depth_ocean   J
```

### Total water is `ρq_tot` alone

The categories **partition** `ρq_tot` and are never added to it.
`set_precomputed_quantities!` builds the thermodynamic state as
`q_liq = q_lcl + q_rai`, `q_ice = q_icl + q_sno`, with `q_tot ≥ q_liq + q_ice`,
so rain and snow sit inside `q_tot` exactly as cloud water does. One-moment
microphysics applies no direct source to `ρq_tot` or `ρ` at all; it moves the
category fields.

So `W = ∫ρq_tot` in every moist configuration, and `ρq_lcl`, `ρq_icl`, `ρq_rai`
and `ρq_sno` are excluded from it. Adding any of them would invent water every
time cloud condensate became rain.

| Model                          | `W`            |
|:------------------------------ |:-------------- |
| `DryModel`                     | not applicable |
| `EquilibriumMicrophysics0M`    | `∫ ρq_tot`     |
| `NonEquilibriumMicrophysics1M` | `∫ ρq_tot`     |

### Total energy is `ρe_tot` alone

Total energy is prognostic, so it is authoritative and nothing is reconstructed
from momentum and thermodynamic state. `ρtke`, the tagged tracers `ρe_tag_*`,
the source tags `ρe_src_*`, and the process records `prc_e_*` are excluded:
they are diagnostics of the energy, not additional energy. The same holds for
component diagnostics of any kind.

One canonical `Thermodynamics` and gravitational energy convention is used for
the closure implementation, and every report records which one.

### Dry air is a derived diagnostic

```
D = M − W = ∫ (ρ − ρq_tot)      kg
```

`D` is well defined and the two forms are the same number, because the integral
is linear. **`D` is not a conservation invariant and is not a parent quantity.**
It is a diagnostic derived from state, useful for checking the mass and water
budgets against each other, and it closes nothing on its own.

`ρ` does move with `ρq_tot` on the paths that move air and water together:
0-moment removal, sedimentation in `vertical_advection_of_water_tendency!`, the
surface flux, the viscous sponge, and `enforce_mass_energy_consistency!`. It
does not move with it on the prescribed forcing paths:

| Path                                    | Writes                                 | Writes `ρ` |
|:--------------------------------------- |:-------------------------------------- |:---------- |
| `large_scale_advection_tendency_ρq_tot` | `ρq_tot`, `ρe_tot`                     | no         |
| `subsidence_tendency!`                  | `ρq_tot`, `ρe_tot`, `ρq_lcl`, `ρq_icl` | no         |
| `apply_Tq_forcing!`                     | `ρq_tot`, `ρe_tot`                     | no         |

Each adds water to a column without adding air to it, so `W` moves while `M`
stays put and `D` moves by `−ΔW`. A forced configuration therefore has a dry-air
budget that is open by construction.

**The rule this fixes.** A forcing path that changes `ρq_tot` without changing
`ρ` records a mass component of `invariant zero` — the implemented equation adds
no mass, and that zero is the measured truth about the code — with its water and
energy components measured independently. A mass contribution is never
manufactured from a water tendency to make `D` close. Whether the model ought to
add air along with prescribed moisture is a question about the physics; it
belongs in the limitations register, not in the bookkeeping.

### Linearity

`M`, `W`, `E`, `W_sfc` and `E_sfc` are linear extensive functionals of
authoritative prognostic state. `E_sfc` is linear because the slab heat capacity
is constant. Exact additive process attribution is therefore well defined and no
allocation convention is needed. This is a property of these particular
integrals, so changing one reopens the question.

## Reservoirs and control volumes

| Reservoir    | State         | Owns          | Exists when                              |
|:------------ |:------------- |:------------- |:---------------------------------------- |
| Atmosphere   | `Y.c`, `Y.f`  | `M`, `W`, `E` | always                                   |
| Slab surface | `Y.sfc.T`     | `E`           | `SlabOceanTemperature`                   |
| Slab surface | `Y.sfc.water` | `M`, `W`      | `SlabOceanTemperature` and a moist model |
| Exterior     | none          | —             | always                                   |

There is no prognostic snow, soil-water, deposited-condensate or wave-energy
reservoir. Every other surface is prescribed or diagnosed and is therefore
exterior, not a reservoir: its state is not owned by the model, so a flux into
it is a boundary crossing and never an internal transfer.

Two control volumes are supported:

  - **Atmosphere only.** The atmospheric leg of a surface exchange is a boundary
    crossing.
  - **Atmosphere and surface.** Both legs are inside, and their sum is tested
    for cancellation.

A control volume naming a reservoir the configuration does not have is
unavailable and is not reported at all. Reporting it would return the
atmosphere-only numbers under the coupled name.

### `Y.sfc.water` is an accounting accumulator

`Y.sfc.water` integrates the water the atmosphere has delivered to the surface.
It is **not** a physically complete ocean, soil-water, snow, or
deposited-condensate reservoir. It has no hydrology of its own: it does not
constrain evaporation, it does not run off, it does not freeze, and nothing
reads it back into a surface flux. Its value is a bookkeeping total, and its
physical incompleteness is a level-5 limitation recorded in the register.

Treating it as a reservoir is still right for the accounting, because the water
that enters it did leave the atmosphere, and a coupled control volume has to see
where the water went. What must not happen is a report that presents it as
surface hydrology.

### The slab's mass and water are one endpoint projected twice

`Y.sfc.water` is a water content in kg m⁻², and the water it gains left the
atmosphere as `ρq_tot` and therefore also as `ρ`. So the slab contributes to `M`
and to `W` with the same amount.

These are **two projections of one endpoint, not two measurements.** Both read
`Y.sfc.water` through one reduction, so they cannot disagree, and no test of
them can discover anything.

**Independent collection is a property of the transfer legs.** The atmospheric
side of a surface exchange and the surface side of it are measured separately,
from different quadratures, and must be allowed to disagree. That is where a
coupling mismatch becomes visible, and it is what the transfer residual
measures. A leg is never created by negating its counterparty: a synthesized
counter-leg guarantees cancellation and therefore measures nothing.

## Accepted-step boundaries

One transaction per accepted timestep. It opens on the finalized endpoint of
step `n` and closes on the finalized endpoint of step `n+1`.

Finalization means the `ClimaODEFunction` hooks that touch authoritative state,
in the order `ClimaTimeSteppers` runs them: `lim!` (`limiters_func!`), `dss!`,
`constrain_state!` at its configured cadence, and `cache!`. The endpoint is read
after the last of these and **before** any discrete callback fires, so a
callback that mutates `Y` belongs to transaction `n+1` and never to the step
whose hooks just finished. In every supported configuration no discrete callback
mutates `Y`, so the choice is currently unobservable; it is fixed here so that a
callback which does mutate state falls on one side by rule rather than by
accident.

`update_constrain_state_every` defaults to `"step"` and accepts `"stage"` and
`"dss"`. The ledger reads the configured cadence and records one correction per
firing, so a leg carries a stage index and an occurrence alongside its event and
step. Without them the same correction firing at four ARS343 stages would
collide as one leg.

**Endpoint continuity is enforced, not assumed.** Transaction `n+1` opens on the
endpoint transaction `n` closed on, and the ledger checks amounts *and* statuses
rather than trusting the caller. A gap between them is a change nothing
accounted for, and it would otherwise vanish from the cumulative total without
leaving a residual anywhere.

## What a ledger amount is

A leg's amount is an **accepted-step-weighted extensive contribution**: the part
of `Bⁿ⁺¹ − Bⁿ` that this path is responsible for. It is not a raw tendency, and
it is not a raw before/after difference taken on an intermediate stage array.
Those three are different numbers and the schema does not let them be mixed.

The distinction matters because `lim!`, `dss!` and `constrain_state!` run on
intermediate stage arrays as well as on the accepted state.

  - A **final map**, applied to the accepted state, contributes its raw
    before/after difference. That difference *is* part of the endpoint change.
  - An **intermediate-stage map** does not. It changes the array a later tendency
    evaluation reads, so the tableau mediates its effect on the endpoint. It
    enters the identity only with its exact accepted weight, generally involving
    the tableau's `bᵢ` and the implicit `γᵢ`.

A raw intermediate-stage difference is a **stage observation**. It is kept, in a
structure that no projection iterates and no residual reads, because knowing
which stage moved the state is what turns "the step does not close" into a
located defect. It is evidence, not accounting, and the type system keeps it out
of the parent identity rather than relying on care.

### A decomposition amount is net

A process row books one signed amount per quantity, reservoir and accepted step:
the integral of everything the process applied, its positive and negative parts
together. The three identities are linear in these net amounts and need nothing
else.

Gross accounting keeps the positive and negative parts of a row apart. It is a
diagnostic, not a closure term: it shows activity behind a zero net, such as
interior diffusion or a limiter that moves water between columns, and it costs
two more slots per row. It is available through the configuration key
`parent_budget_attribution`, which takes `net`, the default, or `gross`. Under
`gross` every decomposition row carries its two parts beside the net amount, and
the identities still use the net. Decided on 2026-09-08.

### A hook folded into an aggregate is booked once

Where the timestepper forms a stored implicit stage tendency by differencing the
stage state *after* post-implicit and post-Newton hooks have run, those hooks'
changes are already inside the effective implicit increment. Booking such a hook
again as its own leg double-counts. A hook already folded into an effective
implicit increment may be booked independently **only** if the same amount is
subtracted back out of that aggregate.

Which hooks are folded in is a property of the pinned timestepper version, and
the coverage registry records what the pinned version does. Until a folded
hook's accepted weight has been measured, an intermediate-stage leg is
`unknown`, never `measured`.

## Component status and evidence

Every `(M, W, E)` component of every leg and endpoint carries its own status and
its own evidence. Status is **per component**, never per leg: one event can
measure energy, prove a mass zero, and have nothing to say about water.

| Status         | Amount                           | In totals | Effect on a claim                      |
|:-------------- |:-------------------------------- |:--------- |:-------------------------------------- |
| Measured       | any                              | yes       | none                                   |
| Invariant zero | exactly zero, with a named proof | yes       | none                                   |
| Not applicable | exactly zero                     | no        | none; the quantity does not exist here |
| Unknown        | exactly zero                     | no        | blocks the claim                       |

Only a measured component may carry a nonzero amount, and that is enforced by
construction rather than documented. An invariant zero must name the proof that
makes it zero.

**Unknown blocks; it never contributes zero.** A missing term and a zero term
look identical in an output, and that is the failure mode a budget diagnostic is
most likely to have. An unknown component contributes nothing to a sum and marks
the affected claim blocked, and a blocked report names which components blocked
it so that it says what would unblock it.

**Not applicable is not a measured zero.** A quantity no reservoir in the view
owns — water in a dry model, anything at all in a coupled view of a
configuration without a slab — is reported as inapplicable. Reporting it as a
closed budget at zero would be a claim that was never made.

### Evidence

Each component's evidence identifies its status, the collection or proof method,
the source adapter or registry entry it came from, and the precision and
reduction route where that is relevant. Evidence is what lets a report say
*how* a number was obtained rather than only what it was, and it is what makes a
missing, duplicated or mismatched leg distinguishable after the fact.

## Output statuses

Every quantity, in every available control volume, reports exactly one of:

| Status           | Meaning                                                      |
|:---------------- |:------------------------------------------------------------ |
| `pass`           | Applicable, unblocked, and the residual is within tolerance. |
| `fail`           | Applicable, unblocked, and the residual exceeds tolerance.   |
| `blocked`        | A required component is unknown, open or missing; no claim.  |
| `not_applicable` | No reservoir in this view owns the quantity.                 |
| `reported`       | A crossing: a signed boundary flux with no verdict to give.  |

A blocked result still reports its numbers, because they are informative, but no
closure claim may be made from it. A `reported` result is not a verdict: a
crossing out of the view or out of the model has nothing to cancel against, and
its total is the boundary flux, which `not_applicable` would misdescribe as a
quantity nobody owns. An unavailable control volume is not reported at all,
which is different from `not_applicable`.

## Tolerance model

A residual is judged against a tolerance built from named parts, because the
parts scale differently and conflating them is how a real defect is absorbed.

For each parent quantity `q`:

```
τ_q = a_q + r_q · S_q + κ · ε_acc · ( abs(B_q ⁿ) + abs(B_q ⁿ⁺¹) + Σ_k abs(Q_q,k) )
```

  - `a_q` is an absolute floor in the units of the quantity, kg or J.
  - `r_q` is dimensionless.
  - `S_q` is a positive scale. It is never a signed total, so it cannot pass
    through zero and it cannot hide a sign.
  - `ε_acc` is the epsilon of the **accounting** arithmetic type, not of the
    state's type.
  - `κ` covers reduction order and rank dependence. It is **calibrated, never
    chosen**, by the protocol below, and recorded with the result. A guessed `κ`
    presented as universal is not acceptable.

The endpoint magnitudes are inside the last term deliberately. Bounding the
residual by the leg magnitudes alone is a stricter claim than the subtraction
supports, and fails on a step whose legs are tiny against the background.

The algebraic solve defect is **not** inside this tolerance. It is a
leading-order accounting term, reported separately.

### Calibrating `κ`

The protocol, decided on 2026-09-08. A row of the calibration table is one
backend, one state float type and one rank count. For each row, a named
configuration is run for 50 accepted steps with every term of the parent
identity measured, and the largest ratio of `abs(R_q)` to the arithmetic term of
`τ_q` evaluated with `κ = 1` is recorded over its steps and quantities. The
row's `κ` is four times that ratio, rounded up to a power of two.

The table is committed with the configuration name, the commit and the date of
every calibration, and it is read at setup. A run whose backend, float type and
rank count have no row has no tolerance: its verdicts are `blocked`, not `pass`.
One serial row is re-measured in CI and must stay below `κ/4`. Distributed rows
are measured where MPI is available, which is not this repository's GitHub
Actions, and the certificate names the row it used. `a_q` and `r_q` are zero
unless a configuration declares a physically motivated floor.

### The solve defect is leading order

The default is `NewtonsMethod(; max_iters = 1)` against
`ManualSparseJacobian(approximate_solve_iters = 1)`. One Newton iteration
against an approximate Jacobian does not converge the implicit stage, so the
algebraic residual is a first-order term in the accepted update, not a rounding
effect. The primary identity cannot close without it in any implicit
configuration.

This also means there is no nonlinear tolerance to tighten. The test that
establishes the defect sweeps `max_iters` and `approximate_solve_iters` and
requires the defect to shrink with the measured stage residual while the parent
residual stays at arithmetic level throughout. A residual that does not move
under that sweep is a bookkeeping error wearing a solver's clothes.

### Accounting precision

The ledger's arithmetic precision is independent of the state's and is at least
`Float64`. Every accumulation, reduction, endpoint, leg amount, and cumulative
total is in the accounting type even when the state is `Float32`.

Conversion happens **before** accumulation and before the global reduction.
Casting a completed `Float32` reduction to `Float64` is not enough: the
information is already gone. A `Float32` global mass carries about seven
significant digits, so a per-step change eight orders below it vanishes in a
`Float32` accumulation and the residual then measures nothing but rounding.

For the same reason a leg is measured as an **increment** wherever the code
offers one, never as a difference of two large states. A difference inherits the
cancellation of its operands even in `Float64`.

### Which aggregates decide closure quality

Three cumulative numbers are kept per quantity and control volume, not one.

  - `max abs(Rₙ)` names the worst single step.
  - `Σ abs(Rₙ)` cannot cancel and bounds the total unaccounted transfer.
  - `Σ Rₙ` is the signed drift.

**Closure quality is decided by the first two.** The signed cumulative residual
is reported because drift is worth seeing, and it is never used to pass a test
on its own: `+δ` on one step and `−δ` on the next sums to zero and reports a
perfectly closed run that closed on neither step.

The cumulative endpoint change is also compared against an independent
`Bᴺ − B⁰` reading held from the first accepted endpoint. A running sum of
per-step differences telescopes the same measurements and can only reproduce
them plus the rounding of every intermediate addition, so it can never contradict
them. The direct comparison is a second reading that shares no arithmetic with
the first.

Residuals are reported as signed absolute values in kg and J. A relative view
may be added against a documented positive scale. Normalizing by a signed total
is forbidden.

## Cost, and the rule that accounting changes nothing

A global integral is a collective. ARS343 has four stages and the coverage
registry lists dozens of paths, so a leg-per-reduction implementation would add
on the order of a hundred collectives per timestep. That is a design constraint,
not an optimization to defer.

Two rules follow, and they are part of the contract:

 1. Legs accumulate locally, and the global reduction happens **once per
    accepted step** over one packed fixed-layout buffer.
 2. The ledger is off by default behind its own configuration key. A run with it
    off must produce a bitwise-identical trajectory to the same run built
    without the feature, and a run with it on must produce the same trajectory
    as the same run with it off. Both are tested.

The second rule is what makes "the ledger is not a fixer" checkable rather than
merely stated. Nothing in the ledger writes to the state, and no residual may
ever be inserted as a balancing entry: a residual is defined by subtraction and
by nothing else.

## Energy-reference covariance

The admissible shift is audited as

```
E* = E + a·M + b·W
```

with the requirement that every leg and residual transforms as
`Q_E* = Q_E + a·Q_M + b·Q_W` and `R_E* = R_E + a·R_M + b·R_W`.

`a ≠ 0` is admissible. `b` is an open question, not a settled zero, and the
reason is `enforce_mass_energy_consistency!`: when a limiter moves `ρq_tot` by
`Δ`, it moves `ρ` by `Δ` and `ρe_tot` by `Δ·(uᵥ(T) + Φ)`. Whether that carrier
energy is consistent with a `b·W` shift across every water leg, including
precipitation fallout and surface deposition, is what the audit is for.

**Two different things are called covariance and only one can settle `b`.**

*Algebraic re-expression.* Take a completed ledger and apply the substitution to
every amount. The residual transforms the same way for any `a` and `b`, because
the ledger is linear and the substitution is exact. This is a **consistency
check** on the implementation — it verifies that the ledger really is linear and
that no amount was stored in a way that breaks the substitution — and it is
worth nothing as evidence about which `b` the model admits.

*Physical reference experiment.* Rerun the model with shifted thermodynamic
references and compare the two ledgers. This one can reject a `b`. It is an
intervention, so it has to be specified before it is run: which parameters shift
and by how much; how the initial state is transformed so the two runs start at
corresponding states; how boundary and carrier fluxes transform, in particular
the energy carried by a water flux across the surface; and how the slab
transforms, since `E_sfc` is built from `Y.sfc.T` and a constant heat capacity
and does not see the atmospheric reference at all.

Until those four are written down, no covariance result may accept or reject a
value of `b`. Neither audit is permission to invent a carrier-energy term the
model does not have.

## The existing conservation check

`check_conservation` in `src/simulation/solve.jl` is a useful independent
cross-check and the ledger is compared against it. It is not stage-integrated
accounting, for four reasons the ledger must not inherit:

 1. Its boundary term is `Δt × horizontal_integral_at_boundary(...)` from
    `flux_accumulation!`, which is `dt` times one sample rather than the
    stage-weighted flux the integrator applied.
 2. `flux_accumulation!` accumulates radiation only. Turbulent surface fluxes
    never enter it, so for a non-slab surface the energy check closes against an
    incomplete boundary term.
 3. `energy_conservation` divides by `sum(ρe_tot)`, a signed total.
 4. It reports one number for the whole run, so a per-step defect that changes
    sign is invisible.

## Design decisions

The reasoning behind the choices above, kept here so the contract itself stays
normative.

**Why the envelope is not endpoint subtraction.** The primary identity exists to
detect an update the accounting missed. If the right-hand side is rebuilt from
the same endpoints as the left, the identity is a tautology and detects nothing.
Every other property of the design follows from insisting that the two sides
share no arithmetic.

**Why three residuals rather than one.** A single number cannot distinguish "the
step transition was reproduced" from "the named processes explain the channel"
from "the two sides of an exchange agree". Merging them lets a success in one
mask a failure in another, and it makes a failure impossible to localize.

**Why status is per component.** The three budgets have different applicability
and different evidence on the same path. A momentum drag proves an energy zero
and says nothing about water; a forcing path measures water and proves a mass
zero. A per-leg status would have to pick one, and whichever it picked would be
wrong for the others.

**Why accounting precision is fixed above the state's.** The residual is what
survives subtracting two large global totals. Tying it to the state's type would
destroy the measurement in exactly the configuration that most needs it.

**Why the slab is a reservoir despite being incomplete hydrology.** The coupled
control volume has to see where the water went, and the water genuinely left the
atmosphere. Excluding the slab would make every deposition an unexplained loss.
The incompleteness is real and is recorded as a level-5 limitation instead of
being fixed by bookkeeping.

**Why the registry has to become executable.** A hand-maintained table of paths
diverges from the code silently, and a coverage claim resting on a stale table
is worth nothing. The table is a precursor to a registry the code reads, and the
documentation table is then generated from it or checked against it.

## Limitations register

Physical-completeness gaps. These are claim level 5 and are never numerical
residuals. The register is extended by every stack step and published with the
final report.

  - Gravity-wave drag and Rayleigh damping change momentum while `ρe_tot` is
    prognostic and unchanged, so their direct energy contribution is exactly
    zero by construction. Any intended frictional heating, stress work, or
    solid-Earth exchange has no implemented counterpart.
  - Prescribed forcing adds water and energy to a column without adding air to
    it, so a forced run has an open dry-air budget. The ledger records the mass
    component as an invariant zero and reports the open budget rather than
    closing it.
  - `Y.sfc.water` is an accounting accumulator with no hydrology, as described
    above. There is no soil, snow, or deposited-condensate reservoir at all.
  - `flux_accumulation!` omits turbulent surface fluxes.
  - The process record covers only the explicitly bracketed tendency path. Its
    bracket set is not the ledger's coverage set, and a process record is never
    a closure leg.

## Blockers

Each blocks a named claim, not the whole ledger.

| Blocker                                                                                                                               | Blocks                                                                     | Cleared by   |
|:------------------------------------------------------------------------------------------------------------------------------------- |:-------------------------------------------------------------------------- |:------------ |
| `Yₜ_lim` is a second explicit channel and an adapter reading `Yₜ` alone loses horizontal tracer advection and tracer hyperdiffusion   | water closure, and energy closure wherever a limited tracer carries energy | stack step 3 |
| Accepted implicit stage weights not yet read from the pinned timestepper                                                              | claim level 2 for implicit terms                                           | stack step 5 |
| The hooks folded into the effective implicit increment are known from the stepper source; their accepted weights are not yet measured | claim level 3 for the implicit channel                                     | stack step 5 |
| Coupled surface legs measured from two quadratures, agreement not yet measured                                                        | claim level 4 in the coupled view                                          | stack step 6 |
| Energy-reference `b`                                                                                                                  | the covariance claim                                                       | stack step 8 |
| One packed collective per accepted step not yet in place                                                                              | enabling the ledger in a run at acceptable cost                            | stack step 3 |
| `κ` not yet calibrated                                                                                                                | a numeric pass or fail verdict                                             | stack step 8 |
