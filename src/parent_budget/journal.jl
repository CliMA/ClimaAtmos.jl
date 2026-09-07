#####
##### Parent-budget ledger: the transactional event journal
#####
##### One journal holds the signed legs of every event in one accepted timestep.
##### A leg says what one event did to one reservoir, and it is recorded once.
##### Control-volume totals are projections of those legs, not separate entries,
##### so an internal transfer cancels because its two legs cancel and not because
##### anything was synthesized to make it.
#####
##### This file records what happened. `schema.jl` declares what should have, and
##### the two are compared at commit. Nothing here decides what was expected.
#####
##### Three rules from `docs/src/parent_budget/contract.md` are enforced by
##### construction rather than documented and hoped for. Only a measured
##### component may carry a nonzero amount, and a measured one must name how it
##### was obtained. A component that was not established is unknown and blocks
##### the claim it belongs to, so nothing turns silence into zero. And an event
##### is recorded at one collection level, so an envelope and its own
##### decomposition can be compared but can never land in the same sum.
#####
##### Legs are host-side scalar records built after the step's one collective, so
##### these types are written for clarity rather than for a kernel. Nothing here
##### is evaluated on the device.

# ============================================================================
# Component status and evidence
# ============================================================================

"""
    ComponentStatus

What is known about one component of a `BudgetComponent`.

  - `Measured`: the amount was taken from the implemented update.
  - `InvariantZero`: the amount is provably zero, and the proof is named.
  - `NotApplicable`: the quantity does not exist here in this configuration.
  - `UnknownComponent`: not established. It contributes nothing to a sum and
    *blocks* the claim for its quantity.

`UnknownComponent` exists so that an unestablished component cannot be quietly
treated as zero. That is the failure mode a budget diagnostic is most likely to
have, because a missing term and a zero term look identical in the output.
"""
abstract type ComponentStatus end

"""
    Measured()

The component was measured from the implemented update. See `ComponentStatus`.
"""
struct Measured <: ComponentStatus end

"""
    InvariantZero()

The component is provably zero, and its evidence names the proof. See
`ComponentStatus`.
"""
struct InvariantZero <: ComponentStatus end

"""
    NotApplicable()

The quantity does not exist for this path or reservoir in this configuration.
Excluded from every total, and never a measured zero. See `ComponentStatus`.
"""
struct NotApplicable <: ComponentStatus end

"""
    UnknownComponent()

The component has not been established, and blocks the claim it belongs to. See
`ComponentStatus`.
"""
struct UnknownComponent <: ComponentStatus end

"""
    status_name(status) -> Symbol

A short label for a `ComponentStatus`, used when a report tallies statuses
rather than naming each one.
"""
status_name(::Measured) = :measured
status_name(::InvariantZero) = :invariant_zero
status_name(::NotApplicable) = :not_applicable
status_name(::UnknownComponent) = :unknown

"""
    disposition_permits(expected, status) -> Bool

Whether a component with this `status` is what the schema's `expected`
disposition asked for. See `EXPECTED_DISPOSITIONS`.

  - `:open` permits anything at the record. The registry has not established
    what the path does, so nothing is demanded of a leg, and the claim the
    declaration feeds is blocked by the schema at reconciliation instead; see
    `open_dispositions`.
  - `:measured` permits a `Measured` component, and an `UnknownComponent`,
    which is the honest record of a measurement that was expected and not
    taken. It blocks, which is the point.
  - `:invariant_zero` permits an `InvariantZero`, and an `UnknownComponent`
    for a proof not yet established. A `Measured` component is **not**
    permitted: the registry says this path is provably zero, so a measurement
    here means the proof no longer holds or never did, which is a disagreement
    between the registry and the code rather than a residual.
  - `:not_applicable` permits only a `NotApplicable` component. There is
    nothing here to be unknown about.
"""
function disposition_permits(expected::Symbol, status::ComponentStatus)
    expected === :open && return true
    expected === :not_applicable && return status isa NotApplicable
    status isa UnknownComponent && return true
    expected === :measured && return status isa Measured
    expected === :invariant_zero && return status isa InvariantZero
    return error(
        "Unknown expected disposition $expected; expected one of " *
        "$(EXPECTED_DISPOSITIONS).",
    )
end

"""
    BudgetEvidence(; status, method, source, route)

How one component came to have the status it has.

Evidence is **per component**, never per leg. One event routinely measures
energy, proves a mass zero, and has nothing to say about water, so a single
per-leg record would misdescribe two of the three.

  - `status` is the `ComponentStatus`.
  - `method` is how the amount was obtained, or, for an `InvariantZero`, the
    proof that makes it zero.
  - `source` names the adapter or coverage-registry entry it came from.
  - `route` names the precision and reduction path, where that matters. An
    endpoint measured through the packed collective and a leg taken from an
    applied increment are different routes and a report should be able to say
    which.

This is what lets a failing budget be localized. Without it a missing leg, a
duplicated leg and a mismatched pair all look the same afterwards.

`method` defaults to `:unspecified`, which `BudgetComponent` refuses for the two
statuses that make an auditable claim. `source` and `route` are descriptive and
carry no such requirement.
"""
Base.@kwdef struct BudgetEvidence
    status::ComponentStatus
    method::Symbol = :unspecified
    source::Symbol = :unspecified
    route::Symbol = :unspecified
end

"""
    BudgetComponent(amount, evidence)

One signed extensive amount together with the evidence for it.

Units are kg for mass and water and J for energy, always extensive and never
normalized. Positive means addition to the reservoir the leg names.

The constructors `measured`, `invariant_zero`, `not_applicable` and
`unknown_component` are the convenient way in. The rules below live in the inner
constructor instead of in them, so a record built directly from
`BudgetComponent` and `BudgetEvidence` obeys the same rules as one built through
a helper.

**Only a measured component may carry a nonzero amount.** `is_contributing` is
true for `InvariantZero`, so a nonzero one would add a real amount into a total
while labelled as proven zero — a wrong number wearing the one label that says
it cannot be wrong. A nonzero `NotApplicable` or `UnknownComponent` amount is
never read, so it is dead data that misleads anyone inspecting a leg.

**A measured component must name its method.** An amount with no account of
where it came from is not auditable, and a budget whose terms cannot be traced
is a number rather than evidence.

**An invariant zero must name its proof.** A zero with no proof is an assumption,
and an assumed zero is exactly what the `UnknownComponent` status exists to keep
out of the totals.

**An amount must be finite.** `NaN` and `Inf` propagate through every sum and
reach a verdict as a comparison that is silently false, so a non-finite amount
is refused at construction rather than surfacing as a residual nobody can
attribute.
"""
struct BudgetComponent{FT}
    amount::FT
    evidence::BudgetEvidence
    function BudgetComponent{FT}(amount, evidence::BudgetEvidence) where {FT}
        a = convert(FT, amount)
        isfinite(a) || error(
            "A component amount must be finite, got $a. A NaN or Inf would " *
            "pass through every sum and reach the verdict as a comparison " *
            "that is silently false, so it is refused where it enters.",
        )
        status = evidence.status
        if !(status isa Measured) && !iszero(a)
            error(
                "A $(nameof(typeof(status))) component must carry exactly " *
                "zero, got $a. Only a Measured component may hold a nonzero " *
                "amount.",
            )
        end
        if status isa Measured && evidence.method === :unspecified
            error(
                "A Measured component must name the method its amount came " *
                "from. An unattributed amount cannot be audited, and a budget " *
                "whose terms cannot be traced is a number rather than " *
                "evidence.",
            )
        end
        if status isa InvariantZero && evidence.method === :unspecified
            error(
                "An InvariantZero component must name the proof that makes it " *
                "zero. A zero with no proof is an assumption, and an assumed " *
                "zero is what UnknownComponent exists to keep out of a total.",
            )
        end
        return new{FT}(a, evidence)
    end
end

BudgetComponent(amount::FT, evidence::BudgetEvidence) where {FT} =
    BudgetComponent{FT}(amount, evidence)

"""
    component_status(component) -> ComponentStatus

The status of one `BudgetComponent`.
"""
component_status(c::BudgetComponent) = c.evidence.status

"""
    component_method(component) -> Symbol

How the amount was obtained, or the proof of an `InvariantZero`.
"""
component_method(c::BudgetComponent) = c.evidence.method

"""
    component_source(component) -> Symbol

The adapter or coverage-registry entry the amount came from.
"""
component_source(c::BudgetComponent) = c.evidence.source

"""
    component_route(component) -> Symbol

The precision and reduction path the amount travelled.
"""
component_route(c::BudgetComponent) = c.evidence.route

"""
    measured(amount; method, source = :unspecified, route = :unspecified)

A `BudgetComponent` holding a measured signed amount. `method` is required: an
amount with no account of where it came from cannot be audited.
"""
measured(
    amount::FT;
    method::Symbol,
    source::Symbol = :unspecified,
    route::Symbol = :unspecified,
) where {FT} = BudgetComponent{FT}(
    amount,
    BudgetEvidence(; status = Measured(), method, source, route),
)

"""
    invariant_zero(FT; proof, source = :unspecified)

A `BudgetComponent` that is provably zero. The amount is exactly zero, which is
what makes it safe to include in a sum, and `proof` names why.
"""
invariant_zero(
    ::Type{FT};
    proof::Symbol,
    source::Symbol = :unspecified,
) where {FT} = BudgetComponent{FT}(
    zero(FT),
    BudgetEvidence(; status = InvariantZero(), method = proof, source),
)

"""
    not_applicable(FT; reason = :not_in_configuration, source = :contract)

A `BudgetComponent` for a quantity this configuration does not own. It
contributes nothing, blocks nothing, and is not a measured zero.
"""
not_applicable(
    ::Type{FT};
    reason::Symbol = :not_in_configuration,
    source::Symbol = :contract,
) where {FT} = BudgetComponent{FT}(
    zero(FT),
    BudgetEvidence(; status = NotApplicable(), method = reason, source),
)

"""
    unknown_component(FT; reason = :not_established, source = :unspecified)

A `BudgetComponent` that has not been established. It contributes nothing to a
sum and blocks the closure claim for its quantity.
"""
unknown_component(
    ::Type{FT};
    reason::Symbol = :not_established,
    source::Symbol = :unspecified,
) where {FT} = BudgetComponent{FT}(
    zero(FT),
    BudgetEvidence(; status = UnknownComponent(), method = reason, source),
)

"""
    is_contributing(component) -> Bool

Whether the component may be added into a total.

True for `Measured` and `InvariantZero`. False for `NotApplicable` and
`UnknownComponent`, whose amounts are zero anyway; the distinction matters
because only `UnknownComponent` also blocks.
"""
is_contributing(c::BudgetComponent) =
    component_status(c) isa Measured || component_status(c) isa InvariantZero

"""
    is_blocking(component) -> Bool

Whether the component blocks the claim for its quantity. True only for
`UnknownComponent`.
"""
is_blocking(c::BudgetComponent) = component_status(c) isa UnknownComponent

"""
    is_applicable(component) -> Bool

Whether the quantity exists here at all. False only for `NotApplicable`.
"""
is_applicable(c::BudgetComponent) = !(component_status(c) isa NotApplicable)

# ============================================================================
# Legs
# ============================================================================

"""
    BudgetLeg(; event, leg, reservoir, channel, level, mass, water, energy,
              path, process, phase, step, stage, occurrence, weight,
              measured_at)

One signed contribution to one reservoir, recorded once.

`event` is the stable identifier the coverage registry uses, and is shared by
every leg of one exchange, so the atmospheric and surface halves of a surface
flux carry the same `event` and different `leg`.

`channel` is one of `BUDGET_CHANNEL_LABELS` and `level` is a `CollectionLevel`.
Together they place the leg in exactly one of the three identities, and the
schema decides which of them this configuration expected.

# Execution identity

`(reservoir, channel, event, leg, step, stage, occurrence)` is what the journal
refuses to record twice. The reservoir is in it because the two sides of one
exchange may carry the same leg label — the atmospheric and surface halves of a
flux are each `flux` — and an identity without it would refuse the second side
as a repeat of the first. The stage and occurrence are in it because a
correction can fire several times within one accepted step:
`update_constrain_state_every` accepts `"stage"` and `"dss"`, and at `"stage"`
the same `constrain_state!` correction fires once per ARS343 stage. Each firing
is its own leg, so without a stage index the four would collide and three would
be refused as duplicates. `occurrence` covers a path that fires more than once
within a single stage.

`stage` is zero for anything that happens once per accepted step, and
`occurrence` counts from one.

# What the amounts mean

A leg's components are **accepted-step-weighted extensive contributions**: the
part of `Bⁿ⁺¹ - Bⁿ` this path is responsible for. They are not raw tendencies,
and they are not raw before/after differences taken on an intermediate stage
array. Those three are different numbers and must never be added together.

The distinction bites because `lim!`, `dss!` and `constrain_state!` run on
intermediate stage arrays as well as on the accepted state. A map applied to the
accepted state contributes its raw change, because that change *is* part of the
endpoint difference. A map applied to an intermediate stage does not: it changes
the array a later tendency evaluation reads, so the tableau mediates its effect
on the endpoint. Booking its raw difference here is wrong however plausible the
number looks, and a raw stage difference belongs in a `StageObservation`.

`weight` records the accepted-step coefficient already applied to reach the
contribution — `1` for a whole-step map, otherwise a tableau coefficient
generally involving `bᵢ` and the implicit `γᵢ`. `measured_at` records where the
amount was taken. Both exist so a weighting can be audited instead of trusted. A
non-finite `weight` is refused at construction.
"""
Base.@kwdef struct BudgetLeg{FT}
    event::Symbol
    leg::Symbol
    reservoir::BudgetReservoir
    channel::Symbol
    level::CollectionLevel
    mass::BudgetComponent{FT}
    water::BudgetComponent{FT}
    energy::BudgetComponent{FT}
    path::UpdatePath
    process::Symbol
    phase::Symbol
    step::Int
    stage::Int = 0
    occurrence::Int = 1
    weight::FT = one(FT)
    measured_at::Symbol = :accepted_state
    function BudgetLeg{FT}(
        event,
        leg,
        reservoir,
        channel,
        level,
        mass,
        water,
        energy,
        path,
        process,
        phase,
        step,
        stage,
        occurrence,
        weight,
        measured_at,
    ) where {FT}
        w = convert(FT, weight)
        isfinite(w) || error(
            "Leg $event/$leg has weight $w. An accepted-step coefficient is a " *
            "finite number; a NaN or Inf here would silently poison the " *
            "contribution it scales.",
        )
        return new{FT}(
            event,
            leg,
            reservoir,
            channel,
            level,
            mass,
            water,
            energy,
            path,
            process,
            phase,
            step,
            stage,
            occurrence,
            w,
            measured_at,
        )
    end
end

"""
    StageObservation(; event, observation, reservoir, mass, water, energy,
                     process, step, stage, occurrence)

A raw before/after difference taken on an intermediate stage array.

Deliberately **not** a `BudgetLeg`, and sharing no supertype with one. An
intermediate-stage change is not an additive contribution to the accepted
endpoint, so it has no place in any of the three identities, and the cheapest way
to guarantee that is to make it a type `record_leg!` will not accept and no
projection iterates.

It is still worth collecting. When a residual appears, knowing which stage and
which map moved the state is what turns "the step does not close" into a located
defect. It is evidence, not accounting.

There is no `channel`, `level`, `path` or `weight` field, because each of those
places a contribution inside an identity and an observation is not in one.
"""
Base.@kwdef struct StageObservation{FT}
    event::Symbol
    observation::Symbol
    reservoir::BudgetReservoir
    mass::BudgetComponent{FT}
    water::BudgetComponent{FT}
    energy::BudgetComponent{FT}
    process::Symbol
    step::Int
    stage::Int
    occurrence::Int = 1
end

"""
    execution_identity(leg) -> Tuple

The tuple the journal deduplicates on: reservoir, channel, event, leg, step,
stage, occurrence.

The reservoir and channel are part of it because a transfer event declares its
legs as `(reservoir, leg)` pairs and two of them may share a label. Without the
reservoir the surface side of a `flux` would be a duplicate of its atmospheric
side, and a schema the constructor accepts could never be recorded in full.

Deterministic, and stable across runs, so a leg can be named in a report and
found again.
"""
execution_identity(leg::BudgetLeg) = (
    reservoir_name(leg.reservoir),
    leg.channel,
    leg.event,
    leg.leg,
    leg.step,
    leg.stage,
    leg.occurrence,
)

"""
    leg_label(leg) -> String

A human-readable identity for `leg`, used when a report has to name which legs
blocked a claim. It names the reservoir, so the two sides of one exchange are
told apart.
"""
function leg_label(leg::BudgetLeg)
    location = "$(reservoir_name(leg.reservoir)) at step $(leg.step)"
    return "$(leg.event)/$(leg.leg) in $location stage $(leg.stage) #$(leg.occurrence)"
end

"""
    budget_component(record, quantity) -> BudgetComponent

The `quantity` component of a leg or observation, where `quantity` is `:mass`,
`:water`, or `:energy`.
"""
function budget_component(
    record::Union{BudgetLeg, StageObservation},
    quantity::Symbol,
)
    quantity === :mass && return record.mass
    quantity === :water && return record.water
    quantity === :energy && return record.energy
    error("Unknown budget quantity $quantity; expected :mass, :water or :energy")
end
