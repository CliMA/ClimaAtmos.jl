#####
##### Parent-budget ledger: transactions and the three reconciliations
#####
##### One transaction per accepted timestep. It opens on the finalized endpoint
##### of step `n`, collects legs, and closes on the finalized endpoint of step
##### `n+1`. Nothing is committed until it closes, so a step that never completes
##### leaves no entries behind.
#####
##### Three residuals answer three different questions and are never combined:
#####
#####   R_parent(q)         = ΔB(q) − Σ_c Q_envelope(q, c) − Σ_m Q_final_map(q, m)
#####   R_attribution(q, c) = Q_envelope(q, c) − Σ_{e ∈ c} Q(q, e)
#####   R_transfer(q, e)    = Σ_{r ∈ modeled(e)} Q(q, e, r)
#####
##### Every one of them is a subtraction or a sum of recorded amounts. No
##### function here creates a leg, so the ledger cannot close a budget it has not
##### accounted for, and nothing in this file knows what a ClimaAtmos process is
##### or how a collective is issued.
#####
##### What is expected comes from the schema and what happened comes from the
##### journal. Every enumeration below walks the schema, so a channel or event
##### that recorded nothing is a blocked row naming it rather than an absent one.

# ============================================================================
# Endpoints
# ============================================================================

"""
    ReservoirEndpoint(; reservoir, mass, water, energy)

One reservoir's three parent quantities at one instant, with the evidence for
each.

The components are independent: a slab measures energy, owns water only in a
moist configuration, and carries the mass that goes with that water.
"""
Base.@kwdef struct ReservoirEndpoint{FT}
    reservoir::BudgetReservoir
    mass::BudgetComponent{FT}
    water::BudgetComponent{FT}
    energy::BudgetComponent{FT}
end

function budget_component(endpoint::ReservoirEndpoint, quantity::Symbol)
    quantity === :mass && return endpoint.mass
    quantity === :water && return endpoint.water
    quantity === :energy && return endpoint.energy
    error("Unknown budget quantity $quantity; expected :mass, :water or :energy")
end

"""
    BudgetEndpoints(reservoirs, step)

Every declared reservoir's endpoint at the end of accepted step `step`, in the
schema's declaration order.
"""
struct BudgetEndpoints{FT}
    reservoirs::Vector{ReservoirEndpoint{FT}}
    step::Int
end

"""
    endpoint_reservoir(group) -> BudgetReservoir

Return the reservoir a packet group belongs to.

Errors for anything else, so a group the layout should never have produced fails
here rather than becoming a reservoir nothing else knows about.
"""
function endpoint_reservoir(group::Symbol)
    group === ATMOSPHERE_ENDPOINT_GROUP && return AtmosphereReservoir()
    group === SLAB_SURFACE_ENDPOINT_GROUP && return SlabSurfaceReservoir()
    return error("No budget reservoir for packet group $group.")
end

"""
    endpoint_component(packet, group, quantity, FT)

Read one endpoint component out of a reduced packet.

A measured slot becomes a `Measured` component naming the packed collective as
its route. A slot the schema declared not applicable becomes `NotApplicable`,
which is excluded from every total and is never a measured zero. An unset slot
has no reading at all and is refused, because unpacking a slot nothing wrote
would put a zero where a measurement should be.
"""
function endpoint_component(
    packet::BudgetPacket,
    group::Symbol,
    quantity::Symbol,
    ::Type{FT},
) where {FT}
    state = slot_state(packet, group, quantity)
    state isa UnsetSlot && error(
        "Budget packet slot $group/$quantity is unset and cannot be unpacked.",
    )
    if state isa NotApplicableSlot
        return not_applicable(
            FT;
            reason = :not_in_configuration,
            source = :schema,
        )
    end
    return measured(
        FT(packet_value(packet, group, quantity));
        method = :authoritative_state_integral,
        source = :parent_budget_integrals,
        route = :packed_global_reduction,
    )
end

"""
    budget_endpoints(packet, step)

Unpack a reduced packet into one endpoint per reservoir.

Refused while any slot is unresolved. Every slot must have an explicit
disposition before it is read, because an unset slot and a slot the
configuration does not own are different facts and only one of them is a defect.
"""
function budget_endpoints(packet::BudgetPacket, step::Int)
    check_packet_resolved(packet, "unpacked")
    FT = BUDGET_ACCOUNTING_TYPE
    reservoirs = ReservoirEndpoint{FT}[]
    for group in packet_groups(packet)
        push!(
            reservoirs,
            ReservoirEndpoint{FT}(;
                reservoir = endpoint_reservoir(group),
                mass = endpoint_component(packet, group, :mass, FT),
                water = endpoint_component(packet, group, :water, FT),
                energy = endpoint_component(packet, group, :energy, FT),
            ),
        )
    end
    return BudgetEndpoints{FT}(reservoirs, step)
end

"""
    budget_endpoints(Y, schema, surface_temperature, step)

Measure every declared reservoir's endpoint with **one** global collective and
unpack the result.
"""
budget_endpoints(Y, schema::BudgetSchema, surface_temperature, step::Int) =
    budget_endpoints(
        reduced_endpoint_packet(Y, schema, surface_temperature),
        step,
    )

"""
    endpoint_total(endpoints, quantity, control_volume)

Sum one quantity over the reservoirs of `control_volume`.

Returns `(; total, magnitude, applicable, blocked_by)`. A reservoir outside the
control volume is skipped entirely, so it can neither contribute nor block.

`applicable` is false when no reservoir in the view owns the quantity at all,
which is not the same as a total of zero. Water in a dry model would otherwise
be reported as an ordinary closed budget at zero. The ledger never made that
claim.

`magnitude` is the sum of absolute endpoint values, which the tolerance needs
and which a signed total cannot supply.

`blocked_by` names the reservoirs whose component was unknown, so a blocked
report says what would unblock it instead of only that it is blocked.
"""
function endpoint_total(
    endpoints::BudgetEndpoints{FT},
    quantity::Symbol,
    cv::ControlVolume,
) where {FT}
    total = zero(FT)
    magnitude = zero(FT)
    applicable = false
    blocked_by = String[]
    for endpoint in endpoints.reservoirs
        is_inside(cv, endpoint.reservoir) || continue
        c = budget_component(endpoint, quantity)
        is_applicable(c) && (applicable = true)
        if is_contributing(c)
            total += c.amount
            magnitude += abs(c.amount)
        end
        if is_blocking(c)
            push!(blocked_by, "$(reservoir_name(endpoint.reservoir)) endpoint")
        end
    end
    return (; total, magnitude, applicable, blocked_by)
end

"""
    check_endpoint_layout(opening, closing)

Verify that the closing endpoints describe the same reservoirs, in the same
order, owning the same quantities as the opening ones.

A supported configuration has a static reservoir graph, so a reservoir that
appears, disappears, or changes what it owns part way through a step is a
defect. Checking it here means a malformed closing endpoint is refused before
`commit_transaction!` has advanced anything, which is part of what lets the
commit be atomic.
"""
function check_endpoint_layout(opening::BudgetEndpoints, closing::BudgetEndpoints)
    length(opening.reservoirs) == length(closing.reservoirs) || error(
        "The budget reservoir set changed within step $(closing.step), from " *
        "$(length(opening.reservoirs)) reservoirs to " *
        "$(length(closing.reservoirs)).",
    )
    for (before, after) in zip(opening.reservoirs, closing.reservoirs)
        before.reservoir === after.reservoir ||
            error("The budget reservoir order changed within a step.")
        for quantity in BUDGET_QUANTITIES
            b = component_status(budget_component(before, quantity))
            a = component_status(budget_component(after, quantity))
            typeof(b) === typeof(a) || error(
                "Budget component status for $quantity in " *
                "$(reservoir_name(after.reservoir)) changed within step " *
                "$(closing.step), from $(nameof(typeof(b))) to " *
                "$(nameof(typeof(a))).",
            )
        end
    end
    return nothing
end

"""
    check_schema_endpoints(schema, endpoints, what)

Verify that `endpoints` describes exactly the reservoirs the schema declares, in
declaration order, owning exactly the quantities the schema says they own.

The schema is what every enumeration and every packet layout is built from, so
endpoints that disagree with it would be reconciled against expectations
belonging to a different configuration. `what` names which endpoints are being
checked, so the message says whether the opening or the closing reading is the
one that disagrees.
"""
function check_schema_endpoints(
    schema::BudgetSchema,
    endpoints::BudgetEndpoints,
    what::AbstractString,
)
    declared = schema_reservoir_names(schema)
    length(endpoints.reservoirs) == length(declared) || error(
        "The $what budget endpoints hold $(length(endpoints.reservoirs)) " *
        "reservoirs, but the schema declares $(length(declared)).",
    )
    for (endpoint, name) in zip(endpoints.reservoirs, declared)
        reservoir_name(endpoint.reservoir) === name || error(
            "The $what budget endpoints name reservoir " *
            "$(reservoir_name(endpoint.reservoir)) where the schema declares " *
            "$name. The declaration order fixes the packet layout, so the two " *
            "cannot differ.",
        )
        for quantity in BUDGET_QUANTITIES
            component = budget_component(endpoint, quantity)
            expected = quantity_applicable(schema, name, quantity)
            is_applicable(component) == expected || error(
                "The $what budget endpoint for $quantity in $name is " *
                "$(is_applicable(component) ? "applicable" : "not applicable") " *
                "where the schema declares the opposite. Applicability comes " *
                "from the configuration, so a disagreement means the endpoints " *
                "and the schema describe different configurations.",
            )
        end
    end
    return nothing
end

# ============================================================================
# Tolerance
# ============================================================================

"""
    BudgetTolerance(; absolute, relative, scale, kappa)

The tolerance one quantity's residual is judged against.

```
τ = absolute + relative · scale + kappa · eps(FT) · (abs(Bⁿ) + abs(Bⁿ⁺¹) + Σ abs(Q))
```

  - `absolute` is a floor in the units of the quantity, kg or J.
  - `relative` is dimensionless.
  - `scale` is a positive scale for the relative term. It is **never** a signed
    total: a signed scale can pass through zero, and it hides the sign the
    ledger exists to expose.
  - `kappa` covers reduction order and rank dependence.

Every field must be finite, and `scale` must be strictly positive. A `NaN`
tolerance compares false against everything and turns every step into a failure;
an infinite one compares true against everything and turns every step into a
pass. Both are worse than having no tolerance at all, which at least reports
`blocked`.

`kappa` has no default on purpose. It must be calibrated against measured serial
and distributed runs and recorded with the result, and a guessed value presented
as universal is not a tolerance. Until it is calibrated, a reconciliation
computed without a tolerance reports `blocked` rather than `pass`.
"""
struct BudgetTolerance{FT}
    absolute::FT
    relative::FT
    scale::FT
    kappa::FT
    function BudgetTolerance{FT}(absolute, relative, scale, kappa) where {FT}
        check_tolerance_field(absolute, "absolute")
        check_tolerance_field(relative, "relative")
        check_tolerance_field(scale, "scale")
        check_tolerance_field(kappa, "kappa")
        scale > 0 || error(
            "BudgetTolerance scale must be positive, got $scale. The relative " *
            "term is a fraction of a magnitude, and a zero magnitude is not " *
            "one.",
        )
        return new{FT}(absolute, relative, scale, kappa)
    end
end

# A tolerance field has to be a finite non-negative number. Non-finite values
# are singled out because they fail silently in the comparison rather than at
# construction: NaN rejects every step and Inf accepts every step.
function check_tolerance_field(value, name::AbstractString)
    isfinite(value) || error(
        "BudgetTolerance $name must be finite, got $value. A non-finite " *
        "tolerance decides every comparison the same way whatever the residual " *
        "is.",
    )
    value >= 0 || error(
        "BudgetTolerance $name must be non-negative, got $value.",
    )
    return nothing
end

function BudgetTolerance(; absolute, relative, scale, kappa)
    a, r, s, k = promote(absolute, relative, scale, kappa)
    return BudgetTolerance{typeof(a)}(a, r, s, k)
end

"""
    tolerance_value(tolerance, opening, closing, recorded_magnitude)

Evaluate the tolerance for one step.

The endpoint magnitudes are inside the arithmetic term deliberately. Bounding a
residual by the recorded magnitudes alone is a stricter claim than the endpoint
subtraction supports, and it fails on a step whose legs are tiny against the
background.
"""
function tolerance_value(
    tolerance::BudgetTolerance{FT},
    opening,
    closing,
    recorded_magnitude,
) where {FT}
    relative_term = tolerance.relative * tolerance.scale
    magnitudes = abs(opening) + abs(closing) + recorded_magnitude
    arithmetic_term = tolerance.kappa * eps(FT) * magnitudes
    return tolerance.absolute + relative_term + arithmetic_term
end

"""
    claim_status(applicable, blocked_by, residual, tolerance) -> Symbol

Return the verdict for one claim, resolved in this order:

 1. `:not_applicable` when nothing in the view owns the quantity, so there is no
    claim to make.
 2. `:blocked` when required evidence is unavailable, whatever the numbers look
    like. A missing tolerance blocks for the same reason a missing measurement
    does.
 3. `:fail` when complete evidence violates the tolerance.
 4. `:pass` when complete evidence satisfies it.

The order matters and is the same everywhere. A blocked claim is not a failed
one, and neither is a claim the configuration never made.
"""
function claim_status(applicable::Bool, blocked_by, residual, tolerance)
    applicable || return :not_applicable
    isempty(blocked_by) || return :blocked
    isnothing(tolerance) && return :blocked
    return abs(residual) <= tolerance ? :pass : :fail
end

# The blocker a reconciliation carries when it was asked for a verdict with no
# calibrated tolerance. Written once so a report can match on it.
const UNCALIBRATED_TOLERANCE_BLOCKER = "tolerance not declared; kappa is uncalibrated"

# ============================================================================
# The ledger
# ============================================================================

"""
    BudgetLedger{FT}(schema)

The declared expectations, the open transaction, and the running cumulative
totals.

A ledger is opened on an endpoint, collects legs, and is committed on the next
endpoint. Its `schema` is fixed at construction and never changes: it is what
every recording is checked against and what every reconciliation is enumerated
from, so a channel or event that recorded nothing still produces a row.

Three cumulative residuals are kept per quantity and control volume, not one,
because a signed sum cancels the very failure the ledger exists to expose: `+δ`
on one step and `−δ` on the next sums to zero and reports a perfectly closed run
that closed on neither step. `cumulative_residual` is the signed drift.
`cumulative_abs_residual` cannot cancel and bounds the total unaccounted
transfer. `max_abs_residual` names the worst single step. Closure quality is
decided by the last two; the signed sum is reported and never passes a test on
its own.

The ledger is a diagnostic. Nothing in it writes to the state, and a run with it
enabled must produce the same trajectory as one without it.
"""
mutable struct BudgetLedger{FT}
    schema::BudgetSchema
    step::Int
    is_open::Bool
    opening::Union{Nothing, BudgetEndpoints{FT}}
    initial::Union{Nothing, BudgetEndpoints{FT}}
    last_closing::Union{Nothing, BudgetEndpoints{FT}}
    legs::Vector{BudgetLeg{FT}}
    observations::Vector{StageObservation{FT}}
    recorded_keys::Set{ExecutionIdentity}
    envelope_keys::Set{Tuple{Symbol, Symbol, Int}}
    event_levels::Dict{Tuple{Symbol, Int}, Symbol}
    cumulative_residual::Dict{Tuple{Symbol, Symbol}, FT}
    cumulative_abs_residual::Dict{Tuple{Symbol, Symbol}, FT}
    max_abs_residual::Dict{Tuple{Symbol, Symbol}, FT}
    cumulative_change::Dict{Tuple{Symbol, Symbol}, FT}
    committed_steps::Int
end

BudgetLedger{FT}(schema::BudgetSchema) where {FT} = BudgetLedger{FT}(
    schema,
    0,
    false,
    nothing,
    nothing,
    nothing,
    BudgetLeg{FT}[],
    StageObservation{FT}[],
    Set{ExecutionIdentity}(),
    Set{Tuple{Symbol, Symbol, Int}}(),
    Dict{Tuple{Symbol, Int}, Symbol}(),
    Dict{Tuple{Symbol, Symbol}, FT}(),
    Dict{Tuple{Symbol, Symbol}, FT}(),
    Dict{Tuple{Symbol, Symbol}, FT}(),
    Dict{Tuple{Symbol, Symbol}, FT}(),
    0,
)

"""
    clear_open_transaction!(ledger)

Drop everything belonging to the open transaction.

Per-step storage is bounded because this runs on every commit and every abort. A
long run keeps the fixed set of cumulative totals and nothing that grows with
the number of steps.
"""
function clear_open_transaction!(ledger::BudgetLedger)
    empty!(ledger.legs)
    empty!(ledger.observations)
    empty!(ledger.recorded_keys)
    empty!(ledger.envelope_keys)
    empty!(ledger.event_levels)
    return nothing
end

"""
    check_endpoint_continuity(ledger, opening)

Verify that `opening` is the endpoint the previous transaction closed on.

The first transaction has nothing to compare against and passes. Afterwards the
step numbers, the reservoir set, every component's **status**, and every
contributing amount must agree exactly. Exactly, not within a tolerance: a
reduction over an unchanged state is deterministic, so any difference is a real
change that happened between the two transactions and that nothing recorded.

Status is checked as well as amount because comparing amounts alone lets a
reservoir change what it owns without anyone noticing. A quantity that was
measured and is now unknown, or was inapplicable and is now measured, would slip
through a check that skips any pair where one side does not contribute. That is
precisely the pair a status change produces. A configuration transition mid-run
is not supported, so a status change is a defect until it is deliberately
represented.
"""
function check_endpoint_continuity(
    ledger::BudgetLedger{FT},
    opening::BudgetEndpoints{FT},
) where {FT}
    previous = ledger.last_closing
    isnothing(previous) && return nothing
    previous.step == opening.step || error(
        "Budget transaction opens on step $(opening.step) but the previous " *
        "one closed on step $(previous.step).",
    )
    length(previous.reservoirs) == length(opening.reservoirs) || error(
        "The budget reservoir set changed between transactions, from " *
        "$(length(previous.reservoirs)) reservoirs to " *
        "$(length(opening.reservoirs)).",
    )
    for (before, after) in zip(previous.reservoirs, opening.reservoirs)
        before.reservoir === after.reservoir ||
            error("The budget reservoir order changed between transactions.")
        for quantity in BUDGET_QUANTITIES
            b = budget_component(before, quantity)
            a = budget_component(after, quantity)
            typeof(component_status(b)) === typeof(component_status(a)) || error(
                "Budget endpoint status changed for $quantity in " *
                "$(reservoir_name(after.reservoir)) at step $(opening.step): " *
                "the previous transaction closed as " *
                "$(nameof(typeof(component_status(b)))) and this one opens as " *
                "$(nameof(typeof(component_status(a)))). What a reservoir owns " *
                "may not change mid-run without being represented as a " *
                "transition.",
            )
            is_contributing(a) || continue
            b.amount == a.amount || error(
                "Budget endpoint discontinuity in $quantity for " *
                "$(reservoir_name(after.reservoir)) at step $(opening.step): " *
                "the previous transaction closed on $(b.amount) and this one " *
                "opens on $(a.amount), a gap of $(a.amount - b.amount) that " *
                "no transaction accounts for.",
            )
        end
    end
    return nothing
end

"""
    open_transaction!(ledger, endpoints)

Begin the transaction for the step `endpoints.step + 1`.

`endpoints` is the state after accepted step `n` was finalized, and the
transaction it opens ends after accepted step `n + 1` is finalized. Opening
while a transaction is already open is an error, because it would mean the
previous step neither committed nor aborted.

The endpoints are checked against the schema and against the previous
transaction's closing reading rather than assumed. A gap between transactions is
a change that nothing accounted for, and it would otherwise disappear from the
cumulative total without leaving a residual anywhere.

The first opening endpoint is also kept as the initial one, so that `Bᴺ − B⁰`
can later be read directly rather than telescoped out of the per-step
differences.
"""
function open_transaction!(
    ledger::BudgetLedger{FT},
    endpoints::BudgetEndpoints{FT},
) where {FT}
    ledger.is_open && error(
        "A budget transaction for step $(ledger.step) is already open. " *
        "Commit or abort it before opening another.",
    )
    check_schema_endpoints(ledger.schema, endpoints, "opening")
    check_endpoint_continuity(ledger, endpoints)
    ledger.step = endpoints.step + 1
    ledger.is_open = true
    ledger.opening = endpoints
    isnothing(ledger.initial) && (ledger.initial = endpoints)
    clear_open_transaction!(ledger)
    return nothing
end

"""
    open_transaction!(ledger)

Open the next transaction on the endpoint the previous one closed, without
measuring the state again.

Measuring the opening endpoint repeats, identically, the reduction the previous
commit already performed, so reusing it halves the endpoint collectives per
step.

**It also gives up the only check that would catch an unrecorded change between
steps.** `check_endpoint_continuity` exists because the closing state of step `n`
and the opening state of step `n+1` are read at different moments with discrete
callbacks in between, and comparing them is what turns a callback that quietly
mutates `Y` into an error instead of a silent gap in the cumulative total. Reuse
makes that comparison compare a value with itself.

Nothing in the ledger chooses between the two openings. The caller takes the
trade, and it is sound exactly while no callback mutates the state. That premise
is a property of the model the coverage registry establishes, not of the ledger.

Errors when there is no previous closing endpoint, which is the first step.
"""
function open_transaction!(ledger::BudgetLedger)
    previous = ledger.last_closing
    isnothing(previous) && error(
        "No previous closing endpoint to reuse; the first transaction has to " *
        "measure its opening endpoint.",
    )
    return open_transaction!(ledger, previous)
end

# ============================================================================
# Recording
# ============================================================================

# A leg's components have to be what the declaration said they would be. The
# proof obligations in the coverage registry are worth nothing if a row can
# declare a quantity provably zero and then record a measurement of it, so the
# disagreement is refused where it happens rather than surfacing as a residual
# nobody can attribute.
function check_leg_dispositions(spec, leg::BudgetLeg)
    for quantity in BUDGET_QUANTITIES
        expected = expected_disposition(spec, quantity)
        status = component_status(budget_component(leg, quantity))
        disposition_permits(expected, status) || error(
            "Leg $(leg_label(leg)) records $quantity as " *
            "$(status_name(status)), but the schema declares it $expected. A " *
            "declared disposition is a proof obligation about the code, so a " *
            "record that contradicts it is a disagreement between the registry " *
            "and the implementation rather than a residual.",
        )
    end
    return nothing
end

# Every leg is checked against the schema before it is stored. A record the
# schema does not declare is refused rather than becoming a new row, because a
# row nothing expected is a row nothing will check.
function check_leg_declared(schema::BudgetSchema, leg::BudgetLeg)
    reservoir = reservoir_name(leg.reservoir)
    has_reservoir(schema, reservoir) || error(
        "Leg $(leg_label(leg)) names reservoir $reservoir, which the schema " *
        "does not declare. Expectations come from the configuration, so a " *
        "reservoir nothing declared has no endpoint to reconcile against.",
    )
    if leg.level isa FinalMap
        has_final_map(schema, leg.channel) || error(
            "Leg $(leg_label(leg)) records final map $(leg.channel), which the " *
            "schema does not declare.",
        )
        spec = final_map_spec(schema, leg.channel)
        reservoir in spec.reservoirs || error(
            "Leg $(leg_label(leg)) records final map $(leg.channel) in " *
            "$reservoir, which that map does not declare.",
        )
        check_leg_dispositions(spec, leg)
        return nothing
    end
    if leg.level isa ReservoirTransfer
        has_transfer_event(schema, leg.event) || error(
            "Leg $(leg_label(leg)) records transfer event $(leg.event), which " *
            "the schema does not declare. An event's topology decides which " *
            "test applies to it, and an undeclared event has no topology.",
        )
        spec = transfer_event_spec(schema, leg.event)
        (reservoir, leg.leg) in spec.modeled_legs || error(
            "Leg $(leg_label(leg)) is not one of the modeled legs event " *
            "$(leg.event) declares. No leg is ever created for an exterior " *
            "counterparty, and a modeled leg the schema did not declare would " *
            "take part in a cancellation nobody expected.",
        )
        leg.channel === spec.channel || error(
            "Leg $(leg_label(leg)) names channel $(leg.channel), but event " *
            "$(leg.event) is declared in channel $(spec.channel).",
        )
        check_leg_dispositions(spec, leg)
        return nothing
    end
    has_channel(schema, leg.channel) || error(
        "Leg $(leg_label(leg)) names channel $(leg.channel), which the schema " *
        "does not declare as an accepted integrator channel.",
    )
    spec = channel_spec(schema, leg.channel)
    reservoir in spec.reservoirs || error(
        "Leg $(leg_label(leg)) records channel $(leg.channel) in $reservoir, " *
        "which that channel does not write.",
    )
    if leg.level isa ProcessDecomposition
        (leg.process, reservoir) in spec.processes || error(
            "Leg $(leg_label(leg)) decomposes channel $(leg.channel) with " *
            "process $(leg.process) in $reservoir, which the channel does not " *
            "declare. A row the registry did not declare is a row nothing " *
            "checks for, so an omitted process could never be missed.",
        )
    end
    check_leg_dispositions(spec, leg)
    return nothing
end

"""
    record_leg!(ledger, leg)

Add `leg` to the open transaction.

Refused, loudly, in these cases.

  - There is no open transaction, so the leg belongs to no accepted step.
  - The leg's `step` disagrees with the open one, which means a stage or a
    callback is writing into the wrong transaction.
  - The schema does not declare the leg's reservoir, channel, final map, or
    transfer event, or declares the event with different legs or in a different
    channel, or does not name the leg's process among the channel's declared
    decomposition rows. Expectations come from the configuration, so a record
    nothing declared fails closed.
  - A component contradicts the disposition its declaration gave it, such as a
    measurement on a quantity the registry says the path leaves provably zero.
  - A leg with the same `execution_identity` is already recorded, which is how a
    bracket that fires twice at the same point shows up. A correction that
    legitimately fires once per stage carries a different `stage` and is not a
    duplicate.
  - The same event is already recorded at a different `CollectionLevel`, or a
    second envelope is offered for a channel and reservoir that already has one.

The last one is worth stating precisely, because the contract's rule is about
sums rather than about recording. An envelope and its decomposition are
*supposed* to be recorded together: comparing them is the attribution identity.
What must never happen is both landing in one total, and that is prevented by
`enters_parent_identity` rather than by refusing the recording. What is refused
here is one event claiming to be two different kinds of thing, which is a
classification error and would make both identities wrong.

The duplicate guard is kept even though a duplicated nonzero leg already moves
the residual and fails closure, for three reasons. It localizes the fault at the
second recording
rather than in a residual a whole step later. It catches a doubled leg whose
amount happens to be zero, which genuinely would pass every closure test. And it
forces each firing of a repeating path to carry a distinct execution identity,
which is what makes a per-stage correction legible at all.
"""
function record_leg!(ledger::BudgetLedger{FT}, leg::BudgetLeg{FT}) where {FT}
    ledger.is_open || error(
        "No open budget transaction; cannot record leg $(leg.event)/$(leg.leg).",
    )
    leg.step == ledger.step || error(
        "Leg $(leg.event)/$(leg.leg) is for step $(leg.step), but the open " *
        "transaction is step $(ledger.step).",
    )
    check_leg_declared(ledger.schema, leg)

    key = execution_identity(leg)
    key in ledger.recorded_keys && error(
        "Leg $(leg_label(leg)) is already recorded. A leg is recorded once; " *
        "control-volume totals are projections of it. A path that legitimately " *
        "fires more than once in a step distinguishes its firings with `stage` " *
        "and `occurrence`.",
    )

    event_key = (leg.event, leg.step)
    level = level_name(leg.level)
    recorded_level = get(ledger.event_levels, event_key, level)
    recorded_level === level || error(
        "Leg $(leg_label(leg)) records event $(leg.event) at level $level, but " *
        "it is already recorded at level $recorded_level in this step. An " *
        "event is one kind of thing: an envelope, a decomposition, a final " *
        "map, or a transfer leg.",
    )

    if leg.level isa ChannelEnvelope
        envelope_key = (leg.channel, reservoir_name(leg.reservoir), leg.step)
        envelope_key in ledger.envelope_keys && error(
            "Leg $(leg_label(leg)) is a second envelope for channel " *
            "$(leg.channel) in $(reservoir_name(leg.reservoir)) at step " *
            "$(leg.step). A channel applies one accepted update to one " *
            "reservoir, so a second envelope for it double-counts that update.",
        )
        push!(ledger.envelope_keys, envelope_key)
    end

    ledger.event_levels[event_key] = level
    push!(ledger.recorded_keys, key)
    push!(ledger.legs, leg)
    return nothing
end

"""
    record_observation!(ledger, observation)

Add a `StageObservation` to the open transaction's audit trail.

Observations are evidence, never accounting. They are stored apart from the
legs, no projection iterates them, and no residual is computed from them, so
recording one cannot move a budget. That is the whole reason the type is
separate: an intermediate-stage difference is not an additive contribution to
the accepted endpoint, and the safest way to keep it out of the identities is to
make it something `record_leg!` will not take.

Unlike a leg, an observation is not deduplicated. Observing the same map at the
same stage twice is a reading, not a double count.
"""
function record_observation!(
    ledger::BudgetLedger{FT},
    observation::StageObservation{FT},
) where {FT}
    ledger.is_open || error(
        "No open budget transaction; cannot record observation " *
        "$(observation.event)/$(observation.observation).",
    )
    observation.step == ledger.step || error(
        "Observation $(observation.event)/$(observation.observation) is for " *
        "step $(observation.step), but the open transaction is step " *
        "$(ledger.step).",
    )
    push!(ledger.observations, observation)
    return nothing
end

"""
    abort_transaction!(ledger)

Discard the open transaction without committing anything.

This exists for the rejected-attempt rule, not because a supported configuration
reaches it. `args_integrator` passes a fixed `dt` and no controller, so the IMEX
integrator never rejects a step and never retries. If an adaptive controller is
added, this is the hook that keeps a rejected attempt from committing, and the
rollback of the *state* becomes real work that the contract does not currently
cover.
"""
function abort_transaction!(ledger::BudgetLedger)
    ledger.is_open = false
    ledger.opening = nothing
    clear_open_transaction!(ledger)
    return nothing
end

# ============================================================================
# Projections
# ============================================================================

"""
    project_parent(ledger, quantity, control_volume)

Sum the primary identity's recorded terms for one quantity over one control
volume.

Returns `(; envelopes, final_maps, recorded, magnitude, blocked_by)`.
Applicability is not among them: whether a view owns a quantity at all is a
property of its endpoints and of the schema, not of which legs arrived.

Only `ChannelEnvelope` and `FinalMap` legs are summed. A decomposition or
transfer leg explains an envelope rather than adding to it, so including it here
would count the same update twice; that is where the rule against summing an
aggregate with its own decomposition is enforced.

A leg outside the control volume is skipped, which is what makes a surface
exchange a boundary crossing in one view and an internal transfer in another
without recording it twice.
"""
function project_parent(
    ledger::BudgetLedger{FT},
    quantity::Symbol,
    cv::ControlVolume,
) where {FT}
    envelopes = zero(FT)
    final_maps = zero(FT)
    magnitude = zero(FT)
    blocked_by = String[]
    for leg in ledger.legs
        is_inside(cv, leg.reservoir) || continue
        enters_parent_identity(leg.level) || continue
        c = budget_component(leg, quantity)
        is_blocking(c) && push!(blocked_by, leg_label(leg))
        is_contributing(c) || continue
        magnitude += abs(c.amount)
        if leg.level isa ChannelEnvelope
            envelopes += c.amount
        else
            final_maps += c.amount
        end
    end
    recorded = envelopes + final_maps
    return (; envelopes, final_maps, recorded, magnitude, blocked_by)
end

"""
    missing_parent_terms(ledger, control_volume) -> Vector{String}

Return every term the schema declares for the primary identity in this view that
no leg recorded.

This is what keeps a channel that never reported from disappearing from the
report. The list is built by walking the schema's declarations, so a term that
was expected and never arrived is named here whether or not anything else in the
step referred to it.
"""
function missing_parent_terms(ledger::BudgetLedger, cv::ControlVolume)
    missing_terms = String[]
    for spec in ledger.schema.channels
        append!(missing_terms, missing_channel_envelopes(ledger, spec, cv))
    end
    for spec in ledger.schema.final_maps
        for reservoir in spec.reservoirs
            is_inside(cv, reservoir) || continue
            has_final_map_leg(ledger, spec.name, reservoir) && continue
            push!(
                missing_terms,
                "expected final map $(spec.name) in $reservoir was not recorded",
            )
        end
    end
    return missing_terms
end

"""
    missing_channel_envelopes(ledger, spec, control_volume) -> Vector{String}

Return the envelopes `spec` requires in this view that no leg recorded, one per
reservoir the channel writes.

Checked per reservoir rather than per channel, so a channel that recorded one of
its two envelopes is still short one term and says which. The parent and the
attribution reconciliations read the same list, so they cannot disagree about
whether a channel reported.
"""
function missing_channel_envelopes(
    ledger::BudgetLedger,
    spec::ChannelSpec,
    cv::ControlVolume,
)
    missing_envelopes = String[]
    spec.requires_envelope || return missing_envelopes
    for reservoir in spec.reservoirs
        is_inside(cv, reservoir) || continue
        has_envelope(ledger, spec.name, reservoir) && continue
        push!(
            missing_envelopes,
            "expected envelope for channel $(spec.name) in $reservoir was not " *
            "recorded",
        )
    end
    return missing_envelopes
end

# Whether an envelope for this channel and reservoir was recorded in the open
# transaction.
has_envelope(ledger::BudgetLedger, channel::Symbol, reservoir::Symbol) =
    (channel, reservoir, ledger.step) in ledger.envelope_keys

# Whether a final-map leg for this map and reservoir was recorded.
function has_final_map_leg(
    ledger::BudgetLedger,
    name::Symbol,
    reservoir::Symbol,
)
    for leg in ledger.legs
        leg.level isa FinalMap || continue
        leg.channel === name || continue
        reservoir_name(leg.reservoir) === reservoir && return true
    end
    return false
end

"""
    missing_processes(ledger, spec, control_volume) -> Vector{String}

Return the decomposition rows `spec` declares in this view that no leg recorded,
one per `(process, reservoir)` in the channel's roster.

Read from the specification and never from what arrived. A decomposition that
was satisfied by whichever rows happened to be recorded would let an omitted
process vanish: the rows present can cancel exactly and the residual says
nothing about the one that is not there. Each declared row is named, so a
channel that recorded some of its processes is short by the rest and says
which.
"""
function missing_processes(
    ledger::BudgetLedger,
    spec::ChannelSpec,
    cv::ControlVolume,
)
    missing_rows = String[]
    for (process, reservoir) in spec.processes
        is_inside(cv, reservoir) || continue
        has_process_leg(ledger, spec.name, process, reservoir) && continue
        push!(
            missing_rows,
            "expected process $process of channel $(spec.name) in $reservoir " *
            "was not recorded",
        )
    end
    return missing_rows
end

# Whether a decomposition leg for this channel, process and reservoir was
# recorded in the open transaction. Any status counts: a row recorded as not
# applicable is a record, and an unknown one blocks on its own.
function has_process_leg(
    ledger::BudgetLedger,
    channel::Symbol,
    process::Symbol,
    reservoir::Symbol,
)
    for leg in ledger.legs
        leg.level isa ProcessDecomposition || continue
        leg.channel === channel || continue
        leg.process === process || continue
        reservoir_name(leg.reservoir) === reservoir && return true
    end
    return false
end

"""
    open_dispositions(specs, quantity, control_volume) -> Vector{String}

Return every declaration in `specs` that touches `control_volume` and whose
disposition for `quantity` is still `:open`, as blockers.

An open row is one the coverage registry has not established from the code. It
demands nothing of a record, so a leg on it is accepted, but the claim it feeds
cannot be evaluated: the registry does not know what the path writes, so no sum
that includes it proves anything. A zero residual over an open row is a
coincidence and not a closure, and the row blocks until it is declared. This is
read from the schema alone, so a declaration that recorded nothing blocks
exactly as one that recorded a measurement does.
"""
function open_dispositions(specs, quantity::Symbol, cv::ControlVolume)
    blockers = String[]
    for spec in specs
        touches_view(spec, cv) || continue
        expected_disposition(spec, quantity) === :open || continue
        push!(
            blockers,
            "$(spec_kind(spec)) $(spec.name) leaves $quantity open in " *
            "$(cv.name); the registry has not established what it writes",
        )
    end
    return blockers
end

touches_view(spec::ChannelSpec, cv::ControlVolume) =
    any(r -> is_inside(cv, r), spec.reservoirs)
touches_view(spec::FinalMapSpec, cv::ControlVolume) =
    any(r -> is_inside(cv, r), spec.reservoirs)
touches_view(spec::TransferEventSpec, cv::ControlVolume) = event_in_view(spec, cv)

spec_kind(::ChannelSpec) = "channel"
spec_kind(::FinalMapSpec) = "final map"
spec_kind(::TransferEventSpec) = "transfer event"

"""
    declared_applicable(schema, reservoirs, quantity, control_volume) -> Bool

Return whether the configuration says any of `reservoirs` inside
`control_volume` owns `quantity`.

Applicability is read from the schema and never from whether a leg arrived. An
expected channel or event that recorded nothing is a **blocked** claim, not a
claim the configuration never made. Those are different answers and only one of
them is a defect, so deriving one from the absence of records would report every
silent gap as a quantity nobody has.
"""
function declared_applicable(
    schema::BudgetSchema,
    reservoirs,
    quantity::Symbol,
    cv::ControlVolume,
)
    for reservoir in reservoirs
        is_inside(cv, reservoir) || continue
        quantity_applicable(schema, reservoir, quantity) && return true
    end
    return false
end

"""
    project_attribution(ledger, quantity, control_volume, channel)

Sum one channel's envelope and its explaining legs for one quantity over one
control volume.

Returns `(; envelope, attributed, magnitude, explaining_count, blocked_by)`.
Applicability is not among them: it comes from the schema, through
`declared_applicable`, rather than from which legs arrived. Whether the channel
recorded its required envelope and every row of its declared decomposition is
likewise read from the schema, by `missing_channel_envelopes` and
`missing_processes`. `explaining_count` counts the contributing rows and is
informational; a count is satisfied by whichever rows arrive, so nothing is
decided by it.
"""
function project_attribution(
    ledger::BudgetLedger{FT},
    quantity::Symbol,
    cv::ControlVolume,
    channel::Symbol,
) where {FT}
    envelope = zero(FT)
    attributed = zero(FT)
    magnitude = zero(FT)
    explaining_count = 0
    blocked_by = String[]
    for leg in ledger.legs
        leg.channel === channel || continue
        is_inside(cv, leg.reservoir) || continue
        c = budget_component(leg, quantity)
        is_blocking(c) && push!(blocked_by, leg_label(leg))
        if leg.level isa ChannelEnvelope
            is_contributing(c) || continue
            envelope += c.amount
            magnitude += abs(c.amount)
        elseif explains_envelope(leg.level)
            is_contributing(c) || continue
            explaining_count += 1
            attributed += c.amount
            magnitude += abs(c.amount)
        end
    end
    return (; envelope, attributed, magnitude, explaining_count, blocked_by)
end

"""
    project_transfer(ledger, spec, quantity, control_volume)

Sum the recorded legs of one declared event for one quantity over one control
volume.

Returns `(; total, magnitude, leg_count, status_counts, blocked_by, missing_legs)`.

Applicability comes from the schema, through `declared_applicable`, not from
this projection.

Four answers have to stay distinguishable, and a bare total tells none of them
apart. "The legs cancel" is a total of zero with every declared leg present and
no blockers. "A declared leg is missing" is a non-empty `missing_legs`. "Nobody
measured the legs" is legs found with `blocked_by` naming them. And "this
quantity does not exist for this event", as water does not for a dry-model
exchange, is the schema saying so, which would otherwise look exactly like a
measured cancellation.

`missing_legs` comes from the specification, not from what arrived, so a leg
that was declared and never recorded blocks the event rather than being read as
a zero.
"""
function project_transfer(
    ledger::BudgetLedger{FT},
    spec::TransferEventSpec,
    quantity::Symbol,
    cv::ControlVolume,
) where {FT}
    total = zero(FT)
    magnitude = zero(FT)
    leg_count = 0
    status_counts = Dict(
        :measured => 0,
        :invariant_zero => 0,
        :not_applicable => 0,
        :unknown => 0,
    )
    blocked_by = String[]
    for leg in ledger.legs
        leg.event === spec.name || continue
        is_inside(cv, leg.reservoir) || continue
        leg_count += 1
        c = budget_component(leg, quantity)
        status_counts[status_name(component_status(c))] += 1
        is_blocking(c) && push!(blocked_by, leg_label(leg))
        is_contributing(c) || continue
        total += c.amount
        magnitude += abs(c.amount)
    end
    missing_legs = String[]
    for (reservoir, name) in spec.modeled_legs
        is_inside(cv, reservoir) || continue
        recorded_leg(ledger, spec.name, reservoir, name) && continue
        push!(
            missing_legs,
            "expected leg $(spec.name)/$name in $reservoir was not recorded",
        )
    end
    return (;
        total,
        magnitude,
        leg_count,
        status_counts,
        blocked_by,
        missing_legs,
    )
end

# Whether one declared leg of an event was recorded in the open transaction.
function recorded_leg(
    ledger::BudgetLedger,
    event::Symbol,
    reservoir::Symbol,
    name::Symbol,
)
    for leg in ledger.legs
        leg.event === event || continue
        leg.leg === name || continue
        reservoir_name(leg.reservoir) === reservoir && return true
    end
    return false
end

"""
    transfer_expectation(spec, control_volume) -> Symbol

Return what this event's signed sum means in this view, from the declared
topology.

  - `:exterior_crossing` when the far side is not modeled. There is one modeled
    leg and nothing for it to cancel against, so no cancellation is tested and
    the total is a signed boundary source or sink.
  - `:cancellation` when every modeled reservoir the event names is inside the
    view. The legs are expected to sum to zero and the sum is tested.
  - `:boundary_crossing` when some modeled reservoirs are outside the view. The
    same coupled event is internal to a larger view and crosses out of this one.

The topology comes from the specification, so the meaning of the total does not
depend on which legs a run happened to record.
"""
function transfer_expectation(spec::TransferEventSpec, cv::ControlVolume)
    tests_cancellation(spec.topology) || return :exterior_crossing
    for reservoir in event_reservoir_names(spec)
        is_inside(cv, reservoir) || return :boundary_crossing
    end
    return :cancellation
end

# ============================================================================
# Reconciliation results
# ============================================================================

"""
    ParentReconciliation

What one quantity did over one accepted step in one control volume, and whether
the recorded accepted updates account for it.

`residual` is `endpoint_change - recorded` and nothing else. `recorded` is the
sum of the channel envelopes and the final maps, never of their decompositions.

`missing_expectations` names the declared channels and final maps that recorded
nothing. Each of them blocks, because the identity has a term the run never
supplied, and a report that dropped them would close over the gap.

`endpoint_change_from_initial` is read directly from the first accepted
endpoint. `cumulative_endpoint_change` telescopes the per-step differences
instead, and `telescoping_discrepancy` is what separates them. The telescoped
sum reproduces the same measurements plus the rounding of every intermediate
addition, so it can never contradict them; the direct reading shares no
arithmetic with it and can.

`status` is one of `:pass`, `:fail`, `:blocked` or `:not_applicable`. A blocked
reconciliation still reports its numbers, because they are informative, but no
closure claim may be made from it.
"""
Base.@kwdef struct ParentReconciliation{FT}
    quantity::Symbol
    control_volume::Symbol
    step::Int
    status::Symbol
    applicable::Bool
    endpoint_change::FT
    envelopes::FT
    final_maps::FT
    recorded::FT
    residual::FT
    tolerance::Union{Nothing, FT}
    cumulative_endpoint_change::FT
    endpoint_change_from_initial::FT
    telescoping_discrepancy::FT
    cumulative_residual::FT
    cumulative_abs_residual::FT
    max_abs_residual::FT
    missing_expectations::Vector{String}
    blocked_by::Vector{String}
end

"""
    AttributionReconciliation

Whether one channel's classified events explain the whole of its accepted
envelope, for one quantity in one control volume.

`residual` is `envelope - attributed`. A channel can reconcile perfectly in the
primary identity while its attribution is entirely unexplained, which is why
this is a separate result and not a field of `ParentReconciliation`.

A final accepted-state map never produces one of these. It is a term of the
primary identity and not an attribution channel, so it has no envelope to
explain and demands no decomposition.
"""
Base.@kwdef struct AttributionReconciliation{FT}
    quantity::Symbol
    control_volume::Symbol
    channel::Symbol
    step::Int
    status::Symbol
    applicable::Bool
    envelope::FT
    attributed::FT
    residual::FT
    tolerance::Union{Nothing, FT}
    blocked_by::Vector{String}
end

"""
    TransferReconciliation

Whether the independently measured legs of one declared event agree, for one
quantity in one control volume.

`topology` is what the schema declared: `:internal`, `:coupled` or `:exterior`.
`expectation` is what that means in this view.

  - `:cancellation` is the only case judged against a tolerance.
  - `:boundary_crossing` is a coupled event seen from a view it leaves. Its
    total is the boundary flux, which is not expected to vanish.
  - `:exterior_crossing` is an event whose far side the model does not carry.
    Its `counterparty` names what it crosses to, and that name never becomes a
    numerical leg: a synthesized counterparty guarantees cancellation and
    therefore measures nothing. The total is the signed source or sink.

Neither crossing gets a cancellation verdict. Their `status` is `:reported`
when the quantity is applicable and nothing blocks: a signed flux with no
verdict attached. That is a different fact from `:not_applicable`, which is
reserved for a quantity nothing in the view owns, and the two do not share a
label, so `status` can be read on its own without a boundary flux looking like a
budget nobody owned.

A nonzero cancellation is a finding, not permission to synthesize a
counter-entry. It names lagged coupling, clipping, inconsistent quadrature, or a
reservoir the model does not represent.
"""
Base.@kwdef struct TransferReconciliation{FT}
    quantity::Symbol
    event::Symbol
    control_volume::Symbol
    step::Int
    status::Symbol
    topology::Symbol
    expectation::Symbol
    counterparty::Union{Nothing, Symbol}
    applicable::Bool
    total::FT
    leg_count::Int
    tolerance::Union{Nothing, FT}
    status_counts::Dict{Symbol, Int}
    missing_legs::Vector{String}
    blocked_by::Vector{String}
end

"""
    is_blocked(reconciliation) -> Bool

Return whether the reconciliation's status is `:blocked`.
"""
is_blocked(r::ParentReconciliation) = r.status === :blocked
is_blocked(r::AttributionReconciliation) = r.status === :blocked
is_blocked(r::TransferReconciliation) = r.status === :blocked

"""
    BudgetCommit

The three families of reconciliation produced by one committed step, kept apart.

They answer different questions and are never added together: `parent` asks
whether accounting reproduced the accepted state transition, `attribution`
whether the classified paths explained each channel, and `transfer` whether
independently measured legs of one exchange agreed. None of the three
establishes physical completeness or provenance.
"""
Base.@kwdef struct BudgetCommit{FT}
    step::Int
    parent::Vector{ParentReconciliation{FT}}
    attribution::Vector{AttributionReconciliation{FT}}
    transfer::Vector{TransferReconciliation{FT}}
end

# ============================================================================
# Reconciliation
# ============================================================================

"""
    quantity_tolerance(tolerances, quantity)

Return the `BudgetTolerance` declared for `quantity`, or `nothing`.

`tolerances` is a mapping from quantity to tolerance, or `nothing` when none has
been declared. There is no default: `kappa` has to be calibrated, and a
reconciliation asked for a verdict without one reports `blocked`.
"""
quantity_tolerance(::Nothing, ::Symbol) = nothing
quantity_tolerance(tolerances, quantity::Symbol) =
    get(tolerances, quantity, nothing)

# Evaluate a tolerance if one was declared, and otherwise record why no verdict
# is available. Returning the blocker rather than a permissive default is what
# keeps an uncalibrated run from reporting `pass`.
function resolve_tolerance(tolerance, blocked_by, opening, closing, magnitude)
    isnothing(tolerance) || return (
        tolerance_value(tolerance, opening, closing, magnitude),
        blocked_by,
    )
    return (nothing, vcat(blocked_by, [UNCALIBRATED_TOLERANCE_BLOCKER]))
end

"""
    reconcile_parent(ledger, closing, quantity, control_volume; tolerances)

Compute one `ParentReconciliation` from the open transaction and the closing
endpoints. Pure; it does not mutate the ledger.
"""
function reconcile_parent(
    ledger::BudgetLedger{FT},
    closing::BudgetEndpoints{FT},
    quantity::Symbol,
    cv::ControlVolume;
    tolerances = nothing,
) where {FT}
    opening = ledger.opening
    isnothing(opening) && error("No open budget transaction to reconcile.")

    before = endpoint_total(opening, quantity, cv)
    after = endpoint_total(closing, quantity, cv)
    endpoint_change = after.total - before.total

    projected = project_parent(ledger, quantity, cv)
    residual = endpoint_change - projected.recorded

    key = (quantity, cv.name)
    cumulative_change =
        get(ledger.cumulative_change, key, zero(FT)) + endpoint_change

    initial = ledger.initial
    from_initial = if isnothing(initial)
        endpoint_change
    else
        after.total - endpoint_total(initial, quantity, cv).total
    end

    applicable = before.applicable || after.applicable
    missing_expectations = missing_parent_terms(ledger, cv)
    schema = ledger.schema
    blocked_by = vcat(
        before.blocked_by,
        after.blocked_by,
        projected.blocked_by,
        missing_expectations,
        open_dispositions(schema.channels, quantity, cv),
        open_dispositions(schema.final_maps, quantity, cv),
    )
    cumulative_residual = get(ledger.cumulative_residual, key, zero(FT)) + residual
    previous_abs = get(ledger.cumulative_abs_residual, key, zero(FT))
    previous_max = get(ledger.max_abs_residual, key, zero(FT))
    tolerance, blocked_by = resolve_tolerance(
        quantity_tolerance(tolerances, quantity),
        blocked_by,
        before.total,
        after.total,
        projected.magnitude,
    )

    return ParentReconciliation{FT}(;
        quantity,
        control_volume = cv.name,
        step = ledger.step,
        status = claim_status(applicable, blocked_by, residual, tolerance),
        applicable,
        endpoint_change,
        envelopes = projected.envelopes,
        final_maps = projected.final_maps,
        recorded = projected.recorded,
        residual,
        tolerance,
        cumulative_endpoint_change = cumulative_change,
        endpoint_change_from_initial = from_initial,
        telescoping_discrepancy = cumulative_change - from_initial,
        cumulative_residual,
        cumulative_abs_residual = previous_abs + abs(residual),
        max_abs_residual = max(previous_max, abs(residual)),
        missing_expectations,
        blocked_by,
    )
end

"""
    reconcile_attribution(ledger, quantity, control_volume, spec; tolerances)

Compute one `AttributionReconciliation` for a declared channel. Pure.

Three things block. A required envelope that was not recorded. A declared
decomposition row that was not recorded. A disposition the registry has left
open. All three are read from the specification rather than from what arrived,
so a channel that reported nothing at all is a blocked row naming it, and a
channel that reported some of its processes is blocked by the rest.
"""
function reconcile_attribution(
    ledger::BudgetLedger{FT},
    quantity::Symbol,
    cv::ControlVolume,
    spec::ChannelSpec;
    tolerances = nothing,
) where {FT}
    projected = project_attribution(ledger, quantity, cv, spec.name)
    applicable =
        declared_applicable(ledger.schema, spec.reservoirs, quantity, cv)
    residual = projected.envelope - projected.attributed
    blocked_by = vcat(
        projected.blocked_by,
        missing_channel_envelopes(ledger, spec, cv),
        missing_processes(ledger, spec, cv),
        open_dispositions((spec,), quantity, cv),
    )
    tolerance, blocked_by = resolve_tolerance(
        quantity_tolerance(tolerances, quantity),
        blocked_by,
        projected.envelope,
        projected.attributed,
        projected.magnitude,
    )
    return AttributionReconciliation{FT}(;
        quantity,
        control_volume = cv.name,
        channel = spec.name,
        step = ledger.step,
        status = claim_status(applicable, blocked_by, residual, tolerance),
        applicable,
        envelope = projected.envelope,
        attributed = projected.attributed,
        residual,
        tolerance,
        blocked_by,
    )
end

"""
    reconcile_transfer(ledger, spec, quantity, control_volume; tolerances)

Compute one `TransferReconciliation` for a declared event. Pure.

Only a cancellation is judged against a tolerance. A crossing, whether out of
this view or out of the model, reports its signed total as a boundary source or
sink and takes no cancellation verdict, because there is nothing on the other
side of it to cancel against. Testing it against zero would report a flux as a
broken budget, and fabricating a counter-leg to make it vanish would measure
nothing at all.
"""
function reconcile_transfer(
    ledger::BudgetLedger{FT},
    spec::TransferEventSpec,
    quantity::Symbol,
    cv::ControlVolume;
    tolerances = nothing,
) where {FT}
    projected = project_transfer(ledger, spec, quantity, cv)
    expectation = transfer_expectation(spec, cv)
    topology = topology_name(spec.topology)
    applicable = declared_applicable(
        ledger.schema,
        event_reservoir_names(spec),
        quantity,
        cv,
    )
    blocked_by = vcat(
        projected.blocked_by,
        projected.missing_legs,
        open_dispositions((spec,), quantity, cv),
    )

    if expectation !== :cancellation
        # A crossing takes no verdict. `:reported` is a signed flux with nothing
        # to judge it against, which is a different fact from a quantity nothing
        # in the view owns, and the two must not share a label.
        status = if !applicable
            :not_applicable
        elseif !isempty(blocked_by)
            :blocked
        else
            :reported
        end
        return TransferReconciliation{FT}(;
            quantity,
            event = spec.name,
            control_volume = cv.name,
            step = ledger.step,
            status,
            topology,
            expectation,
            counterparty = spec.counterparty,
            applicable,
            total = projected.total,
            leg_count = projected.leg_count,
            tolerance = nothing,
            status_counts = projected.status_counts,
            missing_legs = projected.missing_legs,
            blocked_by,
        )
    end

    tolerance, blocked_by = resolve_tolerance(
        quantity_tolerance(tolerances, quantity),
        blocked_by,
        zero(FT),
        zero(FT),
        projected.magnitude,
    )
    return TransferReconciliation{FT}(;
        quantity,
        event = spec.name,
        control_volume = cv.name,
        step = ledger.step,
        status = claim_status(
            applicable,
            blocked_by,
            projected.total,
            tolerance,
        ),
        topology,
        expectation,
        counterparty = spec.counterparty,
        applicable,
        total = projected.total,
        leg_count = projected.leg_count,
        tolerance,
        status_counts = projected.status_counts,
        missing_legs = projected.missing_legs,
        blocked_by,
    )
end

# Whether a declared event has anything to say in this view. An event whose
# every declared leg lies outside the control volume is not part of that view at
# all, which is different from an event that was expected here and recorded
# nothing.
function event_in_view(spec::TransferEventSpec, cv::ControlVolume)
    for reservoir in event_reservoir_names(spec)
        is_inside(cv, reservoir) && return true
    end
    return false
end

"""
    commit_transaction!(ledger, closing; tolerances)

Close the transaction and return a `BudgetCommit` holding the parent,
attribution and transfer reconciliations for every quantity, every declared
control volume, every declared channel, and every declared transfer event.

The closing endpoints must be for the step the transaction opened, which is the
check that catches a missed or a doubled step, and they must describe the
configuration the schema declares.

Every row comes from a declaration rather than from a record, so a channel or
event that reported nothing produces a blocked row naming it instead of
vanishing. Cumulative totals are updated here and only here, so an aborted
transaction contributes nothing to them.

The commit is **atomic**. Every check and every reconciliation is computed into
a temporary first, and the ledger is not touched until all of them have
succeeded. Updating the cumulative totals inside the loop would leave an error
raised part way through with some quantities already advanced in a transaction
that was still open. The ledger would have half-counted a step it never
committed, with no way to tell from its own state.
"""
function commit_transaction!(
    ledger::BudgetLedger{FT},
    closing::BudgetEndpoints{FT};
    tolerances = nothing,
) where {FT}
    ledger.is_open || error("No open budget transaction to commit.")
    closing.step == ledger.step || error(
        "Closing endpoints are for step $(closing.step), but the open " *
        "transaction is step $(ledger.step).",
    )

    check_schema_endpoints(ledger.schema, closing, "closing")
    check_endpoint_layout(ledger.opening, closing)

    # Compute everything before changing anything. The reconcile functions read
    # the cumulative dictionaries but never write them, so this section is free
    # of side effects and may fail part way through without consequence.
    schema = ledger.schema
    parent = ParentReconciliation{FT}[]
    attribution = AttributionReconciliation{FT}[]
    transfer = TransferReconciliation{FT}[]
    for cv in schema.control_volumes, quantity in BUDGET_QUANTITIES
        push!(
            parent,
            reconcile_parent(ledger, closing, quantity, cv; tolerances),
        )
        for spec in schema.channels
            any(r -> is_inside(cv, r), spec.reservoirs) || continue
            push!(
                attribution,
                reconcile_attribution(ledger, quantity, cv, spec; tolerances),
            )
        end
        for spec in schema.transfer_events
            event_in_view(spec, cv) || continue
            push!(
                transfer,
                reconcile_transfer(ledger, spec, quantity, cv; tolerances),
            )
        end
    end

    # Past here nothing can fail, so the ledger may be advanced.
    for r in parent
        key = (r.quantity, r.control_volume)
        ledger.cumulative_change[key] = r.cumulative_endpoint_change
        ledger.cumulative_residual[key] = r.cumulative_residual
        ledger.cumulative_abs_residual[key] = r.cumulative_abs_residual
        ledger.max_abs_residual[key] = r.max_abs_residual
    end

    commit = BudgetCommit{FT}(; step = ledger.step, parent, attribution, transfer)
    ledger.is_open = false
    ledger.opening = nothing
    ledger.last_closing = closing
    ledger.committed_steps += 1
    clear_open_transaction!(ledger)
    return commit
end
