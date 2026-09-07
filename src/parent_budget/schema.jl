#####
##### Parent-budget ledger: the declaration layer
#####
##### What a configuration is expected to produce, declared before anything is
##### collected. The journal records what happened; this file says what should
##### have. Reconciliation compares the two in both directions, so an expected
##### channel that recorded nothing blocks rather than vanishing, and a record
##### this file does not declare is refused rather than becoming a new row.
#####
##### Deriving the expected set from the records is the failure the separation
##### exists to prevent. A process that never reported would remove itself from
##### its own audit, and the report would close over whatever happened to arrive.
#####
##### Nothing here reads a leg, a packet, or a transaction. A schema is built
##### once from the model configuration and never changes afterwards, which is
##### also what makes the packet layout computable before the first record.

# ============================================================================
# Reservoirs
# ============================================================================

"""
    ATMOSPHERE_ENDPOINT_GROUP

The packet group and reservoir label of the atmospheric endpoint slots.

Groups are plain symbols so the reduction mechanics need no reservoir types, and
`reservoir_name` returns the same symbols, which is what ties a group back to its
reservoir.
"""
const ATMOSPHERE_ENDPOINT_GROUP = :atmosphere

"""
    SLAB_SURFACE_ENDPOINT_GROUP

The packet group and reservoir label of the slab surface endpoint slots. See
`ATMOSPHERE_ENDPOINT_GROUP`.
"""
const SLAB_SURFACE_ENDPOINT_GROUP = :slab_surface

"""
    EXTERIOR_LABEL

The label of an unmodeled exterior counterparty.

Never a reservoir and never a packet group. The exterior has no state the model
owns, so it has no endpoint to measure and no numerical leg to record.
"""
const EXTERIOR_LABEL = :exterior

"""
    BudgetReservoir

A place the model owns state in, which can gain or lose a parent quantity.

The graph is small and saying so is part of the contract. The atmosphere always
exists. The slab surface exists only for a
`SurfaceConditions.SlabOceanTemperature`. Everything else is exterior: its state
is not owned by the model, so a flux into it is a boundary crossing and never an
internal transfer.
"""
abstract type BudgetReservoir end

"""
    AtmosphereReservoir()

The atmospheric column state, `Y.c` and `Y.f`. See `BudgetReservoir`.
"""
struct AtmosphereReservoir <: BudgetReservoir end

"""
    SlabSurfaceReservoir()

The slab surface state, `Y.sfc`. Owns energy, and in a moist configuration water
and the mass that goes with it. See `BudgetReservoir`.
"""
struct SlabSurfaceReservoir <: BudgetReservoir end

"""
    reservoir_name(reservoir) -> Symbol

A short label for `reservoir`, and the packet group holding its endpoint slots.
"""
reservoir_name(::AtmosphereReservoir) = ATMOSPHERE_ENDPOINT_GROUP
reservoir_name(::SlabSurfaceReservoir) = SLAB_SURFACE_ENDPOINT_GROUP

"""
    ControlVolume(name, reservoirs)

A named set of reservoirs to project the journal onto.

The two views a supported configuration can declare are `ATMOSPHERE_ONLY` and
`ATMOSPHERE_AND_SURFACE`. A leg counts toward a view when its reservoir is
inside it, so one recorded exchange is a boundary crossing in the first view and
an internal transfer in the second without being recorded twice.
"""
struct ControlVolume
    name::Symbol
    reservoirs::Tuple{Vararg{BudgetReservoir}}
end

"""
    ATMOSPHERE_ONLY

The atmosphere alone. Every surface exchange leaves this view.
"""
const ATMOSPHERE_ONLY = ControlVolume(:atmosphere_only, (AtmosphereReservoir(),))

"""
    ATMOSPHERE_AND_SURFACE

The atmosphere together with the slab surface. A surface exchange between the
two is internal here, and its declared legs are expected to cancel. The
expectation is tested, never imposed.
"""
const ATMOSPHERE_AND_SURFACE = ControlVolume(
    :atmosphere_and_surface,
    (AtmosphereReservoir(), SlabSurfaceReservoir()),
)

"""
    is_inside(control_volume, reservoir) -> Bool

Whether `reservoir` is one of the reservoirs `control_volume` contains.
"""
is_inside(cv::ControlVolume, reservoir::BudgetReservoir) =
    any(r -> r === reservoir, cv.reservoirs)

"""
    is_inside(control_volume, name::Symbol) -> Bool

Whether the reservoir labelled `name` is inside `control_volume`.
"""
is_inside(cv::ControlVolume, name::Symbol) =
    any(r -> reservoir_name(r) === name, cv.reservoirs)

# ============================================================================
# Channels, final maps, update paths and collection levels
# ============================================================================

"""
    ATTRIBUTION_CHANNELS

The accepted integrator channels. Each is a term of the primary identity and the
target of an attribution identity of its own.

`remaining_tendency!` writes two explicit channels, not one: `Yₜ` and the
*limited* `Yₜ_lim`, which `ClimaTimeSteppers` integrates through the limiter.
Horizontal tracer advection and tracer hyperdiffusion live only in the limited
one, so an adapter reading `Yₜ` alone loses them silently.
"""
const ATTRIBUTION_CHANNELS = (:explicit_main, :explicit_limited, :implicit, :post_implicit)

"""
    FINAL_STATE_MAPS

The labels a map applied to the accepted state may carry.

A final map is a term of the primary identity and **not** an attribution
channel. It contributes its raw before/after difference on the accepted state,
and it has no envelope, no decomposition, and no attribution residual. Recording
one therefore creates no requirement for a channel envelope; an operation
acquires that requirement only by being declared an accepted integrator channel.
"""
const FINAL_STATE_MAPS = (:lim!, :dss!, :constrain_state!, :initialization)

"""
    BUDGET_CHANNEL_LABELS

Every label a leg's `channel` field may hold: the attribution channels and the
final-map labels together. Membership alone places nothing; the leg's
`CollectionLevel` decides which identity it takes part in.
"""
const BUDGET_CHANNEL_LABELS = (ATTRIBUTION_CHANNELS..., FINAL_STATE_MAPS...)

"""
    UpdatePath

What kind of update a leg records. This classifies the *nature* of a
contribution; `CollectionLevel` says which identity it belongs to.
"""
abstract type UpdatePath end

"""
    EquationTerm()

Accepted integration of an explicit or implicit equation term.
"""
struct EquationTerm <: UpdatePath end

"""
    DiscreteMap()

An accepted sequential, split, coupling, callback, or post-solve map.
"""
struct DiscreteMap <: UpdatePath end

"""
    NumericalCorrection()

A limiter, clipping, projection, or consistency repair. These are numerical
corrections and are never reported as physical tendencies.
"""
struct NumericalCorrection <: UpdatePath end

"""
    AlgebraicSolveDefect()

An independently derived projection of an incomplete algebraic solve.

Not a small term in ClimaAtmos. The default `NewtonsMethod(max_iters = 1)`
against an approximate Jacobian does not converge the implicit stage, so the
defect is leading order and an implicit channel's attribution cannot close
without it.
"""
struct AlgebraicSolveDefect <: UpdatePath end

"""
    CollectionLevel

Which of the three nested identities a leg takes part in.

  - `ChannelEnvelope` and `FinalMap` are the terms of the **primary** identity.
  - `ProcessDecomposition` and `ReservoirTransfer` explain an envelope rather
    than adding to it, so they are terms of the **attribution** identity.
  - `ReservoirTransfer` additionally takes part in the **transfer** identity.

This is how the contract's rule that an aggregate is never summed alongside its
own decomposition is enforced. The two are deliberately recorded together, since
comparing them is the whole point of attribution, and it is the *sums* that are
kept apart: `enters_parent_identity` admits only envelopes and final maps, and no
other total mixes the levels.
"""
abstract type CollectionLevel end

"""
    ChannelEnvelope()

The complete update one channel applied to one reservoir, taken from the applied
increment. A term of the primary identity, and the reference its decomposition
is reconciled against. See `CollectionLevel`.
"""
struct ChannelEnvelope <: CollectionLevel end

"""
    ProcessDecomposition()

One classified process's share of a channel. Explains an envelope; never added
to one. See `CollectionLevel`.
"""
struct ProcessDecomposition <: CollectionLevel end

"""
    FinalMap()

A map applied to the accepted state itself, contributing its raw before/after
difference. A term of the primary identity and not an attribution channel. See
`CollectionLevel`.
"""
struct FinalMap <: CollectionLevel end

"""
    ReservoirTransfer()

One reservoir's leg of an exchange. Explains a channel like a decomposition, and
additionally takes part in the transfer identity. See `CollectionLevel`.
"""
struct ReservoirTransfer <: CollectionLevel end

"""
    level_name(level) -> Symbol

A short label for a `CollectionLevel`.
"""
level_name(::ChannelEnvelope) = :envelope
level_name(::ProcessDecomposition) = :decomposition
level_name(::FinalMap) = :final_map
level_name(::ReservoirTransfer) = :transfer

"""
    enters_parent_identity(level) -> Bool

Whether a leg at this level is one of the primary identity's recorded terms.

True for `ChannelEnvelope` and `FinalMap`. False for the two levels that explain
an envelope instead of adding to it, which is what makes it impossible to sum an
aggregate alongside its own decomposition.
"""
enters_parent_identity(::ChannelEnvelope) = true
enters_parent_identity(::FinalMap) = true
enters_parent_identity(::ProcessDecomposition) = false
enters_parent_identity(::ReservoirTransfer) = false

"""
    explains_envelope(level) -> Bool

Whether a leg at this level is one of the attribution identity's terms. The
complement of `enters_parent_identity`.
"""
explains_envelope(level::CollectionLevel) = !enters_parent_identity(level)

# ============================================================================
# Transfer topology
# ============================================================================

"""
    TransferTopology

How the two sides of a transfer event relate. Declared from the configuration,
never inferred from whichever legs were recorded, because the topology decides
which test applies and applying the wrong test is the mistake this separation
exists to prevent.
"""
abstract type TransferTopology end

"""
    InternalTransfer()

Both sides are modeled reservoirs within one control volume. Every declared leg
is required and their signed sum is tested for cancellation. See
`TransferTopology`.
"""
struct InternalTransfer <: TransferTopology end

"""
    CoupledTransfer()

Both sides are modeled reservoirs, in different control volumes. Cancellation is
expected only in a view holding all of them; in any other view the same event
crosses the boundary. See `TransferTopology`.
"""
struct CoupledTransfer <: TransferTopology end

"""
    ExteriorCrossing()

One modeled side, and a counterparty the model does not carry as state.

The modeled leg is measured and required, the counterparty is named as metadata,
and **no numerical counter-leg is created**: a synthesized counterparty
guarantees cancellation and therefore measures nothing. No cancellation test
applies, and the signed crossing is reported as a boundary source or sink. See
`TransferTopology`.
"""
struct ExteriorCrossing <: TransferTopology end

"""
    topology_name(topology) -> Symbol

A short label for a `TransferTopology`.
"""
topology_name(::InternalTransfer) = :internal
topology_name(::CoupledTransfer) = :coupled
topology_name(::ExteriorCrossing) = :exterior

"""
    tests_cancellation(topology) -> Bool

Whether a signed sum of this event's legs is expected to cancel in a view holding
every modeled reservoir it names.

False for `ExteriorCrossing`, whose single modeled leg has nothing to cancel
against. Testing it against zero would report a boundary flux as a broken
budget.
"""
tests_cancellation(::InternalTransfer) = true
tests_cancellation(::CoupledTransfer) = true
tests_cancellation(::ExteriorCrossing) = false

# ============================================================================
# Specifications
# ============================================================================

"""
    quantity_position(quantity) -> Int

The index of `quantity` in `BUDGET_QUANTITIES`. Errors for anything else, so a
misspelled quantity fails at the call rather than silently missing a slot.
"""
function quantity_position(quantity::Symbol)
    for (i, q) in enumerate(BUDGET_QUANTITIES)
        q === quantity && return i
    end
    error(
        "Unknown budget quantity $quantity; expected one of " *
        "$(BUDGET_QUANTITIES).",
    )
end

"""
    EXPECTED_DISPOSITIONS

What the coverage registry can say a component of a declared row will be.

  - `:measured` — not provably zero, so the ledger has to measure it.
  - `:invariant_zero` — provably zero, with the proof named in the record.
  - `:not_applicable` — the row does not write this quantity in this
    configuration.
  - `:open` — not yet established from the code, so nothing is demanded of the
    record and the claim it feeds stays blocked.

This is the proof obligation half of what a schema declares. Applicability says
whether a reservoir owns a quantity at all; a disposition says what a particular
channel, map or event is expected to do to it. A row that declares
`:invariant_zero` and then measures something is a disagreement between the
registry and the code, not a residual, so it is refused where it happens.
"""
const EXPECTED_DISPOSITIONS =
    (:measured, :invariant_zero, :not_applicable, :open)

"""
    OPEN_DISPOSITIONS

Every quantity `:open`: the default for a declaration whose proof obligations
have not been established yet. It demands nothing of a record and blocks every
claim the declaration feeds, which is what the coverage registry's `open` rows
mean: a sum over a row nothing has established proves nothing, however small it
comes out. A declaration passes only once its dispositions are declared.
"""
const OPEN_DISPOSITIONS = ntuple(_ -> :open, length(BUDGET_QUANTITIES))

# One flag per entry of BUDGET_QUANTITIES, each a member of the vocabulary. A
# misspelled disposition would otherwise permit everything, which is the one
# outcome a proof obligation must not have.
function check_dispositions(what::AbstractString, dispositions)
    for disposition in dispositions
        disposition in EXPECTED_DISPOSITIONS || error(
            "$what declares disposition $disposition, which is not one of " *
            "$(EXPECTED_DISPOSITIONS).",
        )
    end
    return nothing
end

"""
    ReservoirSpec(reservoir, applicable)

Which parent quantities one reservoir owns in this configuration.

`applicable` is one flag per entry of `BUDGET_QUANTITIES`, in that order, and it
comes from the configuration rather than from the state. `Y.sfc.water` exists in
a dry run holding a permanent zero, so field presence cannot tell an inapplicable
quantity from a measured one.
"""
struct ReservoirSpec
    reservoir::BudgetReservoir
    applicable::NTuple{length(BUDGET_QUANTITIES), Bool}
end

"""
    ChannelSpec(name, reservoirs; dispositions = OPEN_DISPOSITIONS,
                requires_envelope = true, processes = ())

An accepted integrator channel the configuration is expected to produce.

`reservoirs` names the reservoirs the channel writes, which is what says in
which control volumes its envelope is expected at all. `dispositions` says what
each quantity of its legs is expected to be; see `EXPECTED_DISPOSITIONS`.

`requires_envelope` says the channel must record the complete update it applied.
A missing required envelope blocks, because the primary identity has a term with
nothing in it.

`processes` is the roster of `(process, reservoir)` rows the channel's
decomposition must record, each in a reservoir the channel names. A channel can
reconcile in the primary identity while its attribution is entirely unexplained,
which is why the roster is separate from the envelope. It is a roster and not a
flag because a flag is satisfied by whichever rows happen to arrive: the rows
present can cancel exactly and the residual says nothing about the one that is
missing. Every declared row that was not recorded blocks the channel's
attribution and is named. A decomposition leg for a process the roster does not
declare is refused, so the roster is the complete list of what the channel
explains itself with. `requires_decomposition` is true when the roster is not
empty.
"""
struct ChannelSpec
    name::Symbol
    reservoirs::Tuple{Vararg{Symbol}}
    dispositions::NTuple{length(BUDGET_QUANTITIES), Symbol}
    requires_envelope::Bool
    requires_decomposition::Bool
    processes::Tuple{Vararg{Tuple{Symbol, Symbol}}}
    function ChannelSpec(
        name::Symbol,
        reservoirs::Tuple{Vararg{Symbol}};
        dispositions = OPEN_DISPOSITIONS,
        requires_envelope::Bool = true,
        processes::Tuple{Vararg{Tuple{Symbol, Symbol}}} = (),
    )
        name in ATTRIBUTION_CHANNELS || error(
            "Channel $name is not one of $(ATTRIBUTION_CHANNELS). A final " *
            "accepted-state map is declared with FinalMapSpec instead: it is a " *
            "term of the primary identity and not an attribution channel.",
        )
        isempty(reservoirs) &&
            error("Channel $name must name at least one reservoir.")
        EXTERIOR_LABEL in reservoirs && error(
            "Channel $name names the exterior, which owns no state and has " *
            "no envelope to record.",
        )
        length(unique(reservoirs)) == length(reservoirs) ||
            error("Channel $name names the same reservoir twice.")
        length(unique(processes)) == length(processes) || error(
            "Channel $name declares the same (process, reservoir) row twice. " *
            "A row is recorded once and a repeated declaration would demand " *
            "two records of it.",
        )
        for (process, reservoir) in processes
            reservoir in reservoirs || error(
                "Channel $name expects process $process in $reservoir, which " *
                "the channel does not write. A decomposition row belongs to a " *
                "reservoir its channel names.",
            )
        end
        check_dispositions("Channel $name", dispositions)
        return new(
            name,
            reservoirs,
            dispositions,
            requires_envelope,
            !isempty(processes),
            processes,
        )
    end
end

function ChannelSpec(name::Symbol, reservoir::Symbol; kwargs...)
    return ChannelSpec(name, (reservoir,); kwargs...)
end

"""
    FinalMapSpec(name, reservoirs; dispositions = OPEN_DISPOSITIONS)

A map applied to the accepted state that the configuration is expected to record.

`name` is one of `FINAL_STATE_MAPS` and `reservoirs` names the reservoirs the map
writes. `dispositions` says what each quantity of its legs is expected to be; a
tag-only or category-only map declares `:invariant_zero` for all three, which is
the proof obligation that it writes no parent field. It contributes directly to
the primary identity and produces no attribution result of its own.
"""
struct FinalMapSpec
    name::Symbol
    reservoirs::Tuple{Vararg{Symbol}}
    dispositions::NTuple{length(BUDGET_QUANTITIES), Symbol}
    function FinalMapSpec(
        name::Symbol,
        reservoirs::Tuple{Vararg{Symbol}};
        dispositions = OPEN_DISPOSITIONS,
    )
        name in FINAL_STATE_MAPS || error(
            "Final map $name is not one of $(FINAL_STATE_MAPS).",
        )
        isempty(reservoirs) &&
            error("Final map $name must name at least one reservoir.")
        EXTERIOR_LABEL in reservoirs && error(
            "Final map $name names the exterior, which owns no state and has " *
            "nothing for a map to write.",
        )
        check_dispositions("Final map $name", dispositions)
        return new(name, reservoirs, dispositions)
    end
end

function FinalMapSpec(name::Symbol, reservoir::Symbol; kwargs...)
    return FinalMapSpec(name, (reservoir,); kwargs...)
end

"""
    TransferEventSpec(name, topology, channel, modeled_legs; counterparty,
                      dispositions)

One exchange the configuration is expected to record, and how its sides relate.

`modeled_legs` is the complete set of `(reservoir, leg)` pairs the event must
record. Every one of them is required: a declared leg that was not recorded
blocks the event and is never read as a zero.

`counterparty` names the unmodeled exterior for an `ExteriorCrossing` and must be
absent otherwise. It is metadata, not a leg. Nothing numerical is ever created
for it.

`dispositions` says what each quantity of the event's legs is expected to be;
see `EXPECTED_DISPOSITIONS`. Radiation at the top of the atmosphere declares
`:invariant_zero` for mass and water and `:measured` for energy, which is the
proof obligation that it moves no air.

The invariants are enforced here so a malformed topology cannot reach a
reconciliation. An exterior crossing has exactly one modeled reservoir, because
a second modeled side would make it an internal or coupled transfer. An internal
or coupled transfer has at least two, because a signed cancellation between one
leg and nothing is a test of that leg against zero.
"""
struct TransferEventSpec
    name::Symbol
    topology::TransferTopology
    channel::Symbol
    modeled_legs::Tuple{Vararg{Tuple{Symbol, Symbol}}}
    counterparty::Union{Nothing, Symbol}
    dispositions::NTuple{length(BUDGET_QUANTITIES), Symbol}
    function TransferEventSpec(
        name::Symbol,
        topology::TransferTopology,
        channel::Symbol,
        modeled_legs::Tuple{Vararg{Tuple{Symbol, Symbol}}};
        counterparty::Union{Nothing, Symbol} = nothing,
        dispositions = OPEN_DISPOSITIONS,
    )
        channel in ATTRIBUTION_CHANNELS || error(
            "Transfer event $name names channel $channel, which is not one of " *
            "$(ATTRIBUTION_CHANNELS).",
        )
        isempty(modeled_legs) && error(
            "Transfer event $name declares no modeled leg. An event with no " *
            "modeled side has nothing the model can measure.",
        )
        length(unique(modeled_legs)) == length(modeled_legs) || error(
            "Transfer event $name declares the same (reservoir, leg) pair " *
            "twice. A leg is recorded once and a repeated declaration would " *
            "demand two records of it.",
        )
        for (reservoir, _) in modeled_legs
            reservoir === EXTERIOR_LABEL && error(
                "Transfer event $name declares a leg in the exterior. The " *
                "exterior owns no state, so there is nothing to measure and no " *
                "counter-leg is ever fabricated for it.",
            )
        end
        reservoirs = unique(first.(modeled_legs))
        if topology isa ExteriorCrossing
            isnothing(counterparty) && error(
                "Transfer event $name is an exterior crossing and must name " *
                "its counterparty. The name is metadata that says what the " *
                "crossing goes to; it never becomes a numerical leg.",
            )
            length(reservoirs) == 1 || error(
                "Transfer event $name is an exterior crossing with " *
                "$(length(reservoirs)) modeled reservoirs. A second modeled " *
                "side makes it an internal or coupled transfer.",
            )
        else
            isnothing(counterparty) || error(
                "Transfer event $name is $(topology_name(topology)) and has no " *
                "exterior counterparty; every side of it is modeled.",
            )
            length(reservoirs) >= 2 || error(
                "Transfer event $name is $(topology_name(topology)) but names " *
                "one modeled reservoir. Cancellation between one leg and " *
                "nothing is a test of that leg against zero.",
            )
        end
        check_dispositions("Transfer event $name", dispositions)
        return new(
            name,
            topology,
            channel,
            modeled_legs,
            counterparty,
            dispositions,
        )
    end
end

"""
    expected_disposition(spec, quantity) -> Symbol

What `spec` declares this quantity's legs will be. One of
`EXPECTED_DISPOSITIONS`.
"""
expected_disposition(spec, quantity::Symbol) =
    spec.dispositions[quantity_position(quantity)]

"""
    event_reservoir_names(spec) -> Vector{Symbol}

The modeled reservoirs `spec` declares, in declaration order and without
repeats.
"""
event_reservoir_names(spec::TransferEventSpec) = unique(first.(spec.modeled_legs))

# ============================================================================
# The schema
# ============================================================================

"""
    BudgetSchema(; reservoirs, control_volumes, channels, final_maps,
                 transfer_events)

Everything a configuration is expected to produce, fixed before collection
begins.

The dictionaries are compiled once, here, so a transaction looks an identity up
rather than discovering it. Nothing in a schema changes after construction, and
no code path adds to one during a step.

The inner constructor rejects a schema that contradicts itself: a repeated name,
a control volume naming an undeclared reservoir, a transfer event whose leg or
channel is undeclared, a final map on an undeclared reservoir. Each of those
would otherwise surface much later as a reconciliation nobody can explain.
"""
struct BudgetSchema
    reservoirs::Vector{ReservoirSpec}
    control_volumes::Vector{ControlVolume}
    channels::Vector{ChannelSpec}
    final_maps::Vector{FinalMapSpec}
    transfer_events::Vector{TransferEventSpec}
    reservoir_index::Dict{Symbol, Int}
    control_volume_index::Dict{Symbol, Int}
    channel_index::Dict{Symbol, Int}
    final_map_index::Dict{Symbol, Int}
    event_index::Dict{Symbol, Int}
end

function BudgetSchema(;
    reservoirs::Vector{ReservoirSpec},
    control_volumes::Vector{ControlVolume},
    channels::Vector{ChannelSpec} = ChannelSpec[],
    final_maps::Vector{FinalMapSpec} = FinalMapSpec[],
    transfer_events::Vector{TransferEventSpec} = TransferEventSpec[],
)
    isempty(reservoirs) &&
        error("A budget schema must declare at least one reservoir.")
    reservoir_index = compile_index(
        [reservoir_name(s.reservoir) for s in reservoirs],
        "reservoir",
    )
    control_volume_index =
        compile_index([cv.name for cv in control_volumes], "control volume")
    channel_index = compile_index([c.name for c in channels], "channel")
    final_map_index = compile_index([m.name for m in final_maps], "final map")
    event_index = compile_index([e.name for e in transfer_events], "transfer event")

    for cv in control_volumes, reservoir in cv.reservoirs
        haskey(reservoir_index, reservoir_name(reservoir)) || error(
            "Control volume $(cv.name) names reservoir " *
            "$(reservoir_name(reservoir)), which the schema does not declare.",
        )
    end
    for final_map in final_maps, reservoir in final_map.reservoirs
        haskey(reservoir_index, reservoir) || error(
            "Final map $(final_map.name) names reservoir $reservoir, which " *
            "the schema does not declare.",
        )
    end
    for channel in channels, reservoir in channel.reservoirs
        haskey(reservoir_index, reservoir) || error(
            "Channel $(channel.name) names reservoir $reservoir, which the " *
            "schema does not declare.",
        )
    end
    for event in transfer_events
        haskey(channel_index, event.channel) || error(
            "Transfer event $(event.name) names channel $(event.channel), " *
            "which the schema does not declare.",
        )
        for (reservoir, _) in event.modeled_legs
            haskey(reservoir_index, reservoir) || error(
                "Transfer event $(event.name) declares a leg in reservoir " *
                "$reservoir, which the schema does not declare.",
            )
        end
    end

    return BudgetSchema(
        reservoirs,
        control_volumes,
        channels,
        final_maps,
        transfer_events,
        reservoir_index,
        control_volume_index,
        channel_index,
        final_map_index,
        event_index,
    )
end

# Build a name-to-position lookup, refusing a repeated name. A duplicate
# identity would make one declaration shadow another and the shadowed one would
# never be checked against anything.
function compile_index(names::Vector{Symbol}, what::AbstractString)
    index = Dict{Symbol, Int}()
    for (i, name) in enumerate(names)
        haskey(index, name) &&
            error("Budget schema declares $what $name twice.")
        index[name] = i
    end
    return index
end

"""
    schema_reservoir_names(schema) -> Vector{Symbol}

The declared reservoirs, in declaration order. This is also the packet group
order, so the layout follows the schema rather than the order values are
produced in.
"""
schema_reservoir_names(schema::BudgetSchema) =
    [reservoir_name(s.reservoir) for s in schema.reservoirs]

"""
    has_reservoir(schema, name) -> Bool

Whether `name` is a declared reservoir.
"""
has_reservoir(schema::BudgetSchema, name::Symbol) =
    haskey(schema.reservoir_index, name)

"""
    reservoir_spec(schema, name) -> ReservoirSpec

The declaration for one reservoir. Errors when the schema does not declare it,
rather than returning a default that would let an undeclared reservoir behave
like an ordinary one.
"""
function reservoir_spec(schema::BudgetSchema, name::Symbol)
    haskey(schema.reservoir_index, name) || error(
        "Budget schema does not declare reservoir $name.",
    )
    return schema.reservoirs[schema.reservoir_index[name]]
end

"""
    quantity_applicable(schema, reservoir, quantity) -> Bool

Whether the configuration says `reservoir` owns `quantity`.
"""
function quantity_applicable(
    schema::BudgetSchema,
    reservoir::Symbol,
    quantity::Symbol,
)
    spec = reservoir_spec(schema, reservoir)
    return spec.applicable[quantity_position(quantity)]
end

"""
    has_channel(schema, name) -> Bool

Whether `name` is a declared attribution channel.
"""
has_channel(schema::BudgetSchema, name::Symbol) =
    haskey(schema.channel_index, name)

"""
    channel_spec(schema, name) -> ChannelSpec

The declaration for one attribution channel.
"""
function channel_spec(schema::BudgetSchema, name::Symbol)
    haskey(schema.channel_index, name) ||
        error("Budget schema does not declare channel $name.")
    return schema.channels[schema.channel_index[name]]
end

"""
    has_final_map(schema, name) -> Bool

Whether `name` is a declared final accepted-state map.
"""
has_final_map(schema::BudgetSchema, name::Symbol) =
    haskey(schema.final_map_index, name)

"""
    final_map_spec(schema, name) -> FinalMapSpec

The declaration for one final accepted-state map.
"""
function final_map_spec(schema::BudgetSchema, name::Symbol)
    haskey(schema.final_map_index, name) ||
        error("Budget schema does not declare final map $name.")
    return schema.final_maps[schema.final_map_index[name]]
end

"""
    has_transfer_event(schema, name) -> Bool

Whether `name` is a declared transfer event.
"""
has_transfer_event(schema::BudgetSchema, name::Symbol) =
    haskey(schema.event_index, name)

"""
    transfer_event_spec(schema, name) -> TransferEventSpec

The declaration for one transfer event, including its topology.
"""
function transfer_event_spec(schema::BudgetSchema, name::Symbol)
    haskey(schema.event_index, name) ||
        error("Budget schema does not declare transfer event $name.")
    return schema.transfer_events[schema.event_index[name]]
end

# ============================================================================
# The configuration adapter
# ============================================================================

"""
    endpoint_schema(surface_temperature, microphysics_model)

The schema a configuration supports at the endpoint-reconciliation stage.

Declares the reservoirs, their per-quantity applicability, and the control
volumes that exist. It deliberately declares **no** channels, final maps, or
transfer events: no runtime path records any yet, and declaring an expectation
nothing can meet would report every configuration as blocked on work that has
not started. Stack steps 3 to 7 add those declarations as they add the
collection that satisfies them.

Applicability comes from the configuration, never from field presence. See
`owns_atmosphere_water`, `has_surface_reservoir` and `owns_surface_water`.
"""
function endpoint_schema(surface_temperature, microphysics_model)
    atmosphere_water = owns_atmosphere_water(microphysics_model)
    reservoirs = ReservoirSpec[
        ReservoirSpec(AtmosphereReservoir(), (true, atmosphere_water, true)),
    ]
    control_volumes = ControlVolume[ATMOSPHERE_ONLY]
    if has_surface_reservoir(surface_temperature)
        # The slab owns mass exactly when it owns water: what it gains left the
        # atmosphere as `ρq_tot`, and `ρ` carries the whole of `ρq_tot`.
        slab_water = owns_surface_water(surface_temperature, microphysics_model)
        push!(
            reservoirs,
            ReservoirSpec(SlabSurfaceReservoir(), (slab_water, slab_water, true)),
        )
        push!(control_volumes, ATMOSPHERE_AND_SURFACE)
    end
    return BudgetSchema(; reservoirs, control_volumes)
end
