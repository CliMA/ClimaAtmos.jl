#####
##### Parent-budget ledger: packet layout and the one collective
#####
##### A global integral is a collective. ARS343 has four stages and the coverage
##### registry lists dozens of paths, so a collective per quantity per leg would
##### cost on the order of a hundred per timestep. The contract's answer is one
##### packed collective per accepted step, and this file is what makes that
##### possible: local values go into a fixed-layout buffer, the buffer is reduced
##### once, and everything downstream reads the reduced buffer.
#####
##### The layout comes from the schema, so it exists before the first value does
##### and is identical on every rank. Nothing here knows what a leg means. It
##### works in plain group and quantity symbols, so the reduction mechanics stay
##### out of the journal and the transaction logic stays out of MPI.

# ============================================================================
# Slot state
# ============================================================================

"""
    PacketSlotState

What has happened to one packet slot.

  - `UnsetSlot`: nothing has written it yet.
  - `MeasuredSlot`: it holds a measured local contribution.
  - `NotApplicableSlot`: the configuration says there is nothing to write.

`UnsetSlot` and `NotApplicableSlot` are not interchangeable, and that is the
reason this type exists. A single "no value" flag makes a forgotten measurement
indistinguishable from a deliberate omission, so the ledger would report a
configuration fact where a defect belongs.
"""
abstract type PacketSlotState end

"""
    UnsetSlot()

Nothing has written this slot. See `PacketSlotState`.
"""
struct UnsetSlot <: PacketSlotState end

"""
    MeasuredSlot()

This slot holds a measured local contribution. See `PacketSlotState`.
"""
struct MeasuredSlot <: PacketSlotState end

"""
    NotApplicableSlot()

The configuration does not own this quantity here, so there is nothing to write.
Never a measured zero. See `PacketSlotState`.
"""
struct NotApplicableSlot <: PacketSlotState end

"""
    slot_state_name(state) -> Symbol

Return a short label for a `PacketSlotState`.
"""
slot_state_name(::UnsetSlot) = :unset
slot_state_name(::MeasuredSlot) = :measured
slot_state_name(::NotApplicableSlot) = :not_applicable

# ============================================================================
# Layout
# ============================================================================

"""
    BudgetPacketLayout(slots)

Which `(group, quantity)` pair each index of a packed buffer holds.

The layout is derived from the schema, not from the order values happen to be
produced in. Every rank builds the same layout from the same configuration,
which is what makes a single elementwise reduction well defined, and the same
run builds the same layout every step, which is what makes a residual
reproducible.

It also bounds memory: a packet is sized from the layout rather than growing
with the number of values recorded.
"""
struct BudgetPacketLayout
    slots::Vector{Tuple{Symbol, Symbol}}
    index::Dict{Tuple{Symbol, Symbol}, Int}
    function BudgetPacketLayout(slots::Vector{Tuple{Symbol, Symbol}})
        index = Dict{Tuple{Symbol, Symbol}, Int}()
        for (i, slot) in enumerate(slots)
            haskey(index, slot) && error(
                "Budget packet layout lists $(slot[1])/$(slot[2]) twice. A " *
                "slot holds one value, so a repeated pair would make two " *
                "different quantities share one index.",
            )
            index[slot] = i
        end
        return new(slots, index)
    end
end

"""
    endpoint_packet_layout(groups)

Build the endpoint layout for `groups`: every quantity of every group, groups in
the order given and quantities in `BUDGET_QUANTITIES` order.

Inapplicable slots are still laid out. A dry configuration's water slot exists
and is marked not applicable, so the buffer length depends on which reservoirs
the schema declares and not on what each of them owns. That keeps the length the
same on every rank without anyone having to agree about moisture separately.
"""
function endpoint_packet_layout(groups)
    slots = Tuple{Symbol, Symbol}[]
    for group in groups, quantity in BUDGET_QUANTITIES
        push!(slots, (group, quantity))
    end
    return BudgetPacketLayout(slots)
end

"""
    endpoint_packet_layout(schema::BudgetSchema)

Build the endpoint layout a schema declares: one group per declared reservoir,
in declaration order.
"""
endpoint_packet_layout(schema::BudgetSchema) =
    endpoint_packet_layout(schema_reservoir_names(schema))

"""
    packet_length(layout) -> Int

Return how many values a packet with this layout holds.
"""
packet_length(layout::BudgetPacketLayout) = length(layout.slots)

"""
    packet_index(layout, group, quantity) -> Int

Return the buffer index of one slot. Errors when the layout has no such slot,
rather than returning a default that would silently read someone else's value.
"""
function packet_index(layout::BudgetPacketLayout, group::Symbol, quantity::Symbol)
    slot = (group, quantity)
    haskey(layout.index, slot) || error(
        "Budget packet layout has no slot for $group/$quantity. The layout " *
        "is built from the schema, so a missing slot means the caller and the " *
        "schema disagree about which reservoirs exist.",
    )
    return layout.index[slot]
end

# ============================================================================
# Packets
# ============================================================================

"""
    BudgetPacket(layout)

A fixed-layout buffer of accounting-precision values, with a
`PacketSlotState` per slot and a record of whether it has been reduced.

`values` is what the collective reduces. `states` is **not** reduced: every
disposition is derived from the schema, so it is already identical on every rank,
and reducing it would spend a second collective to learn nothing. The
one-collective rule is a property of the whole packet, so anything the reduction
does have to carry belongs in `values` alongside the numbers.

The `is_reduced` flag exists so that reading a local value as though it were
global, or reducing the same packet twice, is an error rather than a wrong
number. Both are easy mistakes and neither is visible in the result.
"""
mutable struct BudgetPacket
    layout::BudgetPacketLayout
    values::Vector{BUDGET_ACCOUNTING_TYPE}
    states::Vector{PacketSlotState}
    is_reduced::Bool
end

function BudgetPacket(layout::BudgetPacketLayout)
    n = packet_length(layout)
    return BudgetPacket(
        layout,
        zeros(BUDGET_ACCOUNTING_TYPE, n),
        PacketSlotState[UnsetSlot() for _ in 1:n],
        false,
    )
end

"""
    slot_state(packet, group, quantity) -> PacketSlotState

Return the disposition of one slot.
"""
slot_state(packet::BudgetPacket, group::Symbol, quantity::Symbol) =
    @inbounds packet.states[packet_index(packet.layout, group, quantity)]

# A slot is written once, and both ways of writing it require an unset slot. A
# second write is refused where it happens rather than surfacing later as a
# doubled or overwritten value that nothing can attribute.
function claim_slot!(
    packet::BudgetPacket,
    group::Symbol,
    quantity::Symbol,
    state::PacketSlotState,
)
    packet.is_reduced && error(
        "Budget packet has already been reduced; $group/$quantity cannot be " *
        "written now. Build the packet, reduce it once, then read it.",
    )
    i = packet_index(packet.layout, group, quantity)
    current = @inbounds packet.states[i]
    current isa UnsetSlot || error(
        "Budget packet slot $group/$quantity is already " *
        "$(slot_state_name(current)) and cannot be set to " *
        "$(slot_state_name(state)). A slot is written once.",
    )
    @inbounds packet.states[i] = state
    return i
end

"""
    set_local!(packet, group, quantity, value)

Write one rank's local contribution to a slot and mark it measured.

Requires an unset slot. Refused once the packet has been reduced, because the
reduced buffer is the global answer and writing into it would corrupt a value
nothing would recompute. Refuses a non-finite value before the slot is claimed,
so a refused write leaves the slot unset rather than measured.
"""
function set_local!(
    packet::BudgetPacket,
    group::Symbol,
    quantity::Symbol,
    value,
)
    isfinite(value) || error(
        "Budget packet slot $group/$quantity cannot be set to $value. A NaN " *
        "or Inf sent into the collective would poison every rank's total, " *
        "and no rank could say where it came from.",
    )
    i = claim_slot!(packet, group, quantity, MeasuredSlot())
    @inbounds packet.values[i] = BUDGET_ACCOUNTING_TYPE(value)
    return nothing
end

"""
    set_inapplicable!(packet, group, quantity)

Mark a slot as a quantity this configuration does not own.

This is a positive act with a configuration behind it, not what happens when
nothing writes the slot. It requires an unset slot for the same reason
`set_local!` does. The slot keeps its zero and takes no part in any total.
"""
function set_inapplicable!(
    packet::BudgetPacket,
    group::Symbol,
    quantity::Symbol,
)
    i = claim_slot!(packet, group, quantity, NotApplicableSlot())
    @inbounds packet.values[i] = zero(BUDGET_ACCOUNTING_TYPE)
    return nothing
end

"""
    unresolved_slots(packet) -> Vector{Tuple{Symbol, Symbol}}

Return every slot still `UnsetSlot`, in layout order.
"""
function unresolved_slots(packet::BudgetPacket)
    unresolved = Tuple{Symbol, Symbol}[]
    for (i, slot) in enumerate(packet.layout.slots)
        (@inbounds packet.states[i]) isa UnsetSlot && push!(unresolved, slot)
    end
    return unresolved
end

"""
    check_packet_resolved(packet, what)

Refuse a packet that still has an unset slot, naming the slots.

Every slot has to have an explicit disposition, because an unset one is a
measurement nobody took and a zero would be indistinguishable from one that was
taken and came out zero.

The check is safe immediately before a collective. Every disposition is decided
by the schema, which is identical on every rank, so all ranks reach the same
verdict and none can throw while its peers enter the reduction.
"""
function check_packet_resolved(packet::BudgetPacket, what::AbstractString)
    unresolved = unresolved_slots(packet)
    isempty(unresolved) && return nothing
    named = join(("$(g)/$(q)" for (g, q) in unresolved), ", ")
    verb = length(unresolved) == 1 ? "is" : "are"
    return error(
        "Budget packet cannot be $what: $named $verb still unset. An unset " *
        "slot is a measurement nobody took, which is not the same as a " *
        "quantity this configuration does not own.",
    )
end

"""
    reduce_packet!(context, packet)

Sum the packet across every rank of `context` with one collective, and mark it
reduced.

Refused while any slot is unset, and refused a second time. A second reduction
would double every value and the result would look entirely ordinary.
"""
function reduce_packet!(context, packet::BudgetPacket)
    packet.is_reduced && error(
        "Budget packet has already been reduced; reducing again would double " *
        "every value.",
    )
    check_packet_resolved(packet, "reduced")
    reduce_accounting_sums!(context, packet.values)
    packet.is_reduced = true
    return packet
end

"""
    packet_value(packet, group, quantity)

Return the global value of one slot. Errors if the packet has not been reduced,
or if the slot was never resolved.
"""
function packet_value(packet::BudgetPacket, group::Symbol, quantity::Symbol)
    packet.is_reduced || error(
        "Budget packet has not been reduced; $group/$quantity currently holds " *
        "this rank's local share, not the global total.",
    )
    i = packet_index(packet.layout, group, quantity)
    (@inbounds packet.states[i]) isa UnsetSlot && error(
        "Budget packet slot $group/$quantity is unset and has no value.",
    )
    return @inbounds packet.values[i]
end

"""
    packet_local_value(packet, group, quantity)

Return the local value of one slot, before reduction. For tests and for
assembling a packet; a global total comes from `packet_value`.
"""
packet_local_value(packet::BudgetPacket, group::Symbol, quantity::Symbol) =
    @inbounds packet.values[packet_index(packet.layout, group, quantity)]

"""
    packet_applicable(packet, group, quantity) -> Bool

Return whether this slot holds a measurement. False for a slot the configuration
declared not applicable, and false for one nothing has written; use
`slot_state` when the two have to be told apart.
"""
packet_applicable(packet::BudgetPacket, group::Symbol, quantity::Symbol) =
    slot_state(packet, group, quantity) isa MeasuredSlot

"""
    packet_groups(packet) -> Vector{Symbol}

Return the groups the packet holds, in layout order and without repeats.
"""
function packet_groups(packet::BudgetPacket)
    groups = Symbol[]
    for (group, _) in packet.layout.slots
        group in groups || push!(groups, group)
    end
    return groups
end

# ============================================================================
# Endpoint packets
# ============================================================================

"""
    local_endpoint_packet(Y, schema, surface_temperature)

Pack every declared reservoir's authoritative integrals over the part of the
domain this rank owns into one buffer, with no communication.

The schema decides which slots are measured and which are not applicable, and
`surface_temperature` supplies the slab's areal heat capacity. Applicability is
never read off field presence: a slab carries `Y.sfc.water` even in a dry run,
where it holds a permanent zero, so presence cannot distinguish an inapplicable
quantity from a measured one.

`Y.sfc.water` is summed locally **once**. The slab's water and its mass are two
projections of that one endpoint rather than two measurements, so the second
projection reuses that sum instead of repeating it.
"""
function local_endpoint_packet(Y, schema::BudgetSchema, surface_temperature)
    packet = BudgetPacket(endpoint_packet_layout(schema))
    for spec in schema.reservoirs
        fill_endpoint_slots!(packet, Y, schema, spec, surface_temperature)
    end
    check_packet_resolved(packet, "assembled")
    return packet
end

# One reservoir's slots. Split out so each reservoir's measurements sit next to
# the applicability that selects them, and so an added reservoir is one method
# rather than another branch.
function fill_endpoint_slots!(
    packet::BudgetPacket,
    Y,
    schema::BudgetSchema,
    spec::ReservoirSpec,
    surface_temperature,
)
    group = reservoir_name(spec.reservoir)
    if spec.reservoir isa AtmosphereReservoir
        set_local!(packet, group, :mass, local_atmosphere_mass(Y))
        set_local!(packet, group, :energy, local_atmosphere_energy(Y))
        if quantity_applicable(schema, group, :water)
            set_local!(packet, group, :water, local_atmosphere_water(Y))
        else
            set_inapplicable!(packet, group, :water)
        end
        return nothing
    end

    set_local!(
        packet,
        group,
        :energy,
        local_surface_energy(Y, surface_temperature),
    )
    if quantity_applicable(schema, group, :water)
        # One local reduction of `Y.sfc.water`, used for both projections.
        water = local_surface_water(Y, surface_temperature)
        set_local!(packet, group, :water, water)
        set_local!(packet, group, :mass, water)
    else
        set_inapplicable!(packet, group, :water)
        set_inapplicable!(packet, group, :mass)
    end
    return nothing
end

"""
    reduced_endpoint_packet(Y, schema, surface_temperature)

Build the endpoint packet and reduce it. This is `local_endpoint_packet`
followed by `reduce_packet!`: one collective for every endpoint of every declared
reservoir.
"""
function reduced_endpoint_packet(Y, schema::BudgetSchema, surface_temperature)
    packet = local_endpoint_packet(Y, schema, surface_temperature)
    reduce_packet!(budget_context(Y), packet)
    return packet
end
