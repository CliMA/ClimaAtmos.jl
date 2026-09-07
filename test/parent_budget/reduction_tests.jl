using Test
using ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos.Internals.ParentBudget as PB

# Packet mechanics for the parent-budget ledger.
#
# These tests are state-free. They exercise the rules the packet enforces rather
# than the integrals it later carries: a slot is unset until something writes
# it, it is written once, an unset slot and a slot this configuration does not
# own are different facts, and a local buffer is never read as a global total.
#
# The endpoint packet built from real state, including the single reduction of
# `Y.sfc.water`, is exercised in `endpoint_tests.jl`, which has the state.

const ATMOS = PB.ATMOSPHERE_ENDPOINT_GROUP
const SLAB = PB.SLAB_SURFACE_ENDPOINT_GROUP

singleton_context() =
    ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded())

atmosphere_packet() = PB.BudgetPacket(PB.endpoint_packet_layout((ATMOS,)))

# Every slot resolved, so the packet is ready to reduce.
function resolved_atmosphere_packet(; mass = 1.0, water = 2.0, energy = 3.0)
    packet = atmosphere_packet()
    PB.set_local!(packet, ATMOS, :mass, mass)
    PB.set_local!(packet, ATMOS, :water, water)
    PB.set_local!(packet, ATMOS, :energy, energy)
    return packet
end

@testset "Parent-budget packets" begin
    @testset "The layout is fixed and deterministic" begin
        layout = PB.endpoint_packet_layout((ATMOS, SLAB))
        @test PB.packet_length(layout) == 2 * length(PB.BUDGET_QUANTITIES)
        # Groups in the order given, quantities in BUDGET_QUANTITIES order. The
        # same configuration builds the same layout on every rank and on every
        # step, which is what makes one elementwise reduction well defined and a
        # residual reproducible.
        @test layout.slots == [
            (ATMOS, :mass),
            (ATMOS, :water),
            (ATMOS, :energy),
            (SLAB, :mass),
            (SLAB, :water),
            (SLAB, :energy),
        ]
        @test PB.endpoint_packet_layout((ATMOS, SLAB)).slots == layout.slots
        @test PB.packet_index(layout, SLAB, :water) == 5
        # A slot the layout does not have is an error rather than a default,
        # which would silently read someone else's value.
        @test_throws ErrorException PB.packet_index(layout, :nowhere, :mass)
        @test_throws ErrorException PB.BudgetPacketLayout([
            (ATMOS, :mass),
            (ATMOS, :mass),
        ])
    end

    @testset "The layout follows the schema" begin
        schema = PB.BudgetSchema(;
            reservoirs = [
                PB.ReservoirSpec(PB.AtmosphereReservoir(), (true, false, true)),
                PB.ReservoirSpec(PB.SlabSurfaceReservoir(), (true, true, true)),
            ],
            control_volumes = [PB.ATMOSPHERE_ONLY, PB.ATMOSPHERE_AND_SURFACE],
        )
        # Declaration order fixes the group order, and an inapplicable quantity
        # still gets a slot, so the buffer length depends on which reservoirs
        # exist and not on what each of them owns.
        @test PB.endpoint_packet_layout(schema).slots ==
              PB.endpoint_packet_layout((ATMOS, SLAB)).slots
    end

    @testset "A fresh slot is unset" begin
        packet = atmosphere_packet()
        for quantity in PB.BUDGET_QUANTITIES
            @test PB.slot_state(packet, ATMOS, quantity) isa PB.UnsetSlot
            # Unset is not applicable, and it is not inapplicable either.
            @test !PB.packet_applicable(packet, ATMOS, quantity)
        end
        @test PB.unresolved_slots(packet) == packet.layout.slots
    end

    @testset "A slot is written once" begin
        packet = atmosphere_packet()
        PB.set_local!(packet, ATMOS, :mass, 1.0)
        @test PB.slot_state(packet, ATMOS, :mass) isa PB.MeasuredSlot
        @test PB.packet_applicable(packet, ATMOS, :mass)
        @test PB.packet_local_value(packet, ATMOS, :mass) == 1.0

        PB.set_inapplicable!(packet, ATMOS, :water)
        @test PB.slot_state(packet, ATMOS, :water) isa PB.NotApplicableSlot
        @test !PB.packet_applicable(packet, ATMOS, :water)
        @test PB.packet_local_value(packet, ATMOS, :water) == 0.0

        # A second write is refused where it happens rather than surfacing later
        # as a doubled or overwritten value nothing can attribute.
        @test_throws ErrorException PB.set_local!(packet, ATMOS, :mass, 2.0)
        @test_throws ErrorException PB.set_inapplicable!(packet, ATMOS, :mass)
        @test_throws ErrorException PB.set_local!(packet, ATMOS, :water, 2.0)
        @test_throws ErrorException PB.set_inapplicable!(packet, ATMOS, :water)
        # The refused writes changed nothing.
        @test PB.packet_local_value(packet, ATMOS, :mass) == 1.0
        @test PB.slot_state(packet, ATMOS, :water) isa PB.NotApplicableSlot
    end

    @testset "A slot refuses a non-finite value" begin
        packet = atmosphere_packet()
        for bad in (NaN, Inf, -Inf)
            @test_throws ErrorException PB.set_local!(packet, ATMOS, :mass, bad)
        end
        # The refused write claimed nothing, so the slot is still unset and can
        # still be measured. A NaN that had claimed the slot would have gone
        # into the collective as a measurement.
        @test PB.slot_state(packet, ATMOS, :mass) isa PB.UnsetSlot
        PB.set_local!(packet, ATMOS, :mass, 1.0)
        @test PB.packet_local_value(packet, ATMOS, :mass) == 1.0
    end

    @testset "An unset slot blocks the reduction" begin
        packet = atmosphere_packet()
        PB.set_local!(packet, ATMOS, :mass, 1.0)
        PB.set_inapplicable!(packet, ATMOS, :water)
        # Energy is still unset. Reducing now would send a zero nobody measured
        # into the collective, and the result would be indistinguishable from a
        # measurement that came out zero.
        @test PB.unresolved_slots(packet) == [(ATMOS, :energy)]
        @test_throws ErrorException PB.reduce_packet!(singleton_context(), packet)
        @test !packet.is_reduced

        PB.set_local!(packet, ATMOS, :energy, 3.0)
        @test isempty(PB.unresolved_slots(packet))
        PB.reduce_packet!(singleton_context(), packet)
        @test packet.is_reduced
    end

    @testset "Unresolved data cannot be unpacked" begin
        packet = atmosphere_packet()
        PB.set_local!(packet, ATMOS, :mass, 1.0)
        PB.set_local!(packet, ATMOS, :water, 2.0)
        # Unpacking a slot nothing wrote would put a zero where a measurement
        # should be.
        @test_throws ErrorException PB.budget_endpoints(packet, 0)
        @test_throws ErrorException PB.check_packet_resolved(packet, "unpacked")
    end

    @testset "A local buffer is not a global total" begin
        packet = resolved_atmosphere_packet()
        # Reading a local share as though it were the global answer is the
        # mistake that produces a plausible wrong number on every rank.
        @test_throws ErrorException PB.packet_value(packet, ATMOS, :mass)
        @test PB.packet_local_value(packet, ATMOS, :mass) == 1.0

        PB.reduce_packet!(singleton_context(), packet)
        @test PB.packet_value(packet, ATMOS, :mass) == 1.0
    end

    @testset "A reduced packet is closed" begin
        packet = resolved_atmosphere_packet()
        PB.reduce_packet!(singleton_context(), packet)
        # A second reduction would double every value and the result would look
        # perfectly ordinary.
        @test_throws ErrorException PB.reduce_packet!(
            singleton_context(),
            packet,
        )
        # Writing into a reduced buffer is refused for the same reason.
        @test_throws ErrorException PB.set_local!(packet, ATMOS, :mass, 2.0)
        @test_throws ErrorException PB.set_inapplicable!(packet, ATMOS, :mass)
    end

    @testset "Values are held in the accounting type" begin
        packet = atmosphere_packet()
        # A Float32 input is widened on the way in, so the buffer the collective
        # sees is already in the accounting type rather than being converted
        # after a narrower reduction has lost the information.
        PB.set_local!(packet, ATMOS, :mass, Float32(1.5))
        @test eltype(packet.values) === PB.BUDGET_ACCOUNTING_TYPE
        @test PB.packet_local_value(packet, ATMOS, :mass) isa
              PB.BUDGET_ACCOUNTING_TYPE
    end

    @testset "Groups are reported in layout order" begin
        packet = PB.BudgetPacket(PB.endpoint_packet_layout((ATMOS, SLAB)))
        @test PB.packet_groups(packet) == [ATMOS, SLAB]
    end

    @testset "Endpoints need a reduced packet" begin
        packet = resolved_atmosphere_packet()
        @test_throws ErrorException PB.budget_endpoints(packet, 0)

        PB.reduce_packet!(singleton_context(), packet)
        endpoints = PB.budget_endpoints(packet, 7)
        @test endpoints.step == 7
        @test length(endpoints.reservoirs) == 1
        atmosphere = only(endpoints.reservoirs)
        @test atmosphere.reservoir === PB.AtmosphereReservoir()
        @test atmosphere.mass.amount == 1.0
        @test PB.component_status(atmosphere.mass) isa PB.Measured
        # The evidence records how the number was obtained, which is what lets a
        # report distinguish a packed endpoint from a leg taken from an applied
        # increment.
        @test PB.component_route(atmosphere.mass) === :packed_global_reduction
        @test PB.component_method(atmosphere.mass) ===
              :authoritative_state_integral
    end

    @testset "Explicit non-applicability survives the reduction" begin
        packet = atmosphere_packet()
        PB.set_local!(packet, ATMOS, :mass, 1.0)
        PB.set_inapplicable!(packet, ATMOS, :water)
        PB.set_local!(packet, ATMOS, :energy, 3.0)
        PB.reduce_packet!(singleton_context(), packet)
        # The collective sums the values; it does not touch the dispositions,
        # which are schema-derived and already identical on every rank.
        @test PB.slot_state(packet, ATMOS, :water) isa PB.NotApplicableSlot

        atmosphere = only(PB.budget_endpoints(packet, 0).reservoirs)
        @test PB.component_status(atmosphere.water) isa PB.NotApplicable
        @test atmosphere.water.amount == 0
        @test !PB.is_contributing(atmosphere.water)
        @test !PB.is_blocking(atmosphere.water)
    end

    @testset "The exterior has no endpoint" begin
        # The exterior is a label on a declared counterparty, never a reservoir.
        # There is no group for it, so nothing can unpack one.
        @test_throws ErrorException PB.endpoint_reservoir(PB.EXTERIOR_LABEL)
        @test PB.endpoint_reservoir(ATMOS) === PB.AtmosphereReservoir()
        @test PB.reservoir_name(PB.AtmosphereReservoir()) === ATMOS
        @test PB.reservoir_name(PB.SlabSurfaceReservoir()) === SLAB
    end
end
