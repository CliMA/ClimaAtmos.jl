using Test
using ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaAtmos.Internals.ParentBudget as PB

include("../test_helpers.jl")

# Endpoint integrals against real ClimaCore state.
#
# `journal_tests.jl` exercises the ledger's rules on synthetic scalars. Nothing
# there ever builds a state, so nothing there can check the one thing only a
# state can answer: that applicability comes from the configuration and not from
# which fields happen to exist. A slab carries `Y.sfc.water` even in a dry run,
# where it holds a permanent zero, and a presence check cannot tell that apart
# from a real measurement. These tests call the integrals on a state that exists.
#
# The spaces are small and built directly, with no simulation and no config, so
# these stay unit tests.

const ATMOS = PB.ATMOSPHERE_ENDPOINT_GROUP
const SLAB = PB.SLAB_SURFACE_ENDPOINT_GROUP

# Uniform fields, so each integral can be checked against another integral of
# the same state rather than against a hard-coded number that only restates the
# quadrature.
#
# `ρ_VAL` and the increment below are exact in `Float32` as well as `Float64`:
# 1.25 is 1.01 in binary and the increment is a power of two times it, so a
# `Float32` state holds both endpoint values with no rounding of its own. That
# is what lets the small-increment test below attribute what it measures to the
# accounting arithmetic rather than to the state's representation.
const ρ_VAL = 1.25
const Δρ_REL = 2.0^-13
const Q_VAL = 0.01
const E_VAL = 2.5e5
const Q_LCL, Q_RAI = 0.002, 0.003
const SFC_T, SFC_WATER = 290.0, 3.0

# Field constructors. Written as plain functions broadcast over the local
# geometry, which is how the model itself builds prognostic state. Types are
# scalars under broadcasting, so `FT` rides along as an argument.
dry_center(_, FT, ρ, e) = (; ρ = FT(ρ), ρe_tot = FT(e))

moist_center(_, FT, ρ, e) =
    (; ρ = FT(ρ), ρq_tot = FT(Q_VAL), ρe_tot = FT(e))

function one_moment_center(_, FT, ρ, e)
    return (;
        ρ = FT(ρ),
        ρq_tot = FT(Q_VAL),
        ρq_lcl = FT(Q_LCL),
        ρq_icl = zero(FT),
        ρq_rai = FT(Q_RAI),
        ρq_sno = zero(FT),
        ρe_tot = FT(e),
    )
end

slab_variables(_, FT) = (; T = FT(SFC_T), water = FT(SFC_WATER))

function center_field(space, FT, moist, categories, ρ, e)
    lg = Fields.local_geometry_field(space)
    moist || return dry_center.(lg, FT, ρ, e)
    categories && return one_moment_center.(lg, FT, ρ, e)
    return moist_center.(lg, FT, ρ, e)
end

function test_state(
    spaces,
    FT;
    moist,
    categories = false,
    slab,
    ρ = ρ_VAL,
    e = E_VAL,
)
    c = center_field(spaces.cent_space, FT, moist, categories, ρ, e)
    slab || return Fields.FieldVector(; c)
    surface_space = Fields.level(spaces.face_space, Fields.half)
    sfc = slab_variables.(Fields.local_geometry_field(surface_space), FT)
    return Fields.FieldVector(; c, sfc)
end

# A slab that counts how often the ledger reduces `Y.sfc.water`. The counter is
# what pins the single reduction: the slab's water and its mass are two
# projections of one endpoint, so reading the field twice would pay for a second
# device reduction to learn a number it already had.
struct CountingSlab{FT} <: CA.SurfaceConditions.SurfaceTemperature
    inner::CA.SurfaceConditions.SlabOceanTemperature{FT}
end

const SFC_WATER_READS = Ref(0)

PB.has_surface_reservoir(::CountingSlab) = true

PB.local_surface_energy(Y, slab::CountingSlab) =
    PB.local_surface_energy(Y, slab.inner)

function PB.local_surface_water(Y, slab::CountingSlab)
    SFC_WATER_READS[] += 1
    return PB.local_surface_water(Y, slab.inner)
end

PB.local_surface_mass(Y, slab::CountingSlab) = PB.local_surface_water(Y, slab)

# The five supported combinations of moisture model and surface. A dry slab is
# in the list because it is the case only a real state can settle: the state
# carries `Y.sfc.water` whatever the moisture model is.
function configurations(FT)
    prescribed = CA.SurfaceConditions.ExternalTemperature()
    slab = CA.SurfaceConditions.SlabOceanTemperature{FT}()
    return (
        (;
            name = "dry model, prescribed surface",
            surface = prescribed,
            model = CA.DryModel(),
            moist = false,
            categories = false,
            slab = false,
            owns_water = false,
            has_slab = false,
        ),
        (;
            name = "dry model, slab surface",
            surface = slab,
            model = CA.DryModel(),
            moist = false,
            categories = false,
            slab = true,
            owns_water = false,
            has_slab = true,
        ),
        (;
            name = "moist model, prescribed surface",
            surface = prescribed,
            model = CA.EquilibriumMicrophysics0M(),
            moist = true,
            categories = false,
            slab = false,
            owns_water = true,
            has_slab = false,
        ),
        (;
            name = "0M moist model, slab surface",
            surface = slab,
            model = CA.EquilibriumMicrophysics0M(),
            moist = true,
            categories = false,
            slab = true,
            owns_water = true,
            has_slab = true,
        ),
        (;
            name = "1M moist model, slab surface",
            surface = slab,
            model = CA.NonEquilibriumMicrophysics1M(),
            moist = true,
            categories = true,
            slab = true,
            owns_water = true,
            has_slab = true,
        ),
    )
end

state_for(spaces, FT, config; kwargs...) = test_state(
    spaces,
    FT;
    moist = config.moist,
    categories = config.categories,
    slab = config.slab,
    kwargs...,
)

schema_for(config) = PB.endpoint_schema(config.surface, config.model)

endpoints_for(spaces, FT, config; kwargs...) = PB.budget_endpoints(
    state_for(spaces, FT, config; kwargs...),
    schema_for(config),
    config.surface,
    0,
)

reservoir_endpoint(endpoints, reservoir) =
    only(filter(e -> e.reservoir === reservoir, endpoints.reservoirs))

atmosphere_endpoint(endpoints) =
    reservoir_endpoint(endpoints, PB.AtmosphereReservoir())

surface_endpoint(endpoints) =
    reservoir_endpoint(endpoints, PB.SlabSurfaceReservoir())

amount(endpoint, quantity) = PB.budget_component(endpoint, quantity).amount

status_of(endpoint, quantity) =
    PB.component_status(PB.budget_component(endpoint, quantity))

control_volume_names(schema) = [cv.name for cv in schema.control_volumes]

@testset "Parent-budget endpoints on real state" begin
    for FT in (Float32, Float64)
        # One set of spaces per float type; building them is the expensive part.
        spaces = get_spherical_extruded_spaces(; FT)

        @testset "Applicability and reservoirs ($FT)" begin
            for config in configurations(FT)
                @testset "$(config.name)" begin
                    schema = schema_for(config)
                    endpoints = endpoints_for(spaces, FT, config)

                    expected = config.has_slab ? 2 : 1
                    @test length(endpoints.reservoirs) == expected
                    @test length(schema.reservoirs) == expected

                    atmosphere = atmosphere_endpoint(endpoints)
                    @test status_of(atmosphere, :mass) isa PB.Measured
                    @test status_of(atmosphere, :energy) isa PB.Measured
                    if config.owns_water
                        @test status_of(atmosphere, :water) isa PB.Measured
                        @test amount(atmosphere, :water) > 0
                    else
                        @test status_of(atmosphere, :water) isa PB.NotApplicable
                        @test amount(atmosphere, :water) == 0
                    end
                    @test PB.quantity_applicable(schema, ATMOS, :water) ==
                          config.owns_water

                    # Signs and units: mass is positive, and energy carries the
                    # sign of the field rather than a magnitude.
                    @test amount(atmosphere, :mass) > 0
                    @test amount(atmosphere, :energy) > 0

                    if config.has_slab
                        surface = surface_endpoint(endpoints)
                        @test status_of(surface, :energy) isa PB.Measured
                        @test amount(surface, :energy) > 0
                        if config.owns_water
                            @test status_of(surface, :water) isa PB.Measured
                            @test status_of(surface, :mass) isa PB.Measured
                            # Two projections of one endpoint. They read the
                            # same `Y.sfc.water` through one reduction, so this
                            # pins the equality rather than pretending it is an
                            # independent measurement.
                            @test amount(surface, :mass) ==
                                  amount(surface, :water)
                            @test amount(surface, :water) > 0
                        else
                            # `Y.sfc.water` is built whatever the moisture model
                            # is, so field presence cannot decide ownership. The
                            # schema does, and it says the quantity is not one
                            # this configuration has.
                            @test status_of(surface, :water) isa
                                  PB.NotApplicable
                            @test status_of(surface, :mass) isa PB.NotApplicable
                            @test !PB.quantity_applicable(schema, SLAB, :water)
                        end
                    end

                    # The coupled view exists exactly when the reservoir it
                    # names does. Declaring it otherwise would report the
                    # atmosphere-only numbers under the coupled name.
                    views = control_volume_names(schema)
                    @test first(views) === :atmosphere_only
                    @test (:atmosphere_and_surface in views) == config.has_slab
                    @test length(views) == expected
                end
            end
        end

        @testset "Slab water is reduced once per endpoint packet ($FT)" begin
            config = configurations(FT)[4]
            counting = CountingSlab(config.surface)
            Y = state_for(spaces, FT, config)
            schema = PB.endpoint_schema(counting, config.model)

            SFC_WATER_READS[] = 0
            packet = PB.local_endpoint_packet(Y, schema, counting)
            # One reduction fills both slots. Calling the mass projection as
            # well would reduce the same field a second time to obtain a number
            # it cannot disagree with.
            @test SFC_WATER_READS[] == 1
            @test PB.packet_local_value(packet, SLAB, :mass) ==
                  PB.packet_local_value(packet, SLAB, :water)
            @test PB.slot_state(packet, SLAB, :mass) isa PB.MeasuredSlot
            # Every slot has an explicit disposition before the packet leaves
            # the assembly path.
            @test isempty(PB.unresolved_slots(packet))
        end

        @testset "A dry slab resolves its slots without measuring them ($FT)" begin
            config = configurations(FT)[2]
            Y = state_for(spaces, FT, config)
            schema = schema_for(config)
            packet = PB.local_endpoint_packet(Y, schema, config.surface)
            # Not applicable is a positive act with a configuration behind it,
            # and it is not what an unwritten slot looks like.
            @test PB.slot_state(packet, SLAB, :water) isa PB.NotApplicableSlot
            @test PB.slot_state(packet, SLAB, :mass) isa PB.NotApplicableSlot
            @test PB.slot_state(packet, SLAB, :energy) isa PB.MeasuredSlot
            @test PB.slot_state(packet, ATMOS, :water) isa PB.NotApplicableSlot
            @test isempty(PB.unresolved_slots(packet))
        end

        @testset "Accounting stays Float64 whatever the state is ($FT)" begin
            # The residual is what survives subtracting two large global totals.
            # Carrying it in the state's type would destroy it for a Float32
            # run, so the ledger's own arithmetic is always Float64.
            for config in configurations(FT)
                endpoints = endpoints_for(spaces, FT, config)
                @test endpoints isa PB.BudgetEndpoints{Float64}
                for endpoint in endpoints.reservoirs
                    for quantity in PB.BUDGET_QUANTITIES
                        @test amount(endpoint, quantity) isa Float64
                    end
                end
            end
        end

        @testset "Endpoints agree with an independent integral ($FT)" begin
            config = configurations(FT)[4]
            Y = state_for(spaces, FT, config)
            endpoints = endpoints_for(spaces, FT, config)
            atmosphere = atmosphere_endpoint(endpoints)

            # `sum` is ClimaCore's own reduction: it accumulates in the state's
            # type and then reduces across ranks. The ledger accumulates in the
            # accounting type instead, so the two agree to the state's precision
            # and not beyond it. Anything worse than that would mean the ledger
            # is integrating something else.
            tol = 10 * sqrt(eps(FT))
            @test amount(atmosphere, :mass) ≈ sum(Y.c.ρ) rtol = tol
            @test amount(atmosphere, :water) ≈ sum(Y.c.ρq_tot) rtol = tol
            @test amount(atmosphere, :energy) ≈ sum(Y.c.ρe_tot) rtol = tol

            # The fields are uniform, so every integral is its value times the
            # same volume. Comparing ratios checks the integrals against each
            # other without restating the quadrature.
            m = amount(atmosphere, :mass)
            @test amount(atmosphere, :water) ≈ m * (Q_VAL / ρ_VAL) rtol = tol
            @test amount(atmosphere, :energy) ≈ m * (E_VAL / ρ_VAL) rtol = tol
        end

        @testset "Categories partition water and are never added ($FT)" begin
            plain = configurations(FT)[4]
            with_categories = configurations(FT)[5]
            plain_endpoints = endpoints_for(spaces, FT, plain)
            category_endpoints = endpoints_for(spaces, FT, with_categories)

            # Same `ρq_tot`, but one state also carries cloud and rain. Total
            # water is the integral of `ρq_tot` alone, so the two must agree
            # exactly. Adding a category would invent water every time
            # condensate became rain.
            for quantity in PB.BUDGET_QUANTITIES
                @test amount(atmosphere_endpoint(category_endpoints), quantity) ≈
                      amount(atmosphere_endpoint(plain_endpoints), quantity) rtol =
                    1e-12
            end
        end

        @testset "Dry air is a derived diagnostic ($FT)" begin
            moist = configurations(FT)[4]
            dry = configurations(FT)[1]

            Y = state_for(spaces, FT, moist)
            endpoints = endpoints_for(spaces, FT, moist)
            atmosphere = atmosphere_endpoint(endpoints)
            # `∫(ρ - ρq_tot)` is written as one integral so the cancellation is
            # pointwise. It still has to agree with the difference of the two.
            @test PB.atmosphere_dry_mass(Y) ≈
                  amount(atmosphere, :mass) - amount(atmosphere, :water) rtol =
                1e-9

            # With no water state there is no water to subtract.
            dry_Y = state_for(spaces, FT, dry)
            dry_endpoints = endpoints_for(spaces, FT, dry)
            @test PB.atmosphere_dry_mass(dry_Y) ==
                  amount(atmosphere_endpoint(dry_endpoints), :mass)
        end

        @testset "Slab energy and water share one area ($FT)" begin
            config = configurations(FT)[4]
            endpoints = endpoints_for(spaces, FT, config)
            surface = surface_endpoint(endpoints)
            slab = config.surface

            # Two integrals of the same 2D space, so the area they imply has to
            # match. This is the unit check: `W_sfc` is a water content times an
            # area and `E_sfc` is a temperature times an areal heat capacity
            # times the same area.
            area = amount(surface, :water) / SFC_WATER
            expected_energy = SFC_T * area * PB.slab_heat_capacity(slab)
            @test amount(surface, :energy) ≈ expected_energy rtol =
                10 * sqrt(eps(FT))
        end

        @testset "Energy carries its sign ($FT)" begin
            config = configurations(FT)[3]
            positive = endpoints_for(spaces, FT, config)
            negative = endpoints_for(spaces, FT, config; e = -E_VAL)
            # Total energy has no physical zero, so a shifted reference can put
            # it below zero. Nothing in the endpoint path may take a magnitude.
            @test amount(atmosphere_endpoint(negative), :energy) ==
                  -amount(atmosphere_endpoint(positive), :energy)
            @test amount(atmosphere_endpoint(negative), :energy) < 0
        end

        @testset "A small increment survives a large background ($FT)" begin
            config = configurations(FT)[3]
            before = endpoints_for(spaces, FT, config)
            after =
                endpoints_for(spaces, FT, config; ρ = ρ_VAL * (1 + Δρ_REL))

            m_before = amount(atmosphere_endpoint(before), :mass)
            m_after = amount(atmosphere_endpoint(after), :mass)

            # Both states hold their density exactly, and the two integrals
            # use the same geometric weights, so the weights cancel out of the
            # difference and the increment is known analytically. The increment
            # is about 1.2e-4 of the background. A Float32 accumulation over
            # this many degrees of freedom carries a relative error of order
            # 1e-6 to 1e-5, which is a percent or more of the increment, and
            # converting the finished sum to Float64 afterwards would not bring
            # any of it back. Accumulating in the accounting type instead
            # recovers the increment several orders of magnitude better than
            # that, whatever the state's type.
            @test m_after - m_before ≈ Δρ_REL * m_before rtol = 1e-6
            @test m_after > m_before
        end

        @testset "A transaction runs on measured endpoints ($FT)" begin
            config = configurations(FT)[4]
            schema = schema_for(config)
            ledger = PB.BudgetLedger{PB.BUDGET_ACCOUNTING_TYPE}(schema)

            opening = endpoints_for(spaces, FT, config)
            PB.open_transaction!(ledger, opening)
            closing = PB.budget_endpoints(
                state_for(spaces, FT, config; ρ = ρ_VAL * (1 + Δρ_REL)),
                schema,
                config.surface,
                1,
            )
            commit = PB.commit_transaction!(ledger, closing)
            @test commit.step == 1
            @test ledger.committed_steps == 1

            # Nothing was recorded, and the schema declares no channel yet, so
            # there is no expectation to miss. The endpoint change is real and
            # entirely unexplained, and with no calibrated tolerance the claim
            # is blocked rather than passed.
            mass = only(
                filter(
                    r -> r.quantity === :mass &&
                         r.control_volume === :atmosphere_only,
                    commit.parent,
                ),
            )
            @test mass.endpoint_change > 0
            @test mass.recorded == 0
            @test mass.residual == mass.endpoint_change
            @test isempty(mass.missing_expectations)
            @test mass.status === :blocked
            @test PB.UNCALIBRATED_TOLERANCE_BLOCKER in mass.blocked_by
        end
    end
end
