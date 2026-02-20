#=
Allocation regression tests for microphysics code.

Consolidates all allocation tests to guard against
accidental heap allocations that break GPU safety.
=#

using Test
using ClimaAtmos
import ClimaAtmos as CA

import Thermodynamics as TD
import ClimaParams as CP
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.BulkMicrophysicsTendencies as BMT

import ClimaAtmos: limit_sink, coupled_sink_limit_factor

import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore as CC
import ClimaAtmos.Parameters as CAP

include("../../test_helpers.jl")

# ============================================================================
# Helper functions for integration-level allocation tests
# ============================================================================

"""
    test_microphysics_allocations!(ᶜYₜ, Y, p)

Test that `microphysics_tendency!` has stable allocations (no growth).
Runs twice: once for warmup, once for measurement.
The second call should not allocate more than the first.
"""
function test_microphysics_allocations!(ᶜYₜ, Y, p)
    FT = eltype(Y)
    (; turbconv_model, microphysics_model) = p.atmos

    # Warmup (triggers JIT compilation)
    ᶜYₜ .= zero(eltype(ᶜYₜ))
    CA.microphysics_tendency!(
        ᶜYₜ, Y, p, FT(0),
        microphysics_model, turbconv_model,
    )

    # Measure allocations on second call
    ᶜYₜ .= zero(eltype(ᶜYₜ))
    allocs = @allocated CA.microphysics_tendency!(
        ᶜYₜ, Y, p, FT(0),
        microphysics_model, turbconv_model,
    )
    return allocs
end

"""
    test_cache_allocations!(Y, p)

Test that `set_microphysics_tendency_cache!` is zero-allocation.
This catches NamedTuple field allocations from functions like `limited_tendency_2m`.
"""
function test_cache_allocations!(Y, p)
    (; turbconv_model, microphysics_model) = p.atmos

    # Warmup
    CA.set_microphysics_tendency_cache!(Y, p, microphysics_model, turbconv_model)

    # Measure
    allocs = @allocated CA.set_microphysics_tendency_cache!(
        Y, p, microphysics_model, turbconv_model,
    )
    return allocs
end

# ============================================================================
# Scalar-level allocation tests
#
# NOTE: Each test uses a "function barrier" pattern: the warmup + @allocated
# call is wrapped in a helper function that is itself called twice. This
# prevents @allocated from reporting false-positive allocations caused by
# boxed temporaries at top-level scope.
# ============================================================================

# --- Function barrier helpers ---

function _allocs_cloud_fraction_sd(thp)
    FT = Float64
    T = FT(280.0)
    ρ = FT(1.0)
    CA.compute_cloud_fraction_sd(thp, T, ρ, FT(1e-3), FT(0), FT(1), FT(1e-6), FT(0))
    return @allocated CA.compute_cloud_fraction_sd(
        thp, T, ρ, FT(1e-3), FT(0), FT(1), FT(1e-6), FT(0),
    )
end

function _allocs_coupled_sink()
    coupled_sink_limit_factor(-0.01, -1e8, 0.001, 1e8, 1.0)
    return @allocated coupled_sink_limit_factor(-0.01, -1e8, 0.001, 1e8, 1.0)
end

function _allocs_bmt_0m(mp, thp)
    FT = Float64
    T = FT(280.0)
    q_liq = FT(0.001)
    q_ice = FT(0.0005)
    BMT.bulk_microphysics_tendencies(BMT.Microphysics0Moment(), mp, thp, T, q_liq, q_ice)
    return @allocated BMT.bulk_microphysics_tendencies(
        BMT.Microphysics0Moment(), mp, thp, T, q_liq, q_ice,
    )
end

function _allocs_energy_helper(thp)
    FT = Float64
    T = FT(280.0)
    q_liq = FT(0.001)
    q_ice = FT(0.0005)
    Φ = FT(1000.0)
    CA.e_tot_0M_precipitation_sources_helper(thp, T, q_liq, q_ice, Φ)
    return @allocated CA.e_tot_0M_precipitation_sources_helper(thp, T, q_liq, q_ice, Φ)
end

function _allocs_limit_sink()
    FT = Float64
    limit_sink(FT(-0.01), FT(0.01), FT(60.0), 1)
    return @allocated limit_sink(FT(-0.01), FT(0.01), FT(60.0), 1)
end

function _allocs_sgs_sat_adj(thp, quad)
    FT = Float64
    ρ = FT(1.0)
    T_mean = FT(280.0)
    q_mean = FT(0.01)
    T′T′ = FT(1.0)
    q′q′ = FT(1e-6)
    corr_Tq = FT(0.6)
    ClimaAtmos.compute_sgs_saturation_adjustment(
        thp, quad, ρ, T_mean, q_mean, T′T′, q′q′, corr_Tq,
    )
    return @allocated ClimaAtmos.compute_sgs_saturation_adjustment(
        thp, quad, ρ, T_mean, q_mean, T′T′, q′q′, corr_Tq,
    )
end

# NOTE: closure-based tests are omitted here because Julia closures inherently
# allocate ~352 bytes from closure capture. In production, integrate_over_sgs
# is always called with functors (e.g. MicrophysicsEvaluator), which are
# zero-allocation.

struct TestEvaluator{FT}
    a::FT
    b::FT
end
(ev::TestEvaluator)(T, q) = ev.a * T + ev.b * q

function _allocs_integrate_sgs_functor(quad, ::Type{FT}) where {FT}
    T_mean = FT(280.0)
    q_mean = FT(0.01)
    T′T′ = FT(1.0)
    q′q′ = FT(1e-6)
    corr_Tq = FT(0.6)
    evaluator = TestEvaluator(FT(2), FT(3))
    ClimaAtmos.integrate_over_sgs(evaluator, quad, q_mean, T_mean, q′q′, T′T′, corr_Tq)
    return @allocated ClimaAtmos.integrate_over_sgs(
        evaluator, quad, q_mean, T_mean, q′q′, T′T′, corr_Tq,
    )
end

# --- Tests ---

@testset "Allocation Tests" begin

    @testset "Scalar functions" begin

        FT = Float64
        toml_dict = CP.create_toml_dict(FT)
        thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
        mp = CMP.Microphysics0MParams(toml_dict)
        quad = ClimaAtmos.SGSQuadrature(FT; quadrature_order = 3)

        # Warm up the barrier functions themselves (JIT)
        _allocs_cloud_fraction_sd(thp)
        _allocs_coupled_sink()
        _allocs_bmt_0m(mp, thp)
        _allocs_energy_helper(thp)
        _allocs_limit_sink()
        _allocs_sgs_sat_adj(thp, quad)
        _allocs_integrate_sgs_functor(quad, FT)

        @testset "compute_cloud_fraction_sd" begin
            @test _allocs_cloud_fraction_sd(thp) == 0
        end

        @testset "coupled_sink_limit_factor" begin
            @test _allocs_coupled_sink() == 0
        end

        @testset "BMT 0M + energy helper + limit_sink" begin
            @test _allocs_bmt_0m(mp, thp) == 0
            @test _allocs_energy_helper(thp) == 0
            @test _allocs_limit_sink() == 0
        end

        @testset "compute_sgs_saturation_adjustment" begin
            @test _allocs_sgs_sat_adj(thp, quad) == 0
        end

        @testset "integrate_over_sgs (functor)" begin
            for FTi in (Float32, Float64)
                @testset "FT = $FTi" begin
                    qi = ClimaAtmos.SGSQuadrature(FTi; quadrature_order = 3)
                    _allocs_integrate_sgs_functor(qi, FTi)
                    @test _allocs_integrate_sgs_functor(qi, FTi) == 0
                end
            end
        end
    end

    # ============================================================================
    # Integration-level allocation tests (full simulation setups)
    # ============================================================================

    @testset "Integration" begin

        @testset "Equilibrium moisture + 0-moment" begin
            config = CA.AtmosConfig(
                Dict(
                    "initial_condition" => "DYCOMS_RF02",
                    "microphysics_model" => "0M",
                    "config" => "column",
                    "output_default_diagnostics" => false,
                ),
                job_id = "alloc_0M",
            )
            (; Y, p, params) = generate_test_simulation(config)
            FT = eltype(Y)
            ᶜYₜ = zero(Y)

            @testset "allocation stability" begin
                allocs = test_microphysics_allocations!(ᶜYₜ, Y, p)
                @test allocs == 0
            end
            @testset "cache allocation stability" begin
                allocs = test_cache_allocations!(Y, p)
                @test allocs == 0
            end
        end

        @testset "NonEquilibrium moisture + 1-moment" begin
            config = CA.AtmosConfig(
                Dict(
                    "initial_condition" => "PrecipitatingColumn",
                    "microphysics_model" => "1M",
                    "config" => "column",
                    "output_default_diagnostics" => false,
                ),
                job_id = "alloc_1M",
            )
            (; Y, p, params) = generate_test_simulation(config)
            FT = eltype(Y)
            ᶜYₜ = zero(Y)

            @testset "allocation stability" begin
                allocs = test_microphysics_allocations!(ᶜYₜ, Y, p)
                @test allocs == 0
            end
            @testset "cache allocation stability" begin
                allocs = test_cache_allocations!(Y, p)
                @test allocs == 0
            end
        end

        @testset "NonEquilibrium moisture + 2-moment" begin
            config = CA.AtmosConfig(
                Dict(
                    "initial_condition" => "PrecipitatingColumn",
                    "microphysics_model" => "2M",
                    "config" => "column",
                    "output_default_diagnostics" => false,
                    "prescribed_aerosols" => ["SSLT01"],
                ),
                job_id = "alloc_2M",
            )
            (; Y, p, params) = generate_test_simulation(config)
            FT = eltype(Y)
            ᶜYₜ = zero(Y)

            @testset "allocation stability" begin
                allocs = test_microphysics_allocations!(ᶜYₜ, Y, p)
                @test allocs <= 160  # pre-existing: aerosol lookup
            end
            @testset "cache allocation stability" begin
                allocs = test_cache_allocations!(Y, p)
                @test allocs == 0
            end
        end

        @testset "NonEquilibrium moisture + 1-moment + SGS quadrature" begin
            config = CA.AtmosConfig(
                Dict(
                    "initial_condition" => "PrecipitatingColumn",
                    "microphysics_model" => "1M",
                    "config" => "column",
                    "use_sgs_quadrature" => true,
                    "sgs_distribution" => "gaussian",
                    "quadrature_order" => 2,
                    "output_default_diagnostics" => false,
                ),
                job_id = "alloc_1M_sgs",
            )
            (; Y, p, params) = generate_test_simulation(config)
            FT = eltype(Y)
            ᶜYₜ = zero(Y)

            @testset "allocation stability (with quadrature)" begin
                allocs = test_microphysics_allocations!(ᶜYₜ, Y, p)
                @test allocs == 0
            end
            @testset "cache allocation stability" begin
                allocs = test_cache_allocations!(Y, p)
                @test allocs == 0
            end
        end
    end

end
