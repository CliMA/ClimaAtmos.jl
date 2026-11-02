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

function _allocs_cloud_fraction_simple()
    FT = Float64
    σ_abs = FT(1e-7)
    CA._compute_cloud_fraction(
        FT(5e-5), FT(-1e-4), FT(3.16e-4), FT(1), σ_abs,
    )
    return @allocated CA._compute_cloud_fraction(
        FT(5e-5),
        FT(-1e-4),
        FT(3.16e-4),
        FT(1),
        σ_abs,
    )
end

function _allocs_cloud_fraction_fused(thp, sgs_quad)
    FT = Float64
    T = FT(280.0)
    ρ = FT(1.0)
    q_tot = FT(0.011)
    q_liq = FT(1e-3)
    q_ice = FT(0)
    T′T′ = FT(1.0)
    q′q′ = FT(1e-6)
    corr_Tq = FT(0)
    α = FT(1)
    σ_abs = FT(1e-7)
    c = FT(3)
    CA._compute_cloud_fraction(
        thp,
        T,
        ρ,
        q_tot,
        q_liq,
        q_ice,
        sgs_quad,
        T′T′,
        q′q′,
        corr_Tq,
        α,
        σ_abs,
        c,
    )
    return @allocated CA._compute_cloud_fraction(
        thp,
        T,
        ρ,
        q_tot,
        q_liq,
        q_ice,
        sgs_quad,
        T′T′,
        q′q′,
        corr_Tq,
        α,
        σ_abs,
        c,
    )
end

function _allocs_coupled_sink()
    coupled_sink_limit_factor(-0.01, -1e8, 0.001, 1e8, 1.0)
    return @allocated coupled_sink_limit_factor(-0.01, -1e8, 0.001, 1e8, 1.0)
end

function _allocs_bmt_0m(mp, thp)
    FT = Float64
    T = FT(280.0)
    ρ = FT(1.0)
    q_liq = FT(0.001)
    q_ice = FT(0.0005)
    q_vap_sat = TD.q_vap_saturation(thp, T, ρ)
    BMT.bulk_microphysics_tendencies(
        BMT.Microphysics0Moment(),
        mp,
        thp,
        T,
        q_liq,
        q_ice,
        q_vap_sat,
    )
    return @allocated BMT.bulk_microphysics_tendencies(
        BMT.Microphysics0Moment(), mp, thp, T, q_liq, q_ice, q_vap_sat,
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

function _allocs_entr_vel_scale(thp, tc)
    FT = eltype(tc.max_area)
    CA.entrainment_velocity_scale(
        thp,
        tc,
        FT(3500),
        zero(FT),
        FT(70000),
        FT(0.9),
        FT(0.02),
        FT(2),
        one(FT),
        FT(0.01),
        zero(FT),
        FT(0.35),
        zero(FT),
        FT(0.5),
        CA.InvZEntrainment(),
    )
    return @allocated CA.entrainment_velocity_scale(
        thp,
        tc,
        FT(3500),
        zero(FT),
        FT(70000),
        FT(0.9),
        FT(0.02),
        FT(2),
        one(FT),
        FT(0.01),
        zero(FT),
        FT(0.35),
        zero(FT),
        FT(0.5),
        CA.InvZEntrainment(),
    )
end

function _allocs_sgs_sat_adj(thp, quad)
    FT = Float64
    ρ = FT(1.0)
    T_mean = FT(280.0)
    q_mean = FT(0.01)
    T′T′ = FT(1.0)
    q′q′ = FT(1e-6)
    corr_Tq = FT(0.6)
    q_min = FT(1e-10)
    ClimaAtmos.compute_sgs_saturation_adjustment(
        thp, quad, ρ, T_mean, q_mean, T′T′, q′q′, corr_Tq,
    )
    return @allocated ClimaAtmos.compute_sgs_saturation_adjustment(
        thp, quad, ρ, T_mean, q_mean, T′T′, q′q′, corr_Tq,
    )
end

function _allocs_displacement_condensate(::Type{FT}) where {FT}
    σ = FT(1e-3)
    args = (FT(-1.4e-3), σ, FT(0.8) * σ, FT(-0.5) * σ)
    ClimaAtmos.displacement_condensate(args...)
    return @allocated ClimaAtmos.displacement_condensate(args...)
end

function _allocs_sgs_moments_curvature(thp, quad, ::Type{FT}) where {FT}
    ρ, T = FT(1.1), FT(288)
    q_tot = TD.q_vap_saturation(thp, T, ρ) - FT(3e-4)
    args = (thp, ρ, T, q_tot, FT(5e-5), quad, FT(0.25), FT(2.5e-7), FT(0.6), FT(1))
    ClimaAtmos._compute_sgs_moments(args..., q_tot, FT(4e-4), FT(-3e-4))
    return @allocated ClimaAtmos._compute_sgs_moments(
        args..., q_tot, FT(4e-4), FT(-3e-4),
    )
end

function _allocs_conditional_1m(thp, cmp, quad, ::Type{FT}, T = FT(285)) where {FT}
    ρ = FT(1.1)
    q_sat = TD.q_vap_saturation(thp, T, ρ)
    mu_S = FT(-1.5) * FT(4e-4)
    q_tot = q_sat + mu_S
    q_lcl = FT(3e-5)
    q_rai = FT(1e-5)
    mom = ClimaAtmos._compute_sgs_moments(
        thp, ρ, T, q_tot, q_lcl, quad, FT(0.25), FT(2.5e-7), FT(0.5), FT(0.7),
        q_tot - q_rai,
    )
    args = (
        BMT.Microphysics1Moment(), quad, cmp, thp, ρ, T, FT(0.1), q_tot, q_lcl,
        FT(0), q_rai, FT(0), FT(0.25), FT(2.5e-7), FT(0.5), mom.λ_lagrange,
        FT(0.7), FT(0.1), FT(0.1), FT(180), 2, FT(1), mu_S, FT(0.5), FT(0.4),
        mom.sigma_S, FT(0.1), mom.λ_eq,
    )
    ClimaAtmos.microphysics_tendencies_1m(args...)
    a_mom = @allocated ClimaAtmos._compute_sgs_moments(
        thp, ρ, T, q_tot, q_lcl, quad, FT(0.25), FT(2.5e-7), FT(0.5), FT(0.7),
        q_tot - q_rai,
    )
    a_tend = @allocated ClimaAtmos.microphysics_tendencies_1m(args...)
    return a_mom + a_tend
end

# NOTE: closure-based tests are omitted here because Julia closures inherently
# allocate ~352 bytes from closure capture. In production, integrate_over_sgs
# is always called with functors (e.g. Microphysics1MEvaluator), which are
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

function _allocs_edmfx_vel!(Y, p)
    (; turbconv_model, microphysics_model) = p.atmos
    CA.set_precipitation_velocities!(Y, p, microphysics_model, turbconv_model)
    return @allocated CA.set_precipitation_velocities!(
        Y, p, microphysics_model, turbconv_model,
    )
end

function _allocs_edmfx_ed!(ᶜYₜ, Y, p)
    FT = eltype(Y)
    (; turbconv_model) = p.atmos
    CA.edmfx_entr_detr_tendency!(ᶜYₜ, Y, p, FT(0), turbconv_model)
    return @allocated CA.edmfx_entr_detr_tendency!(
        ᶜYₜ, Y, p, FT(0), turbconv_model,
    )
end

function _allocs_edmfx_sweep!(Y, p)
    FT = eltype(Y)
    sgs_tracers = (
        CA.MatrixFields.@name(mse),
        CA.MatrixFields.@name(q_tot),
        CA.MatrixFields.@name(q_lcl),
        CA.MatrixFields.@name(q_icl),
        CA.MatrixFields.@name(q_rai),
        CA.MatrixFields.@name(q_sno),
    )
    CA._sweep_sgs_updraft_tracers!(Y, p, FT(100), sgs_tracers)
    return @allocated CA._sweep_sgs_updraft_tracers!(
        Y, p, FT(100), sgs_tracers,
    )
end

function _measure_edmfx_allocs(enable_shaft_and_support::Bool)
    return mktemp() do toml_path, io
        if enable_shaft_and_support
            write(
                io,
                """
                [sgs_precip_incloud_fraction]
                value = 0.75
                type = "float"
                [sgs_precip_fraction_floor]
                value = 0.2
                type = "float"
                [edmf_tail_entr_weight]
                value = 0.5
                type = "float"
                [edmf_buoyancy_clear_env_weight]
                value = 0.5
                type = "float"
                [edmf_shallow_support_height_scale]
                value = 2200.0
                type = "float"
                [entr_subsat_coeff]
                value = 5.0
                type = "float"
                [entr_subsat_power]
                value = 2.0
                type = "float"
                [entr_subsat_height_min]
                value = 0.2
                type = "float"
                [entr_subsat_height_width]
                value = 0.1
                type = "float"
                [sgs_ice_sedimentation_cf_floor]
                value = 0.02
                type = "float"
                """,
            )
        else
            write(
                io,
                """
                [sgs_precip_incloud_fraction]
                value = 0.0
                type = "float"
                [edmf_tail_entr_weight]
                value = 0.0
                type = "float"
                [edmf_buoyancy_clear_env_weight]
                value = 0.0
                type = "float"
                [edmf_shallow_support_height_scale]
                value = 0.0
                type = "float"
                [entr_subsat_coeff]
                value = 0.0
                type = "float"
                [entr_subsat_power]
                value = 2.0
                type = "float"
                [entr_subsat_height_min]
                value = 0.2
                type = "float"
                [entr_subsat_height_width]
                value = 0.1
                type = "float"
                """,
            )
        end
        flush(io)
        config = CA.AtmosConfig(
            Dict(
                "initial_condition" => "Bomex",
                "microphysics_model" => "1M",
                "cloud_ice_terminal_velocity" => "Chen2022",
                "rain_terminal_velocity" => "Chen2022",
                "snow_terminal_velocity" => "Chen2022",
                "turbconv" => "prognostic_edmfx",
                "edmfx_entr_model" => "Generalized",
                "edmfx_detr_model" => "Generalized",
                "edmfx_sgs_mass_flux" => true,
                "edmfx_sgs_diffusive_flux" => true,
                "edmfx_nh_pressure" => true,
                "edmfx_filter" => true,
                "prognostic_tke" => true,
                "config" => "column",
                "z_stretch" => false,
                "use_sgs_quadrature" => true,
                "sgs_distribution" => "gaussian",
                "quadrature_order" => 3,
                "toml" => [toml_path],
                "output_default_diagnostics" => false,
            ),
            job_id = enable_shaft_and_support ? "alloc_1M_edmfx_on" :
                     "alloc_1M_edmfx_off",
        )
        (; Y, p) = generate_test_simulation(config)
        ᶜYₜ = zero(Y)
        return (;
            mp = test_microphysics_allocations!(ᶜYₜ, Y, p),
            cache = test_cache_allocations!(Y, p),
            vel = _allocs_edmfx_vel!(Y, p),
            ed = _allocs_edmfx_ed!(ᶜYₜ, Y, p),
            sweep = _allocs_edmfx_sweep!(Y, p),
        )
    end
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
        _allocs_cloud_fraction_simple()
        _allocs_cloud_fraction_fused(thp, quad)
        _allocs_coupled_sink()
        _allocs_bmt_0m(mp, thp)
        _allocs_energy_helper(thp)
        _allocs_limit_sink()
        _allocs_sgs_sat_adj(thp, quad)
        _allocs_integrate_sgs_functor(quad, FT)

        @testset "_compute_cloud_fraction" begin
            @test _allocs_cloud_fraction_simple() == 0
            @test _allocs_cloud_fraction_fused(thp, quad) == 0
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

        @testset "displacement_condensate and curvature moments" begin
            for FTi in (Float32, Float64)
                @testset "FT = $FTi" begin
                    @test _allocs_displacement_condensate(FTi) == 0
                    qi = ClimaAtmos.SGSQuadrature(FTi; quadrature_order = 3)
                    thpi = TD.Parameters.ThermodynamicsParameters(
                        CP.create_toml_dict(FTi),
                    )
                    @test _allocs_sgs_moments_curvature(thpi, qi, FTi) == 0
                end
            end
        end

        @testset "GaussianSGS moments and 1M tendencies (node rule)" begin
            for FTi in (Float32, Float64)
                @testset "FT = $FTi" begin
                    tomli = CP.create_toml_dict(FTi)
                    thpi = TD.Parameters.ThermodynamicsParameters(tomli)
                    cmpi = CMP.Microphysics1MParams(tomli)
                    qi = ClimaAtmos.SGSQuadrature(
                        FTi;
                        quadrature_order = 3,
                        distribution = ClimaAtmos.GaussianSGS(),
                    )
                    # 285 K: no temperature split; 273.4 K: the spread straddles freezing
                    for Ti in FTi.((285, 273.4))
                        _allocs_conditional_1m(thpi, cmpi, qi, FTi, Ti)
                        @test _allocs_conditional_1m(thpi, cmpi, qi, FTi, Ti) == 0
                    end
                end
            end
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
        @testset "InvZEntrainment subsaturation velocity scale" begin
            for FTi in (Float32, Float64)
                @testset "FT = $FTi" begin
                    thpi = CAP.thermodynamics_params(CA.ClimaAtmosParameters(FTi))
                    tc_active = CA.TurbulenceConvectionParameters(
                        FTi,
                        (;
                            entr_subsat_coeff = FTi(5),
                            entr_subsat_power = FTi(2),
                            entr_subsat_height_min = FTi(0.2),
                            entr_subsat_height_width = FTi(0.1),
                        ),
                    )
                    _allocs_entr_vel_scale(thpi, tc_active)
                    @test _allocs_entr_vel_scale(thpi, tc_active) == 0
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
                    "z_stretch" => false,
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
                    "z_stretch" => false,
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

        # DISABLED (CloudMicrophysics 0.37 compat): 2-moment microphysics is
        # temporarily blocked by an `@assert` in `precomputed_quantities`
        # (src/cache/precomputed_quantities.jl) - `generate_test_simulation`
        # throws `AssertionError` before this testset's body can run.
        # Re-enable once 2M/2M+P3 compatibility with CloudMicrophysics 0.37
        # is restored.
        # @testset "NonEquilibrium moisture + 2-moment" begin
        #     config = CA.AtmosConfig(
        #         Dict(
        #             "initial_condition" => "PrecipitatingColumn",
        #             "microphysics_model" => "2M",
        #             "config" => "column",
        #             "output_default_diagnostics" => false,
        #             "use_sgs_quadrature" => false,
        #             "prescribed_aerosols" => ["SSLT01"],
        #         ),
        #         job_id = "alloc_2M",
        #     )
        #     (; Y, p, params) = generate_test_simulation(config)
        #     FT = eltype(Y)
        #     ᶜYₜ = zero(Y)
        #
        #     @testset "allocation stability" begin
        #         allocs = test_microphysics_allocations!(ᶜYₜ, Y, p)
        #         @test allocs <= 160  # pre-existing: aerosol lookup
        #     end
        #     @testset "cache allocation stability" begin
        #         allocs = test_cache_allocations!(Y, p)
        #         @test allocs == 0
        #     end
        # end

        @testset "NonEquilibrium moisture + 1-moment + SGS quadrature" begin
            config = CA.AtmosConfig(
                Dict(
                    "initial_condition" => "PrecipitatingColumn",
                    "microphysics_model" => "1M",
                    "config" => "column",
                    "z_stretch" => false,
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

        @testset "PrognosticEDMFX + 1-moment + subgrid shaft + shallow support" begin
            allocs_off = _measure_edmfx_allocs(false)
            allocs_on = _measure_edmfx_allocs(true)
            @test allocs_on.mp == 0
            @test allocs_on.vel <= allocs_off.vel
            @test allocs_on.ed <= allocs_off.ed
            @test allocs_on.sweep <= allocs_off.sweep
            @test allocs_on.cache <= allocs_off.cache
        end
    end

end
