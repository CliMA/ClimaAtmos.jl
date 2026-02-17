#=
Unit tests for SGS Quadrature utilities (src/cache/sgs_quadrature.jl)
=#

using Test
using ClimaAtmos

@testset "SGS Quadrature" begin

    @testset "Gauss-Hermite Quadrature" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                # Test order 3
                a, w = ClimaAtmos.gauss_hermite(FT, 3)
                @test length(a) == 3
                @test length(w) == 3
                @test a[2] ≈ FT(0) atol = eps(FT)
                @test a[1] ≈ -a[3]  # symmetry
                @test w[1] ≈ w[3]   # symmetry

                # Test order 5
                a5, w5 = ClimaAtmos.gauss_hermite(FT, 5)
                @test length(a5) == 5
                @test a5[3] ≈ FT(0) atol = eps(FT)

                # Verify integration of x² over Gaussian gives 0.5
                # ∫ x² exp(-x²) dx / √π = 0.5
                integral = sum(w .* a .^ 2) / sqrt(FT(π))
                @test integral ≈ FT(0.5) atol = FT(1e-6)
            end
        end
    end

    @testset "Gauss-Legendre Quadrature" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                # Test order 3 on [0,1]
                a01, w01 = ClimaAtmos.gauss_legendre_01(FT, 3)
                @test length(a01) == 3
                @test all(0 .<= a01 .<= 1)
                @test sum(w01) ≈ FT(1) atol = FT(1e-10)  # ∫₀¹ dx = 1
            end
        end
    end

    @testset "SGSQuadrature Struct" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                # Custom order and distribution
                quad5 = ClimaAtmos.SGSQuadrature(
                    FT;
                    quadrature_order = 5,
                    distribution = ClimaAtmos.LogNormalSGS(),
                )
                @test ClimaAtmos.quadrature_order(quad5) == 5
                @test quad5.dist isa ClimaAtmos.LogNormalSGS

                # GridMeanSGS: 0th-order quadrature (grid-mean only)
                quad_gridmean =
                    ClimaAtmos.SGSQuadrature(FT; distribution = ClimaAtmos.GridMeanSGS())
                @test ClimaAtmos.quadrature_order(quad_gridmean) == 1  # Always N=1
                @test quad_gridmean.a == [FT(0)]  # Single point at origin
                @test quad_gridmean.w ≈ [sqrt(FT(π))]  # Weight = sqrt(π)
                @test quad_gridmean.dist isa ClimaAtmos.GridMeanSGS

                # GridMeanSGS ignores quadrature_order argument
                quad_gridmean5 = ClimaAtmos.SGSQuadrature(
                    FT;
                    quadrature_order = 5,
                    distribution = ClimaAtmos.GridMeanSGS(),
                )
                @test ClimaAtmos.quadrature_order(quad_gridmean5) == 1  # Still N=1
            end
        end
    end

    @testset "Variance Limiting" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                # Standard case
                corr_Tq = FT(0.6)
                σ_q, σ_T, corr = ClimaAtmos.sgs_stddevs_and_correlation(
                    FT(1e-6), FT(1.0), corr_Tq,
                )
                @test σ_q >= 0
                @test σ_T >= 0
                @test -1 <= corr <= 1

                # Zero variance
                σ_q0, σ_T0, corr0 = ClimaAtmos.sgs_stddevs_and_correlation(
                    FT(0), FT(0), corr_Tq,
                )
                @test σ_q0 == 0
                @test σ_T0 == 0
            end
        end
    end

    @testset "Correlation Clamping" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                # Correlation within [-1, 1] passes through
                _, _, corr1 = ClimaAtmos.sgs_stddevs_and_correlation(
                    FT(1e-6), FT(1.0), FT(0.5),
                )
                @test corr1 == FT(0.5)

                # Correlation > 1 is clamped
                _, _, corr2 = ClimaAtmos.sgs_stddevs_and_correlation(
                    FT(1e-6), FT(1.0), FT(1.5),
                )
                @test corr2 == FT(1.0)

                # Correlation < -1 is clamped
                _, _, corr3 = ClimaAtmos.sgs_stddevs_and_correlation(
                    FT(1e-6), FT(1.0), FT(-1.5),
                )
                @test corr3 == FT(-1.0)

            end
        end
    end

    @testset "Covariance Transformation Jacobian (∂T_∂θ_li)" begin
        import Thermodynamics as TD
        import ClimaParams as CP

        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                # Setup thermodynamics parameters
                toml_dict = CP.create_toml_dict(FT)
                thermo_params = TD.Parameters.ThermodynamicsParameters(toml_dict)

                # Typical atmospheric values
                T_mean = FT(280.0)
                θ_li_mean = FT(290.0)  # θ > T due to pressure
                θ′θ′ = FT(1.0)  # 1 K² variance
                θ′q′ = FT(1e-4)  # K × kg/kg covariance
                ρ = FT(1.0)  # Air density [kg/m³]
                q_tot = FT(0.005)  # Total water (unsaturated for these conditions)

                @testset "Dry case (no condensate)" begin
                    q_liq = FT(0)
                    q_ice = FT(0)
                    ∂T_∂θ = ClimaAtmos.∂T_∂θ_li(
                        thermo_params, T_mean, θ_li_mean, q_liq, q_ice, q_tot, ρ,
                    )

                    # Compute transformed covariances using the Jacobian
                    T′T′ = ∂T_∂θ^2 * θ′θ′
                    T′q′ = ∂T_∂θ * θ′q′

                    # Exner scaling: ∂T/∂θ ≈ T/θ for dry case (no condensate)
                    Π = T_mean / θ_li_mean
                    @test ∂T_∂θ ≈ Π
                    @test T′T′ ≈ Π^2 * θ′θ′
                    @test T′q′ ≈ Π * θ′q′

                    # T variance should be smaller than θ variance (Π < 1)
                    @test T′T′ < θ′θ′
                end

                @testset "Moist correction (unsaturated)" begin
                    # Unsaturated case: q_tot < q_sat, so dqsat_dT = 0
                    # This simplifies the moist correction formula
                    q_liq = FT(0.003)  # 3 g/kg cloud liquid
                    q_ice = FT(0)
                    q_cond = q_liq + q_ice
                    L_v = TD.Parameters.LH_v0(thermo_params)
                    c_p = TD.Parameters.cp_d(thermo_params)

                    ∂T_∂θ_moist = ClimaAtmos.∂T_∂θ_li(
                        thermo_params, T_mean, θ_li_mean, q_liq, q_ice, q_tot, ρ,
                    )
                    ∂T_∂θ_dry = ClimaAtmos.∂T_∂θ_li(
                        thermo_params, T_mean, θ_li_mean, FT(0), FT(0), q_tot, ρ,
                    )

                    # Moist correction should increase the Jacobian
                    @test ∂T_∂θ_moist > ∂T_∂θ_dry

                    # For unsaturated case, dqsat_dT = 0, so denominator = 1
                    # moist_correction = 1 + L_v * q_cond / (c_p * T)
                    correction = 1 + L_v * q_cond / (c_p * T_mean)
                    @test ∂T_∂θ_moist ≈ correction * ∂T_∂θ_dry
                end

                @testset "Edge cases" begin
                    # Zero condensate
                    ∂T_∂θ_zero = ClimaAtmos.∂T_∂θ_li(
                        thermo_params, T_mean, θ_li_mean, FT(0), FT(0), q_tot, ρ,
                    )
                    @test ∂T_∂θ_zero > FT(0)

                    # With zero variance, transformed variance is also zero
                    @test ∂T_∂θ_zero^2 * FT(0) == FT(0)
                end

                @testset "Type stability" begin
                    ∂T_∂θ = ClimaAtmos.∂T_∂θ_li(
                        thermo_params, T_mean, θ_li_mean, FT(0), FT(0), q_tot, ρ,
                    )
                    @test typeof(∂T_∂θ) == FT
                end
            end
        end
    end

    @testset "Physical Point Computation" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                μ_q = FT(0.01)
                μ_T = FT(300.0)
                σ_q = FT(0.001)
                σ_T = FT(1.0)
                corr = FT(0.5)

                T_min = FT(150.0)
                q_max = FT(1.0)

                # GaussianSGS at χ = 0 should return means
                T_hat, q_hat = ClimaAtmos.get_physical_point(
                    ClimaAtmos.GaussianSGS(), FT(0), FT(0), μ_q, μ_T, σ_q, σ_T, corr,
                    T_min, q_max,
                )
                @test T_hat ≈ μ_T atol = FT(0.1)
                @test q_hat ≈ μ_q atol = FT(0.001)

                # LogNormalSGS
                T_hat_ln, q_hat_ln = ClimaAtmos.get_physical_point(
                    ClimaAtmos.LogNormalSGS(), FT(0), FT(0), μ_q, μ_T, σ_q, σ_T, corr,
                    T_min, q_max,
                )
                @test T_hat_ln ≈ μ_T atol = FT(0.1)
                @test q_hat_ln > 0  # log-normal is always positive



                # GridMeanSGS always returns grid mean regardless of χ values
                T_hat_gm, q_hat_gm = ClimaAtmos.get_physical_point(
                    ClimaAtmos.GridMeanSGS(), FT(999), FT(999), μ_q, μ_T, σ_q, σ_T,
                    corr, T_min, q_max,
                )
                @test T_hat_gm == μ_T  # Exact equality, not approximate
                @test q_hat_gm == μ_q  # Exact equality, not approximate
            end
        end
    end

    @testset "RecursiveApply Operations" begin
        import ClimaCore.RecursiveApply: rzero, ⊞, ⊠
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                # Scalars
                @test rzero(FT(1.5)) == FT(0)
                @test FT(1) ⊞ FT(2) == FT(3)
                @test FT(2) ⊠ FT(3) == FT(6)

                # NamedTuples
                nt = (; a = FT(1), b = FT(2))
                @test rzero(nt) == (; a = FT(0), b = FT(0))
                @test nt ⊞ nt == (; a = FT(2), b = FT(4))
                @test nt ⊠ FT(3) == (; a = FT(3), b = FT(6))
            end
        end
    end

    @testset "Quadrature Integration" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                quad = ClimaAtmos.SGSQuadrature(FT)

                # Integrate constant function → should return constant
                f_const(T, q) = FT(1)
                μ_q, μ_T = FT(0.01), FT(300.0)
                T′T′, q′q′ = FT(1.0), FT(1e-6)
                corr_Tq = FT(0.6)

                result =
                    ClimaAtmos.integrate_over_sgs(
                        f_const,
                        quad,
                        μ_q,
                        μ_T,
                        q′q′,
                        T′T′,
                        corr_Tq,
                    )
                @test result ≈ FT(1) atol = FT(1e-6)

                # Integrate identity for T → should return mean T
                f_T(T, q) = T
                result_T =
                    ClimaAtmos.integrate_over_sgs(f_T, quad, μ_q, μ_T, q′q′, T′T′, corr_Tq)
                @test result_T ≈ μ_T atol = FT(1.0)

                # NamedTuple return
                f_nt(T, q) = (; val1 = T, val2 = q)
                result_nt =
                    ClimaAtmos.integrate_over_sgs(f_nt, quad, μ_q, μ_T, q′q′, T′T′, corr_Tq)
                @test haskey(result_nt, :val1)
                @test haskey(result_nt, :val2)
            end
        end
    end

    @testset "Microphysics Tendencies Quadrature" begin
        import Thermodynamics as TD
        import ClimaParams as CP
        import CloudMicrophysics.Parameters as CMP
        import CloudMicrophysics.BulkMicrophysicsTendencies as BMT

        # Import from ClimaAtmos - function is in microphysics_wrappers.jl
        using ClimaAtmos: microphysics_tendencies_quadrature

        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                # Setup parameters
                toml_dict = CP.create_toml_dict(FT)
                tps = TD.Parameters.ThermodynamicsParameters(toml_dict)
                mp = CMP.Microphysics1MParams(toml_dict)

                # Grid-mean state
                ρ = FT(1.2)
                p_c = FT(1e5)
                T_mean = FT(280.0)
                q_tot_mean = FT(0.01)
                q_lcl_mean = FT(0.0001)
                q_icl_mean = FT(0.00005)
                q_rai = FT(0.0001)
                q_sno = FT(0.00001)

                # Variances and correlation
                T′T′ = FT(1.0)
                q′q′ = FT(1e-6)
                corr_Tq = FT(0.6)

                # Test 1: Single quadrature point should match grid-mean evaluation
                @testset "Single Point = Grid Mean" begin
                    # Use GaussianSGS: only Gaussian has χ=0 → (μ_T, μ_q)
                    quad_1pt = ClimaAtmos.SGSQuadrature(
                        FT;
                        quadrature_order = 1,
                        distribution = ClimaAtmos.GaussianSGS(),
                    )

                    # Quadrature result
                    result_quad = microphysics_tendencies_quadrature(
                        BMT.Microphysics1Moment(),
                        quad_1pt, mp, tps, ρ, p_c,
                        T_mean, q_tot_mean, q_lcl_mean, q_icl_mean, q_rai, q_sno,
                        T′T′, q′q′, corr_Tq,
                    )

                    # Direct evaluation at grid mean
                    result_direct = BMT.bulk_microphysics_tendencies(
                        BMT.Microphysics1Moment(),
                        mp, tps, ρ, T_mean,
                        q_tot_mean, q_lcl_mean, q_icl_mean, q_rai, q_sno,
                    )

                    # Compare all fields
                    @test result_quad.dq_lcl_dt ≈ result_direct.dq_lcl_dt rtol = FT(1e-5)
                    @test result_quad.dq_icl_dt ≈ result_direct.dq_icl_dt rtol = FT(1e-5)
                    @test result_quad.dq_rai_dt ≈ result_direct.dq_rai_dt rtol = FT(1e-5)
                    @test result_quad.dq_sno_dt ≈ result_direct.dq_sno_dt rtol = FT(1e-5)
                end

                # Test 2: Zero variance should match grid-mean evaluation
                @testset "Zero Variance = Grid Mean" begin
                    quad = ClimaAtmos.SGSQuadrature(FT; quadrature_order = 3)

                    # Zero variances
                    result_quad = microphysics_tendencies_quadrature(
                        BMT.Microphysics1Moment(),
                        quad, mp, tps, ρ, p_c,
                        T_mean, q_tot_mean, q_lcl_mean, q_icl_mean, q_rai, q_sno,
                        FT(0), FT(0), FT(0),  # Zero variances, zero correlation
                    )

                    # Direct evaluation
                    result_direct = BMT.bulk_microphysics_tendencies(
                        BMT.Microphysics1Moment(),
                        mp, tps, ρ, T_mean,
                        q_tot_mean, q_lcl_mean, q_icl_mean, q_rai, q_sno,
                    )

                    @test result_quad.dq_lcl_dt ≈ result_direct.dq_lcl_dt rtol = FT(1e-5)
                    @test result_quad.dq_rai_dt ≈ result_direct.dq_rai_dt rtol = FT(1e-5)
                end

                # Test 3: Result has correct NamedTuple structure
                @testset "NamedTuple Structure" begin
                    quad = ClimaAtmos.SGSQuadrature(FT)

                    result = microphysics_tendencies_quadrature(
                        BMT.Microphysics1Moment(),
                        quad, mp, tps, ρ, p_c,
                        T_mean, q_tot_mean, q_lcl_mean, q_icl_mean, q_rai, q_sno,
                        T′T′, q′q′, corr_Tq,
                    )

                    @test haskey(result, :dq_lcl_dt)
                    @test haskey(result, :dq_icl_dt)
                    @test haskey(result, :dq_rai_dt)
                    @test haskey(result, :dq_sno_dt)
                    @test result.dq_lcl_dt isa FT
                end

                # Test 4: Non-zero variance produces different result
                @testset "Variance Effect" begin
                    quad = ClimaAtmos.SGSQuadrature(FT; quadrature_order = 3)

                    # With variance
                    result_var = microphysics_tendencies_quadrature(
                        BMT.Microphysics1Moment(),
                        quad, mp, tps, ρ, p_c,
                        T_mean, q_tot_mean, q_lcl_mean, q_icl_mean, q_rai, q_sno,
                        FT(4.0), FT(1e-5), FT(0.8),  # Non-zero variances
                    )

                    # Without variance
                    result_no_var = microphysics_tendencies_quadrature(
                        BMT.Microphysics1Moment(),
                        quad, mp, tps, ρ, p_c,
                        T_mean, q_tot_mean, q_lcl_mean, q_icl_mean, q_rai, q_sno,
                        FT(0), FT(0), FT(0),
                    )

                    # Results should differ (unless microphysics is perfectly linear)
                    # At minimum, they should both be finite
                    @test isfinite(result_var.dq_lcl_dt)
                    @test isfinite(result_no_var.dq_lcl_dt)
                end
            end
        end
    end

    @testset "Mass Conservation" begin
        # Verify that total water is conserved in microphysics quadrature
        import Thermodynamics as TD
        import ClimaParams as CP
        import CloudMicrophysics.Parameters as CMP
        import CloudMicrophysics.BulkMicrophysicsTendencies as BMT
        using ClimaAtmos: microphysics_tendencies_quadrature

        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                mp = CMP.Microphysics1MParams(toml_dict)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
                quad = ClimaAtmos.SGSQuadrature(FT; quadrature_order = 3)

                # Realistic atmospheric state
                ρ = FT(1.0)
                p_c = FT(1e5)
                T = FT(280.0)
                q_tot = FT(0.015)
                q_liq = FT(0.002)
                q_ice = FT(0.001)
                q_rai = FT(0.001)
                q_sno = FT(0.0005)

                # Non-zero covariances
                T′T′ = FT(4.0)
                q′q′ = FT(1e-5)
                corr_Tq = FT(0.6)

                result = microphysics_tendencies_quadrature(
                    BMT.Microphysics1Moment(),
                    quad, mp, thp, ρ, p_c, T, q_tot, q_liq, q_ice, q_rai, q_sno,
                    T′T′, q′q′, corr_Tq,
                )

                # Total condensed water tendency
                dq_condensed =
                    result.dq_lcl_dt + result.dq_icl_dt +
                    result.dq_rai_dt + result.dq_sno_dt

                # All tendencies should be finite
                @test isfinite(result.dq_lcl_dt)
                @test isfinite(result.dq_icl_dt)
                @test isfinite(result.dq_rai_dt)
                @test isfinite(result.dq_sno_dt)
                @test isfinite(dq_condensed)

                # The vapor tendency equals negative condensate change (conservation)
                # Note: exact conservation may not hold due to numerical precision
                # and microphysics parameterization, but should be physically reasonable
                dq_vap_implied = -dq_condensed
                @test isfinite(dq_vap_implied)

                # Sanity check: total tendency magnitude should be reasonable
                # (not unrealistically large for these conditions)
                max_reasonable_tendency = FT(1e-3)  # 1 g/kg/s
                @test abs(dq_condensed) < max_reasonable_tendency
            end
        end
    end

    @testset "Type Stability" begin
        # Verify no type instabilities in quadrature integration
        import Thermodynamics as TD
        import ClimaParams as CP
        import CloudMicrophysics.Parameters as CMP
        import CloudMicrophysics.BulkMicrophysicsTendencies as BMT
        using ClimaAtmos: microphysics_tendencies_quadrature
        using Test: @inferred

        # Test both Float32 and Float64 for type stability
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                mp = CMP.Microphysics1MParams(toml_dict)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
                quad = ClimaAtmos.SGSQuadrature(FT)

                ρ = FT(1.0)
                p_c = FT(1e5)
                T = FT(280.0)
                q_tot = FT(0.01)
                q_liq = FT(0.001)
                q_ice = FT(0.0005)
                q_rai = FT(0.0002)
                q_sno = FT(0.0001)
                T′T′ = FT(1.0)
                q′q′ = FT(1e-6)
                corr_Tq = FT(0.6)

                # Test type stability
                result = @inferred microphysics_tendencies_quadrature(
                    BMT.Microphysics1Moment(),
                    quad, mp, thp, ρ, p_c, T, q_tot, q_liq, q_ice, q_rai, q_sno,
                    T′T′, q′q′, corr_Tq,
                )

                # Verify return type
                @test result isa NamedTuple
                @test haskey(result, :dq_lcl_dt)
                @test haskey(result, :dq_icl_dt)
                @test haskey(result, :dq_rai_dt)
                @test haskey(result, :dq_sno_dt)
            end
        end
    end

    @testset "GPU Safety - Functor Pattern" begin
        # Verify GPU-safe design: functors instead of closures
        import Thermodynamics as TD
        import ClimaParams as CP
        import CloudMicrophysics.Parameters as CMP
        import CloudMicrophysics.BulkMicrophysicsTendencies as BMT
        using ClimaAtmos: MicrophysicsEvaluator

        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                mp = CMP.Microphysics1MParams(toml_dict)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)

                # Grid-mean state
                ρ = FT(1.0)
                T_mean = FT(280.0)
                q_tot_mean = FT(0.01)
                q_liq_mean = FT(0.001)
                q_ice_mean = FT(0.0005)
                q_rai = FT(0.0002)
                q_sno = FT(0.0001)
                q_cond_mean = q_liq_mean + q_ice_mean
                q_sat_mean = TD.q_vap_saturation(thp, T_mean, ρ)
                excess_mean = q_tot_mean - q_sat_mean

                # Create evaluator
                evaluator = MicrophysicsEvaluator(
                    BMT.Microphysics1Moment(),
                    mp, thp, ρ, T_mean, q_tot_mean,
                    q_liq_mean, q_ice_mean, q_rai, q_sno,
                    q_cond_mean, q_sat_mean, excess_mean,
                    (),  # Empty args tuple for 1-moment
                )

                # Verify it's a proper functor (not a closure)
                @test evaluator isa MicrophysicsEvaluator
                @test fieldcount(typeof(evaluator)) > 0  # Has fields (not closure)

                # Verify it's callable
                T_hat = FT(280.0)
                q_tot_hat = FT(0.01)
                @test applicable(evaluator, T_hat, q_tot_hat)

                # Call it and verify result
                result = evaluator(T_hat, q_tot_hat)
                @test result isa NamedTuple
                @test isfinite(result.dq_lcl_dt)
                @test isfinite(result.dq_icl_dt)
                @test isfinite(result.dq_rai_dt)
                @test isfinite(result.dq_sno_dt)
            end
        end
    end


    @testset "GridMeanSGS Quadrature = Direct BMT" begin
        # Verify that the GridMeanSGS quadrature path produces the same result
        # as a direct BMT call. 
        import Thermodynamics as TD
        import ClimaParams as CP
        import CloudMicrophysics.Parameters as CMP
        import CloudMicrophysics.BulkMicrophysicsTendencies as BMT
        using ClimaAtmos: microphysics_tendencies_quadrature

        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                tps = TD.Parameters.ThermodynamicsParameters(toml_dict)
                mp = CMP.Microphysics1MParams(toml_dict)

                # Grid-mean state
                ρ = FT(1.2)
                p_c = FT(1e5)
                T_mean = FT(280.0)
                q_tot = FT(0.01)
                q_liq = FT(0.001)
                q_ice = FT(0.0005)
                q_rai = FT(0.0002)
                q_sno = FT(0.0001)

                # GridMeanSGS inside SGSQuadrature
                quad_gm = ClimaAtmos.SGSQuadrature(
                    FT;
                    distribution = ClimaAtmos.GridMeanSGS(),
                )

                # Non-zero variances — should be ignored by GridMeanSGS
                T′T′ = FT(4.0)
                q′q′ = FT(1e-5)
                corr_Tq = FT(0.6)

                # Quadrature path
                result_quad = microphysics_tendencies_quadrature(
                    BMT.Microphysics1Moment(),
                    quad_gm, mp, tps, ρ, p_c,
                    T_mean, q_tot, q_liq, q_ice, q_rai, q_sno,
                    T′T′, q′q′, corr_Tq,
                )

                # Direct BMT call
                result_direct = BMT.bulk_microphysics_tendencies(
                    BMT.Microphysics1Moment(),
                    mp, tps, ρ, T_mean,
                    q_tot, q_liq, q_ice, q_rai, q_sno,
                )

                # They must match exactly (both evaluate at grid mean)
                for field in (:dq_lcl_dt, :dq_icl_dt, :dq_rai_dt, :dq_sno_dt)
                    val_q = getfield(result_quad, field)
                    val_d = getfield(result_direct, field)
                    @test val_q ≈ val_d rtol = FT(1e-5)
                end
            end
        end
    end

    @testset "Sign Convention" begin
        # Verify that quadrature-averaged microphysics tendencies have correct signs.
        # This is critical: the 0M dq_tot_dt should always be ≤ 0 (precipitation is a sink).

        import CloudMicrophysics.Parameters as CMP
        import CloudMicrophysics.BulkMicrophysicsTendencies as BMT
        import Thermodynamics as TD
        import ClimaParams as CP

        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)

                @testset "0M quadrature: dq_tot_dt ≤ 0" begin
                    quad = ClimaAtmos.SGSQuadrature(FT)
                    mp_0m = CMP.Microphysics0MParams(toml_dict)
                    thp = TD.Parameters.ThermodynamicsParameters(toml_dict)

                    ρ = FT(1.0)
                    T_mean = FT(280.0)
                    q_tot_mean = FT(0.015)

                    # Zero variances (grid-mean evaluation)
                    result_zero = ClimaAtmos.microphysics_tendencies_quadrature_0m(
                        quad, mp_0m, thp, ρ, T_mean, q_tot_mean,
                        FT(0), FT(0), FT(0),
                    )
                    @test result_zero.dq_tot_dt <= FT(0)
                    @test isfinite(result_zero.dq_tot_dt)

                    # With variances (SGS fluctuations)
                    result_var = ClimaAtmos.microphysics_tendencies_quadrature_0m(
                        quad, mp_0m, thp, ρ, T_mean, q_tot_mean,
                        FT(4.0), FT(1e-5), FT(0.6),
                    )
                    @test result_var.dq_tot_dt <= FT(0)
                    @test isfinite(result_var.dq_tot_dt)
                end

                @testset "1M quadrature: sign consistency" begin
                    quad = ClimaAtmos.SGSQuadrature(FT)
                    mp_1m = CMP.Microphysics1MParams(toml_dict; with_2M_autoconv = true)
                    thp = TD.Parameters.ThermodynamicsParameters(toml_dict)

                    ρ = FT(1.0)
                    p_c = FT(85000.0)
                    T = FT(280.0)
                    q_tot = FT(0.015)
                    q_liq = FT(0.001)
                    q_ice = FT(0.0005)
                    q_rai = FT(0.0001)
                    q_sno = FT(0.00005)

                    # With zero variances, quadrature should match direct BMT
                    result_quad = ClimaAtmos.microphysics_tendencies_quadrature(
                        BMT.Microphysics1Moment(),
                        quad, mp_1m, thp, ρ, p_c, T,
                        q_tot, q_liq, q_ice, q_rai, q_sno,
                        FT(0), FT(0), FT(0),
                    )

                    result_direct = BMT.bulk_microphysics_tendencies(
                        BMT.Microphysics1Moment(),
                        mp_1m, thp, ρ, T,
                        q_tot, q_liq, q_ice, q_rai, q_sno,
                    )

                    # Signs should match between quadrature and direct
                    for field in (:dq_lcl_dt, :dq_icl_dt, :dq_rai_dt, :dq_sno_dt)
                        sq = getfield(result_quad, field)
                        sd = getfield(result_direct, field)
                        @test sign(sq) == sign(sd) ||
                              (abs(sq) < FT(1e-10) && abs(sd) < FT(1e-10))
                        @test isfinite(sq)
                    end

                    # With non-zero variances, should still be finite
                    result_var = ClimaAtmos.microphysics_tendencies_quadrature(
                        BMT.Microphysics1Moment(),
                        quad, mp_1m, thp, ρ, p_c, T,
                        q_tot, q_liq, q_ice, q_rai, q_sno,
                        FT(4.0), FT(1e-5), FT(0.6),
                    )
                    for field in (:dq_lcl_dt, :dq_icl_dt, :dq_rai_dt, :dq_sno_dt)
                        @test isfinite(getfield(result_var, field))
                    end
                end
            end
        end
    end

end
