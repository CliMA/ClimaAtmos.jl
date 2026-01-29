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
                # Test order 3 on [-1,1]
                a, w = ClimaAtmos.gauss_legendre(FT, 3)
                @test length(a) == 3
                @test a[2] ≈ FT(0) atol = eps(FT)
                @test sum(w) ≈ FT(2) atol = FT(1e-10)  # ∫₋₁¹ dx = 2

                # Test order 3 on [0,1]
                a01, w01 = ClimaAtmos.gauss_legendre_01(FT, 3)
                @test all(0 .<= a01 .<= 1)
                @test sum(w01) ≈ FT(1) atol = FT(1e-10)  # ∫₀¹ dx = 1
            end
        end
    end

    @testset "SGSQuadrature Struct" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                # Default (Gaussian)
                quad = ClimaAtmos.SGSQuadrature(FT)
                @test ClimaAtmos.quadrature_order(quad) == 3
                @test quad.dist isa ClimaAtmos.GaussianSGS

                # Custom order and distribution
                quad5 = ClimaAtmos.SGSQuadrature(FT; quadrature_order = 5, distribution = ClimaAtmos.LogNormalSGS())
                @test ClimaAtmos.quadrature_order(quad5) == 5
                @test quad5.dist isa ClimaAtmos.LogNormalSGS

                # BetaSGS should have points on [0,1]
                quad_beta = ClimaAtmos.SGSQuadrature(FT; distribution = ClimaAtmos.BetaSGS())
                @test all(0 .<= quad_beta.a .<= 1)
            end
        end
    end

    @testset "Covariance Limiting" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                quad = ClimaAtmos.SGSQuadrature(FT)

                # Standard case
                σ_q, σ_T, corr = ClimaAtmos.limit_covariances(
                    FT(1e-6), FT(1.0), FT(1e-3), FT(0.01), quad
                )
                @test σ_q >= 0
                @test σ_T >= 0
                @test -1 <= corr <= 1

                # Zero variance
                σ_q0, σ_T0, corr0 = ClimaAtmos.limit_covariances(
                    FT(0), FT(0), FT(0), FT(0.01), quad
                )
                @test σ_q0 == 0
                @test σ_T0 == 0
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

                # GaussianSGS at χ = 0 should return means
                T_hat, q_hat = ClimaAtmos.get_physical_point(
                    ClimaAtmos.GaussianSGS(), FT(0), FT(0), μ_q, μ_T, σ_q, σ_T, corr
                )
                @test T_hat ≈ μ_T atol = FT(0.1)
                @test q_hat ≈ μ_q atol = FT(0.001)

                # LogNormalSGS
                T_hat_ln, q_hat_ln = ClimaAtmos.get_physical_point(
                    ClimaAtmos.LogNormalSGS(), FT(0), FT(0), μ_q, μ_T, σ_q, σ_T, corr
                )
                @test T_hat_ln ≈ μ_T atol = FT(0.1)
                @test q_hat_ln > 0  # log-normal is always positive

                # BetaSGS at χ1=0.5 should give q near mean
                T_hat_b, q_hat_b = ClimaAtmos.get_physical_point(
                    ClimaAtmos.BetaSGS(), FT(0.5), FT(0), μ_q, μ_T, σ_q, σ_T, corr
                )
                @test q_hat_b ≈ μ_q atol = σ_q
            end
        end
    end

    @testset "Recursive Operations" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                # Scalars
                @test ClimaAtmos.recursive_zero(FT(1.5)) == FT(0)
                @test ClimaAtmos.recursive_add(FT(1), FT(2)) == FT(3)
                @test ClimaAtmos.recursive_mul(FT(2), FT(3)) == FT(6)

                # NamedTuples
                nt = (; a = FT(1), b = FT(2))
                @test ClimaAtmos.recursive_zero(nt) == (; a = FT(0), b = FT(0))
                @test ClimaAtmos.recursive_add(nt, nt) == (; a = FT(2), b = FT(4))
                @test ClimaAtmos.recursive_mul(nt, FT(3)) == (; a = FT(3), b = FT(6))
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
                T′T′, q′q′, T′q′ = FT(1.0), FT(1e-6), FT(0.0)

                result = ClimaAtmos.integrate_over_sgs(f_const, quad, μ_q, μ_T, q′q′, T′T′, T′q′)
                @test result ≈ FT(1) atol = FT(1e-6)

                # Integrate identity for T → should return mean T
                f_T(T, q) = T
                result_T = ClimaAtmos.integrate_over_sgs(f_T, quad, μ_q, μ_T, q′q′, T′T′, T′q′)
                @test result_T ≈ μ_T atol = FT(1.0)

                # NamedTuple return
                f_nt(T, q) = (; val1 = T, val2 = q)
                result_nt = ClimaAtmos.integrate_over_sgs(f_nt, quad, μ_q, μ_T, q′q′, T′T′, T′q′)
                @test haskey(result_nt, :val1)
                @test haskey(result_nt, :val2)
            end
        end
    end

end
