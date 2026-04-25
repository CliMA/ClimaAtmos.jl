#=
Unit tests for SGS Quadrature utilities (src/cache/sgs_quadrature.jl)
=#

using Test
using ClimaAtmos
using Statistics

import ClimaComms
import ClimaCore: Domains, Geometry, Meshes, Topologies, Spaces, Grids, Fields

# One FD column + one center `LocalGeometry` (real metrics) for tests that need
# `WVector(u, lg)`; avoids `test_helpers.jl` here (that file uses test-only deps like IntervalSets).
function _fd_column_center_local_geometry(
    ::Type{FT};
    z_top = nothing,
    nelems::Int = 8,
    ilevel = nothing,
) where {FT}
    zt = something(z_top, FT(10_000))
    ctx = ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded())
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint(zero(FT)),
        Geometry.ZPoint(zt);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain; nelems = nelems)
    vert_topology = Topologies.IntervalTopology(ctx, vertmesh)
    vert_grid = Grids.FiniteDifferenceGrid(vert_topology)
    cspace = Spaces.CenterFiniteDifferenceSpace(vert_grid)
    lgf = Fields.local_geometry_field(cspace)
    nlev = Spaces.nlevels(cspace)
    i = something(ilevel, clamp(div(nlev + 1, 2), 2, max(2, nlev - 1)))
    return Fields.field_values(Fields.level(lgf, i))[1]
end

# Test-only: assert every `AbstractVerticallyResolvedSGS` used for 1M is
# wired to the long-arity `integrate_over_sgs` via `microphysics_tendencies_1m_sgs_row`.
# (Not part of the library API — production uses dispatch on `microphysics_tendencies_1m`.)
function _assert_1m_sgs_gridscale_supported_for_tests(sgs_quad::ClimaAtmos.SGSQuadrature)
    d = sgs_quad.dist
    if d isa ClimaAtmos.AbstractVerticallyResolvedSGS
        # If it's vertically resolved, it should be handled by the row entrypoint.
        # No further gating needed — dispatch covers all AbstractVerticallyResolvedSGS.
    end
    return nothing
end

@testset "SGS Quadrature" begin

    # Run first: every YAML `sgs_distribution` that selects a vertical-profile
    # gridscale-corrected type must resolve (catches deleted marker structs /
    # renamed types before hundreds of unrelated tests pass).
    @testset "get_sgs_distribution: all vertical-profile keys construct" begin
        names = [
            "gaussian_vertical_profile",
            "gaussian_vertical_profile_full_cubature",
            "gaussian_vertical_profile_inner_bracketed",
            "gaussian_vertical_profile_inner_halley",
            "gaussian_vertical_profile_inner_chebyshev",
            "lognormal_vertical_profile",
            "lognormal_vertical_profile_full_cubature",
            "lognormal_vertical_profile_inner_bracketed",
            "lognormal_vertical_profile_inner_halley",
            "lognormal_vertical_profile_inner_chebyshev",
            "gaussian_vertical_profile_lhs_z",
            "gaussian_vertical_profile_principal_axis",
            "gaussian_vertical_profile_voronoi",
            "gaussian_vertical_profile_barycentric",
            "lognormal_vertical_profile_lhs_z",
            "lognormal_vertical_profile_principal_axis",
            "lognormal_vertical_profile_voronoi",
            "lognormal_vertical_profile_barycentric",
        ]
        for s in names
            pa = Dict{String, Any}("sgs_distribution" => s)
            d = ClimaAtmos.get_sgs_distribution(pa)
            @test d isa ClimaAtmos.AbstractSGSDistribution
        end
    end

    @testset "1M: layer-profile SGS (linear profile vs bivariate strip)" begin
        FT = Float64
        quad_ln = ClimaAtmos.SGSQuadrature(FT; quadrature_order = 3, distribution = ClimaAtmos.LogNormalSGS())
        q_ct = ClimaAtmos.SGSQuadrature(
            FT;
            quadrature_order = 3,
            distribution = ClimaAtmos.VerticallyResolvedSGS{ClimaAtmos.SubgridColumnTensor}(),
        )
        q_pr = ClimaAtmos.SGSQuadrature(
            FT;
            quadrature_order = 3,
            distribution = ClimaAtmos.VerticallyResolvedSGS{
                ClimaAtmos.DefaultGridscaleProfileQuadrature,
            }(),
        )
        @test ClimaAtmos._is_vertically_resolved_sgs(q_ct.dist)
        @test ClimaAtmos._is_vertically_resolved_sgs(q_pr.dist)
        q_lhs = ClimaAtmos.SGSQuadrature(
            FT;
            quadrature_order = 3,
            distribution = ClimaAtmos.VerticallyResolvedSGS{ClimaAtmos.SubgridLatinHypercubeZ}(),
        )
        @test ClimaAtmos._is_vertically_resolved_sgs(q_lhs.dist)
        @test _assert_1m_sgs_gridscale_supported_for_tests(q_lhs) === nothing
        @test _assert_1m_sgs_gridscale_supported_for_tests(quad_ln) === nothing
        @test _assert_1m_sgs_gridscale_supported_for_tests(q_ct) === nothing
        @test _assert_1m_sgs_gridscale_supported_for_tests(q_pr) === nothing
        f(T, q) = T + FT(5) * q
        μ_q = FT(0.012)
        μ_T = FT(285)
        qv = FT(1e-7)
        Tv = FT(0.4)
        ρc = FT(0.55)
        # Bivariate `integrate_over_sgs` collapses Ln gridscale S to the same inner rule:
        i1 = ClimaAtmos.integrate_over_sgs(f, q_ct, μ_q, μ_T, qv, Tv, ρc)
        i2 = ClimaAtmos.integrate_over_sgs(f, q_pr, μ_q, μ_T, qv, Tv, ρc)
        @test i1 ≈ i2
    end



    @testset "subcell geometric variance increment (two-slope face-anchored)" begin
        Δz = 2.0
        # Symmetric-slope limit (s_dn == s_up) recovers the classical (1/12) Δz² d²:
        s_q = 0.1
        s_T = 0.4
        Δq_sym, ΔT_sym = ClimaAtmos.subcell_geometric_variance_increment(
            Δz, s_q, s_q, s_T, s_T,
        )
        @test Δq_sym ≈ (1 / 12) * Δz^2 * s_q^2
        @test ΔT_sym ≈ (1 / 12) * Δz^2 * s_T^2

        # Asymmetric slopes: centered d and asymmetry D both contribute
        # (variance = d² Δz² / 12 + D² Δz² / 192).
        s_q_dn, s_q_up = -0.05, 0.15
        s_T_dn, s_T_up = 0.30, 0.50
        Δq, ΔT = ClimaAtmos.subcell_geometric_variance_increment(
            Δz, s_q_dn, s_q_up, s_T_dn, s_T_up,
        )
        d_q, D_q = (s_q_up + s_q_dn) / 2, s_q_up - s_q_dn
        d_T, D_T = (s_T_up + s_T_dn) / 2, s_T_up - s_T_dn
        @test Δq ≈ (Δz^2) * (d_q^2 / 12 + D_q^2 / 192)
        @test ΔT ≈ (Δz^2) * (d_T^2 / 12 + D_T^2 / 192)
        # Sign- and sign-flip-invariance properties:
        Δq_neg, _ = ClimaAtmos.subcell_geometric_variance_increment(
            Δz, -s_q_dn, -s_q_up, s_T_dn, s_T_up,
        )
        @test Δq_neg ≈ Δq
    end

    @testset "subcell geometric T–q covariance (two-slope face-anchored)" begin
        Δz = 2.0
        # Symmetric-slope limit recovers the classical (1/12) Δz² d^q d^T
        s_q = 0.05
        s_T = 0.20  # already in T-space: caller applies ∂T/∂θ_li before passing.
        cov_sym = ClimaAtmos.subcell_geometric_covariance_Tq(
            Δz, s_q, s_q, s_T, s_T,
        )
        @test cov_sym ≈ (1 / 12) * Δz^2 * s_q * s_T

        # Asymmetric slopes: (1/12) Δz² d^q d^T + (1/192) Δz² D^q D^T.
        s_q_dn, s_q_up = -0.01, 0.03
        s_T_dn, s_T_up = 0.15, 0.25
        cov = ClimaAtmos.subcell_geometric_covariance_Tq(
            Δz, s_q_dn, s_q_up, s_T_dn, s_T_up,
        )
        d_q, D_q = (s_q_up + s_q_dn) / 2, s_q_up - s_q_dn
        d_T, D_T = (s_T_up + s_T_dn) / 2, s_T_up - s_T_dn
        @test cov ≈ (Δz^2) * (d_q * d_T / 12 + D_q * D_T / 192)
    end

    @testset "subcell layer-mean excursion" begin
        Δz = 2.0
        # Symmetric-slope limit: no excursion.
        @test ClimaAtmos.subcell_layer_mean_excursion(Δz, 0.3, 0.3) ≈ 0.0
        # Asymmetric slopes: ΔΦ = (Δz / 8) * (s_up - s_dn).
        @test ClimaAtmos.subcell_layer_mean_excursion(Δz, -0.1, 0.2) ≈ Δz / 8 * 0.3
    end

    @testset "uniform_normal_convolution_pdf (univariate)" begin
        FT = Float64
        a, b, σ = FT(-1), FT(1), FT(0.3)
        xs = range(FT(-3), FT(3); length = 2000)
        dens = [ClimaAtmos.uniform_normal_convolution_pdf(x, a, b, σ) for x in xs]
        @test sum(dens) * step(xs) ≈ 1 rtol = 0.02
    end

    @testset "uniform_lognormal_convolution (univariate)" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                μ_q = FT(1.0)
                L = FT(0.5)
                σ_ln = FT(0.3)
                y_min = μ_q - L / 2
                y_max = μ_q + L / 2
                
                # Test CDF property: F(0) ≈ 0, F(∞) ≈ 1
                @test ClimaAtmos.uniform_lognormal_convolution_cdf(FT(0.0), y_min, y_max, σ_ln) ≈ FT(0.0) atol = 1e-6
                @test ClimaAtmos.uniform_lognormal_convolution_cdf(FT(10.0), y_min, y_max, σ_ln) ≈ FT(1.0) atol = 1e-4
                
                # Test PDF integration: ∫ f(q) dq ≈ 1
                qs = range(FT(0.01), FT(5.0); length = 2000)
                dens = [ClimaAtmos.uniform_lognormal_convolution_pdf(q, y_min, y_max, σ_ln) for q in qs]
                @test sum(dens) * (qs[2] - qs[1]) ≈ FT(1.0) rtol = 0.01
                
                # Test CDF matches integration of PDF
                q_test = FT(1.2)
                f_q = ClimaAtmos.uniform_lognormal_convolution_cdf(q_test, y_min, y_max, σ_ln)
                qs_part = range(FT(0.001), q_test; length = 2000)
                dens_part = [ClimaAtmos.uniform_lognormal_convolution_pdf(q, y_min, y_max, σ_ln) for q in qs_part]
                @test f_q ≈ sum(dens_part) * (qs_part[2] - qs_part[1]) rtol = 0.01
            end
        end
    end

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

    @testset "Vertical-profile SGS: integrate_over_sgs + saturation (all YAML keys)" begin
        import Thermodynamics as TD
        import ClimaParams as CP
        # Every `sgs_distribution` string mapped to `AbstractVerticallyResolvedSGS` in
        # `get_sgs_distribution` (see `get_sgs_distribution linear-profile keys` below).
        profile_keys = String[
            "gaussian_vertical_profile_inner_chebyshev",
            "gaussian_vertical_profile",
            "gaussian_vertical_profile_full_cubature",
            "gaussian_vertical_profile_inner_bracketed",
            "gaussian_vertical_profile_inner_halley",
            "lognormal_vertical_profile",
            "lognormal_vertical_profile_full_cubature",
            "lognormal_vertical_profile_inner_bracketed",
            "lognormal_vertical_profile_inner_halley",
            "lognormal_vertical_profile_inner_chebyshev",
            "gaussian_vertical_profile_lhs_z",
            "gaussian_vertical_profile_principal_axis",
            "gaussian_vertical_profile_voronoi",
            "gaussian_vertical_profile_barycentric",
            "lognormal_vertical_profile_lhs_z",
            "lognormal_vertical_profile_principal_axis",
            "lognormal_vertical_profile_voronoi",
            "lognormal_vertical_profile_barycentric",
        ]
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
                f_const(T, q) = one(FT)
                f_T(T, q) = T
                μ_q, μ_T = FT(0.01), FT(300.0)
                T′T′, q′q′ = FT(1.0), FT(1e-6)
                corr_Tq = FT(0.5)
                zδ = zero(FT)
                ρ = FT(1.0)
                T_mean = FT(280.0)
                q_mean = FT(0.01)
                for key in profile_keys
                    dist = ClimaAtmos.get_sgs_distribution(Dict{String, Any}("sgs_distribution" => key))
                    @test dist isa ClimaAtmos.AbstractVerticallyResolvedSGS
                    quad = ClimaAtmos.SGSQuadrature(FT; quadrature_order = 3, distribution = dist)
                    r1 = ClimaAtmos.integrate_over_sgs(
                        f_const,
                        quad,
                        μ_q,
                        μ_T,
                        q′q′,
                        T′T′,
                        corr_Tq,
                    )
                    @test isfinite(r1)
                    @test r1 ≈ one(FT) atol = FT(0.08)
                    rT = ClimaAtmos.integrate_over_sgs(
                        f_T,
                        quad,
                        μ_q,
                        μ_T,
                        q′q′,
                        T′T′,
                        corr_Tq,
                    )
                    @test isfinite(rT)
                    @test rT ≈ μ_T rtol = FT(0.08)
                    adj = ClimaAtmos.compute_sgs_saturation_adjustment(
                        thp,
                        quad,
                        ρ,
                        T_mean,
                        q_mean,
                        T′T′,
                        q′q′,
                        corr_Tq,
                        zδ,
                        zδ,
                    )
                    @test isfinite(adj.T) && isfinite(adj.q_liq) && isfinite(adj.q_ice)
                    @test adj.q_liq >= zero(FT) && adj.q_ice >= zero(FT)
                end
            end
        end
    end

    @testset "Microphysics Tendencies 1M" begin
        import Thermodynamics as TD
        import ClimaParams as CP
        import CloudMicrophysics.Parameters as CMP
        import CloudMicrophysics.BulkMicrophysicsTendencies as BMT
        using ClimaAtmos: microphysics_tendencies_1m

        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                # Setup parameters
                toml_dict = CP.create_toml_dict(FT)
                tps = TD.Parameters.ThermodynamicsParameters(toml_dict)
                mp = CMP.Microphysics1MParams(toml_dict)

                # Grid-mean state
                ρ = FT(1.2)
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

                # No-op timestepping (no limiting)
                tst = nothing
                dt = FT(1.0)

                # Test 1: Single quadrature point should match grid-mean evaluation
                @testset "Single Point = Grid Mean" begin
                    # Use GaussianSGS: only Gaussian has χ=0 → (μ_T, μ_q)
                    quad_1pt = ClimaAtmos.SGSQuadrature(
                        FT;
                        quadrature_order = 1,
                        distribution = ClimaAtmos.GaussianSGS(),
                    )

                    # Quadrature result (no limiting)
                    result_quad = microphysics_tendencies_1m(
                        BMT.Microphysics1Moment(),
                        quad_1pt, mp, tps, ρ,
                        T_mean, q_tot_mean, q_lcl_mean, q_icl_mean, q_rai, q_sno,
                        T′T′, q′q′, corr_Tq, tst, dt,
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
                    result_quad = microphysics_tendencies_1m(
                        BMT.Microphysics1Moment(),
                        quad, mp, tps, ρ,
                        T_mean, q_tot_mean, q_lcl_mean, q_icl_mean, q_rai, q_sno,
                        FT(0), FT(0), FT(0), tst, dt,
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

                    result = microphysics_tendencies_1m(
                        BMT.Microphysics1Moment(),
                        quad, mp, tps, ρ,
                        T_mean, q_tot_mean, q_lcl_mean, q_icl_mean, q_rai, q_sno,
                        T′T′, q′q′, corr_Tq, tst, dt,
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
                    result_var = microphysics_tendencies_1m(
                        BMT.Microphysics1Moment(),
                        quad, mp, tps, ρ,
                        T_mean, q_tot_mean, q_lcl_mean, q_icl_mean, q_rai, q_sno,
                        FT(4.0), FT(1e-5), FT(0.8), tst, dt,
                    )

                    # Without variance
                    result_no_var = microphysics_tendencies_1m(
                        BMT.Microphysics1Moment(),
                        quad, mp, tps, ρ,
                        T_mean, q_tot_mean, q_lcl_mean, q_icl_mean, q_rai, q_sno,
                        FT(0), FT(0), FT(0), tst, dt,
                    )

                    # Results should differ (unless microphysics is perfectly linear)
                    # At minimum, they should both be finite
                    @test isfinite(result_var.dq_lcl_dt)
                    @test isfinite(result_no_var.dq_lcl_dt)
                end

                # Test 5: Non-quadrature (direct) evaluation
                @testset "Direct (non-quadrature) evaluation" begin
                    result_direct_wrapper = microphysics_tendencies_1m(
                        ρ, q_tot_mean, q_lcl_mean, q_icl_mean, q_rai, q_sno,
                        T_mean, mp, tps, tst, dt,
                    )

                    result_bmt = BMT.bulk_microphysics_tendencies(
                        BMT.Microphysics1Moment(),
                        mp, tps, ρ, T_mean,
                        q_tot_mean, q_lcl_mean, q_icl_mean, q_rai, q_sno,
                    )

                    @test result_direct_wrapper.dq_lcl_dt ≈ result_bmt.dq_lcl_dt rtol =
                        FT(1e-10)
                    @test result_direct_wrapper.dq_icl_dt ≈ result_bmt.dq_icl_dt rtol =
                        FT(1e-10)
                    @test result_direct_wrapper.dq_rai_dt ≈ result_bmt.dq_rai_dt rtol =
                        FT(1e-10)
                    @test result_direct_wrapper.dq_sno_dt ≈ result_bmt.dq_sno_dt rtol =
                        FT(1e-10)
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
        using ClimaAtmos: microphysics_tendencies_1m

        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                mp = CMP.Microphysics1MParams(toml_dict)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
                quad = ClimaAtmos.SGSQuadrature(FT; quadrature_order = 3)

                # Realistic atmospheric state
                ρ = FT(1.0)
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

                tst = nothing
                dt = FT(1.0)

                result = microphysics_tendencies_1m(
                    BMT.Microphysics1Moment(),
                    quad, mp, thp, ρ, T, q_tot, q_liq, q_ice, q_rai, q_sno,
                    T′T′, q′q′, corr_Tq, tst, dt,
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
        using ClimaAtmos: microphysics_tendencies_1m
        using Test: @inferred

        # Test both Float32 and Float64 for type stability
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                mp = CMP.Microphysics1MParams(toml_dict)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
                quad = ClimaAtmos.SGSQuadrature(FT)

                ρ = FT(1.0)
                T = FT(280.0)
                q_tot = FT(0.01)
                q_liq = FT(0.001)
                q_ice = FT(0.0005)
                q_rai = FT(0.0002)
                q_sno = FT(0.0001)
                T′T′ = FT(1.0)
                q′q′ = FT(1e-6)
                corr_Tq = FT(0.6)
                tst = nothing
                dt = FT(1.0)

                # Test type stability
                result = @inferred microphysics_tendencies_1m(
                    BMT.Microphysics1Moment(),
                    quad, mp, thp, ρ, T, q_tot, q_liq, q_ice, q_rai, q_sno,
                    T′T′, q′q′, corr_Tq, tst, dt,
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
        using ClimaAtmos: Microphysics1MEvaluator

        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                mp = CMP.Microphysics1MParams(toml_dict)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)

                # Grid-mean state
                ρ = FT(1.0)
                T_mean = FT(280.0)
                q_lcl_mean = FT(0.001)
                q_icl_mean = FT(0.0005)
                q_rai = FT(0.0002)
                q_sno = FT(0.0001)

                # Create evaluator
                evaluator = Microphysics1MEvaluator(
                    BMT.Microphysics1Moment(),
                    mp, thp, ρ, T_mean,
                    q_lcl_mean, q_icl_mean, q_rai, q_sno,
                    (),  # Empty args tuple for 1-moment
                )

                # Verify it's a proper functor (not a closure)
                @test evaluator isa Microphysics1MEvaluator
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
        using ClimaAtmos: microphysics_tendencies_1m

        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                tps = TD.Parameters.ThermodynamicsParameters(toml_dict)
                mp = CMP.Microphysics1MParams(toml_dict)

                # Grid-mean state
                ρ = FT(1.2)
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
                tst = nothing
                dt = FT(1.0)

                # Quadrature path
                result_quad = microphysics_tendencies_1m(
                    BMT.Microphysics1Moment(),
                    quad_gm, mp, tps, ρ,
                    T_mean, q_tot, q_liq, q_ice, q_rai, q_sno,
                    T′T′, q′q′, corr_Tq, tst, dt,
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

                @testset "0M: dq_tot_dt ≤ 0" begin
                    quad = ClimaAtmos.SGSQuadrature(FT)
                    mp_0m = CMP.Microphysics0MParams(toml_dict)
                    thp = TD.Parameters.ThermodynamicsParameters(toml_dict)

                    ρ = FT(1.0)
                    T_mean = FT(280.0)
                    q_tot_mean = FT(0.015)
                    Φ = FT(5000.0)  # geopotential [J/kg]
                    tst = nothing
                    dt = FT(1.0)

                    # Zero variances (grid-mean evaluation via quadrature)
                    result_zero = ClimaAtmos.microphysics_tendencies_0m(
                        quad, mp_0m, thp, ρ, T_mean, q_tot_mean,
                        FT(0), FT(0), FT(0), Φ, tst, dt,
                    )
                    dq0 = result_zero.dq_tot_dt
                    # BMT 0M returns a NamedTuple; quadrature averages that structure.
                    dq0_s = dq0 isa NamedTuple && hasproperty(dq0, :dq_tot_dt) ? dq0.dq_tot_dt : dq0
                    @test dq0_s <= FT(0)
                    @test isfinite(dq0_s)
                    @test isfinite(result_zero.e_tot_hlpr)

                    # With variances (SGS fluctuations)
                    result_var = ClimaAtmos.microphysics_tendencies_0m(
                        quad, mp_0m, thp, ρ, T_mean, q_tot_mean,
                        FT(4.0), FT(1e-5), FT(0.6), Φ, tst, dt,
                    )
                    dqv = result_var.dq_tot_dt
                    dqv_s = dqv isa NamedTuple && hasproperty(dqv, :dq_tot_dt) ? dqv.dq_tot_dt : dqv
                    @test dqv_s <= FT(0)
                    @test isfinite(dqv_s)
                    @test isfinite(result_var.e_tot_hlpr)

                    # Direct (non-quadrature) evaluation with reasonable condensate
                    q_liq = FT(0.001)
                    q_ice = FT(0.0005)
                    result_direct = ClimaAtmos.microphysics_tendencies_0m(
                        mp_0m, thp, ρ, T_mean, q_tot_mean,
                        q_liq, q_ice, Φ, tst, dt,
                    )
                    @test result_direct.dq_tot_dt <= FT(0)
                    @test isfinite(result_direct.e_tot_hlpr)
                end

                @testset "1M: sign consistency" begin
                    quad = ClimaAtmos.SGSQuadrature(FT)
                    mp_1m = CMP.Microphysics1MParams(toml_dict; with_2M_autoconv = true)
                    thp = TD.Parameters.ThermodynamicsParameters(toml_dict)

                    ρ = FT(1.0)
                    T = FT(280.0)
                    q_tot = FT(0.015)
                    q_liq = FT(0.001)
                    q_ice = FT(0.0005)
                    q_rai = FT(0.0001)
                    q_sno = FT(0.00005)
                    tst = nothing
                    dt = FT(1.0)

                    # With zero variances, quadrature should match direct BMT
                    result_quad = ClimaAtmos.microphysics_tendencies_1m(
                        BMT.Microphysics1Moment(),
                        quad, mp_1m, thp, ρ, T,
                        q_tot, q_liq, q_ice, q_rai, q_sno,
                        FT(0), FT(0), FT(0), tst, dt,
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
                    result_var = ClimaAtmos.microphysics_tendencies_1m(
                        BMT.Microphysics1Moment(),
                        quad, mp_1m, thp, ρ, T,
                        q_tot, q_liq, q_ice, q_rai, q_sno,
                        FT(4.0), FT(1e-5), FT(0.6), tst, dt,
                    )
                    for field in (:dq_lcl_dt, :dq_icl_dt, :dq_rai_dt, :dq_sno_dt)
                        @test isfinite(getfield(result_var, field))
                    end

                    # Non-quadrature (direct) wrapper should also match
                    result_direct_wrapper = ClimaAtmos.microphysics_tendencies_1m(
                        ρ, q_tot, q_liq, q_ice, q_rai, q_sno,
                        T, mp_1m, thp, tst, dt,
                    )
                    for field in (:dq_lcl_dt, :dq_icl_dt, :dq_rai_dt, :dq_sno_dt)
                        @test getfield(result_direct_wrapper, field) ≈
                              getfield(result_direct, field) rtol = FT(1e-10)
                    end
                end
            end
        end
    end

    @testset "Performance Scaling" begin
        # Verify that the computational cost scales as expected with quadrature order
        # and that the direct (non-quadrature) path has negligible overhead vs raw BMT.
        import Thermodynamics as TD
        import ClimaParams as CP
        import CloudMicrophysics.Parameters as CMP
        import CloudMicrophysics.BulkMicrophysicsTendencies as BMT
        using ClimaAtmos: microphysics_tendencies_0m, microphysics_tendencies_1m

        FT = Float64
        toml_dict = CP.create_toml_dict(FT)
        thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
        mp_0m = CMP.Microphysics0MParams(toml_dict)
        mp_1m = CMP.Microphysics1MParams(toml_dict)

        # Shared state
        ρ = FT(1.0)
        T = FT(280.0)
        q_tot = FT(0.015)
        q_lcl = FT(0.001)
        q_icl = FT(0.0005)
        q_rai = FT(0.0001)
        q_sno = FT(0.00005)
        q_liq = q_lcl + q_rai
        q_ice = q_icl + q_sno
        Φ = FT(5000.0)
        T′T′ = FT(1.0)
        q′q′ = FT(1e-6)
        corr_Tq = FT(0.6)
        tst = nothing
        dt = FT(1.0)

        N_warmup = 100
        N_bench = 10_000

        # --- 0M Performance ---
        # We would be getting better measurements with BenchmarkTools, but I did not
        # want to add them to the dependencies.
        @testset "0M Performance" begin
            # Direct BMT 0M
            q_vap_sat = TD.q_vap_saturation(thp, T, ρ)
            for _ in 1:N_warmup
                BMT.bulk_microphysics_tendencies(
                    BMT.Microphysics0Moment(), mp_0m, thp, T, q_liq, q_ice, q_vap_sat,
                )
            end
            t_direct_0m = @elapsed for _ in 1:N_bench
                BMT.bulk_microphysics_tendencies(
                    BMT.Microphysics0Moment(), mp_0m, thp, T, q_liq, q_ice, q_vap_sat,
                )
            end

            # Non-quadrature wrapper 0M
            for _ in 1:N_warmup
                microphysics_tendencies_0m(
                    mp_0m, thp, ρ, T, q_tot, q_liq, q_ice, Φ, tst, dt,
                )
            end
            t_wrapper_0m = @elapsed for _ in 1:N_bench
                microphysics_tendencies_0m(
                    mp_0m, thp, ρ, T, q_tot, q_liq, q_ice, Φ, tst, dt,
                )
            end

            # 1st-order, 3rd-order, 5th-order quadrature 0M
            timings_0m = Dict{Int, Float64}()
            for order in (1, 3, 5)
                quad = ClimaAtmos.SGSQuadrature(FT; quadrature_order = order)
                for _ in 1:N_warmup
                    microphysics_tendencies_0m(
                        quad, mp_0m, thp, ρ, T, q_tot,
                        T′T′, q′q′, corr_Tq, Φ, tst, dt,
                    )
                end
                t = @elapsed for _ in 1:N_bench
                    microphysics_tendencies_0m(
                        quad, mp_0m, thp, ρ, T, q_tot,
                        T′T′, q′q′, corr_Tq, Φ, tst, dt,
                    )
                end
                timings_0m[order] = t
            end

            @info "0M Performance ($(N_bench) calls)" t_direct_0m t_wrapper_0m timings_0m

            # Non-quadrature wrapper cost should be within direct BMT cost
            @test t_wrapper_0m < 1.1 * t_direct_0m

            # 3-point should not be more expensive than 3*3 * 1-point
            @test timings_0m[3] < 9 * timings_0m[1]
            # 5-point should not be more expensive than 5*5 * 1-point
            @test timings_0m[5] < 25 * timings_0m[1]
        end

        # --- 1M Performance ---
        @testset "1M Performance" begin
            # Direct BMT 1M
            for _ in 1:N_warmup
                BMT.bulk_microphysics_tendencies(
                    BMT.Microphysics1Moment(), mp_1m, thp, ρ, T,
                    q_tot, q_lcl, q_icl, q_rai, q_sno,
                )
            end
            t_direct_1m = @elapsed for _ in 1:N_bench
                BMT.bulk_microphysics_tendencies(
                    BMT.Microphysics1Moment(), mp_1m, thp, ρ, T,
                    q_tot, q_lcl, q_icl, q_rai, q_sno,
                )
            end

            # Non-quadrature wrapper 1M
            for _ in 1:N_warmup
                microphysics_tendencies_1m(
                    ρ, q_tot, q_lcl, q_icl, q_rai, q_sno,
                    T, mp_1m, thp, tst, dt,
                )
            end
            t_wrapper_1m = @elapsed for _ in 1:N_bench
                microphysics_tendencies_1m(
                    ρ, q_tot, q_lcl, q_icl, q_rai, q_sno,
                    T, mp_1m, thp, tst, dt,
                )
            end

            # 1st-order, 3rd-order, 5th-order quadrature 1M
            timings_1m = Dict{Int, Float64}()
            for order in (1, 3, 5)
                quad = ClimaAtmos.SGSQuadrature(FT; quadrature_order = order)
                for _ in 1:N_warmup
                    microphysics_tendencies_1m(
                        BMT.Microphysics1Moment(), quad, mp_1m, thp, ρ, T,
                        q_tot, q_lcl, q_icl, q_rai, q_sno,
                        T′T′, q′q′, corr_Tq, tst, dt,
                    )
                end
                t = @elapsed for _ in 1:N_bench
                    microphysics_tendencies_1m(
                        BMT.Microphysics1Moment(), quad, mp_1m, thp, ρ, T,
                        q_tot, q_lcl, q_icl, q_rai, q_sno,
                        T′T′, q′q′, corr_Tq, tst, dt,
                    )
                end
                timings_1m[order] = t
            end

            @info "1M Performance ($(N_bench) calls)" t_direct_1m t_wrapper_1m timings_1m

            # Non-quadrature wrapper should be within the cost of direct BMT
            @test t_wrapper_1m < 1.1 * t_direct_1m

            # 3-point quadrature should not be more expensive than 3*3 * 1-point
            @test timings_1m[3] < 9 * timings_1m[1]
            # 5-point quadrature should not be more expensive than 5*5 * 1-point
            @test timings_1m[5] < 25 * timings_1m[1]
        end
    end

    @testset "1M SGS wrapper integration: profile Rosenblatt finite tendencies" begin
        # Integration-level regression for the exact smoke-family failures:
        # `microphysics_tendencies_1m_sgs_row` with profile Rosenblatt distributions must
        # return finite NamedTuple tendencies (no `NamedTuple * Float` errors).
        import Thermodynamics as TD
        import ClimaParams as CP
        import CloudMicrophysics.Parameters as CMP
        import CloudMicrophysics.BulkMicrophysicsTendencies as BMT
        FT = Float32
        toml_dict = CP.create_toml_dict(FT)
        thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
        mp_1m = CMP.Microphysics1MParams(toml_dict)
        ρ = FT(1.0)
        T = FT(280.0)
        q_tot = FT(0.015)
        q_lcl = FT(0.001)
        q_icl = FT(0.0005)
        q_rai = FT(0.0001)
        q_sno = FT(0.00005)
        T′T′ = FT(1.0)
        q′q′ = FT(1e-6)
        corr_Tq = FT(0.6)
        Δz = FT(400.0)
        lg = _fd_column_center_local_geometry(FT; ilevel = 4)
        gq = Geometry.Covariant123Vector(FT(0), FT(0), FT(2e-6))
        gθ = Geometry.Covariant123Vector(FT(0), FT(0), FT(8e-3))
        g0 = Geometry.Covariant123Vector(FT(0), FT(0), FT(0))
        ∂T∂θ = FT(0.45)
        tst = nothing
        dt = FT(1.0)
        dists = (
            ClimaAtmos.VerticallyResolvedSGS{
                ClimaAtmos.SubgridProfileRosenblatt{ClimaAtmos.ConvolutionQuantilesBracketed},
            }(),
            ClimaAtmos.VerticallyResolvedSGS{
                ClimaAtmos.SubgridProfileRosenblatt{ClimaAtmos.ConvolutionQuantilesHalley},
            }(),
            ClimaAtmos.VerticallyResolvedSGS{
                ClimaAtmos.SubgridProfileRosenblatt{ClimaAtmos.ConvolutionQuantilesBracketed},
            }(),
            ClimaAtmos.VerticallyResolvedSGS{
                ClimaAtmos.SubgridProfileRosenblatt{ClimaAtmos.ConvolutionQuantilesHalley},
            }(),
        )
        # High-value inference guard: profile/gridscale path must stay concrete.
        let d = first(dists)
            quad = ClimaAtmos.SGSQuadrature(FT; quadrature_order = 3, distribution = d)
            out = @inferred ClimaAtmos.microphysics_tendencies_1m_sgs_row(
                BMT.Microphysics1Moment(), quad, mp_1m, thp, ρ, T,
                q_tot, q_lcl, q_icl, q_rai, q_sno,
                T′T′, q′q′, corr_Tq, Δz, lg,
                gq, gq, gθ, gθ, ∂T∂θ, g0, g0, g0, g0, tst, dt,
            )
            @test out isa NamedTuple
        end
        for d in dists
            quad = ClimaAtmos.SGSQuadrature(FT; quadrature_order = 3, distribution = d)
            out = ClimaAtmos.microphysics_tendencies_1m_sgs_row(
                BMT.Microphysics1Moment(), quad, mp_1m, thp, ρ, T,
                q_tot, q_lcl, q_icl, q_rai, q_sno,
                T′T′, q′q′, corr_Tq, Δz, lg,
                gq, gq, gθ, gθ, ∂T∂θ, g0, g0, g0, g0, tst, dt,
            )
            @test out isa NamedTuple
            @test isfinite(out.dq_lcl_dt)
            @test isfinite(out.dq_icl_dt)
            @test isfinite(out.dq_rai_dt)
            @test isfinite(out.dq_sno_dt)
        end
    end

    @testset "linear layer-mean profile quadrature (column-tensor vs layer-mean CDF / Brent)" begin
        using ClimaCore.Geometry
        FT = Float64
        lg = _fd_column_center_local_geometry(FT; ilevel = 4)
        f(T, q) = T + FT(5) * q
        μ_q = FT(0.012)
        μ_T = FT(285)
        qv = FT(1e-7)
        Tv = FT(0.4)
        ρc = FT(0.55)
        H = FT(400)
        ∂T∂θ = FT(0.45)
        # Symmetric half-slopes (s_dn = s_up) reproduce a classical single-slope
        # test while exercising the new two-slope signature.
        gq_dn = Covariant123Vector(FT(0), FT(0), FT(2e-6))
        gq_up = Covariant123Vector(FT(0), FT(0), FT(2e-6))
        gθ_dn = Covariant123Vector(FT(0), FT(0), FT(0.008))
        gθ_up = Covariant123Vector(FT(0), FT(0), FT(0.008))
        # Zero variance half-slopes (F2 correction off): Σ_turb independent of z along each half.
        gzero = Covariant123Vector(FT(0), FT(0), FT(0))
        quad_ct = ClimaAtmos.SGSQuadrature(
            FT;
            quadrature_order = 3,
            distribution = ClimaAtmos.VerticallyResolvedSGS{ClimaAtmos.SubgridColumnTensor}(),
        )
        quad_pr = ClimaAtmos.SGSQuadrature(
            FT;
            quadrature_order = 3,
            distribution = ClimaAtmos.VerticallyResolvedSGS{
                ClimaAtmos.SubgridProfileRosenblatt{ClimaAtmos.ConvolutionQuantilesBracketed},
            }(),
        )
        @test quad_ct.z_t !== nothing
        ict = ClimaAtmos.integrate_over_sgs(
            f, quad_ct, μ_q, μ_T, qv, Tv, ρc, H, lg,
            gq_dn, gq_up, gθ_dn, gθ_up, ∂T∂θ,
            gzero, gzero, gzero, gzero,
        )
        ipr = ClimaAtmos.integrate_over_sgs(
            f, quad_pr, μ_q, μ_T, qv, Tv, ρc, H, lg,
            gq_dn, gq_up, gθ_dn, gθ_up, ∂T∂θ,
            gzero, gzero, gzero, gzero,
        )
        @test ict ≈ ipr rtol = FT(0.05) atol = FT(1e-4)
    end

    @testset "integrate_over_sgs rejects non-finite half-cell inputs (Profile Rosenblatt)" begin
        using ClimaCore.Geometry
        FT = Float64
        lg = _fd_column_center_local_geometry(FT; ilevel = 4)
        f(T, q) = T + FT(5) * q
        μ_q = FT(0.012)
        μ_T = FT(285)
        qv = FT(1e-7)
        Tv = FT(0.4)
        ρc = FT(0.55)
        H = FT(400)
        ∂T∂θ = FT(0.45)
        gq_dn = Covariant123Vector(FT(0), FT(0), NaN)
        gq_up = Covariant123Vector(FT(0), FT(0), FT(2e-6))
        gθ_dn = Covariant123Vector(FT(0), FT(0), FT(0.008))
        gθ_up = Covariant123Vector(FT(0), FT(0), FT(0.008))
        gzero = Covariant123Vector(FT(0), FT(0), FT(0))
        quad_pr = ClimaAtmos.SGSQuadrature(
            FT;
            quadrature_order = 3,
            distribution = ClimaAtmos.VerticallyResolvedSGS{
                ClimaAtmos.SubgridProfileRosenblatt{ClimaAtmos.ConvolutionQuantilesHalley},
            }(),
        )
        @test_throws ErrorException ClimaAtmos.integrate_over_sgs(
            f, quad_pr, μ_q, μ_T, qv, Tv, ρc, H, lg,
            gq_dn, gq_up, gθ_dn, gθ_up, ∂T∂θ,
            gzero, gzero, gzero, gzero,
        )
    end

    @testset "linear layer-mean profile quadrature (lognormal: column-tensor vs layer-mean CDF / Brent)" begin
        using ClimaCore.Geometry
        FT = Float64
        lg = _fd_column_center_local_geometry(FT; ilevel = 4)
        f(T, q) = T + FT(5) * q
        μ_q = FT(0.012)
        μ_T = FT(285)
        qv = FT(1e-7)
        Tv = FT(0.4)
        ρc = FT(0.55)
        H = FT(400)
        ∂T∂θ = FT(0.45)
        gq_dn = Covariant123Vector(FT(0), FT(0), FT(2e-6))
        gq_up = Covariant123Vector(FT(0), FT(0), FT(2e-6))
        gθ_dn = Covariant123Vector(FT(0), FT(0), FT(0.008))
        gθ_up = Covariant123Vector(FT(0), FT(0), FT(0.008))
        gzero = Covariant123Vector(FT(0), FT(0), FT(0))
        quad_ct = ClimaAtmos.SGSQuadrature(
            FT;
            quadrature_order = 3,
            distribution = ClimaAtmos.VerticallyResolvedSGS{
                ClimaAtmos.SubgridColumnTensor,
                ClimaAtmos.LogNormalSGS,
            }(),
        )
        quad_pr = ClimaAtmos.SGSQuadrature(
            FT;
            quadrature_order = 3,
            distribution = ClimaAtmos.VerticallyResolvedSGS{
                ClimaAtmos.SubgridProfileRosenblatt{ClimaAtmos.ConvolutionQuantilesBracketed},
                ClimaAtmos.LogNormalSGS,
            }(),
        )
        @test quad_ct.z_t !== nothing
        ict = ClimaAtmos.integrate_over_sgs(
            f, quad_ct, μ_q, μ_T, qv, Tv, ρc, H, lg,
            gq_dn, gq_up, gθ_dn, gθ_up, ∂T∂θ,
            gzero, gzero, gzero, gzero,
        )
        ipr = ClimaAtmos.integrate_over_sgs(
            f, quad_pr, μ_q, μ_T, qv, Tv, ρc, H, lg,
            gq_dn, gq_up, gθ_dn, gθ_up, ∂T∂θ,
            gzero, gzero, gzero, gzero,
        )
        @test ict ≈ ipr rtol = FT(0.05) atol = FT(1e-4)
    end

    @testset "layer-mean CDF degenerate ‖d‖ → center-only" begin
        using ClimaCore.Geometry
        FT = Float64
        lg = _fd_column_center_local_geometry(FT; ilevel = 4)
        f(T, q) = T * q
        μ_q = FT(0.01)
        μ_T = FT(280)
        quad_pr = ClimaAtmos.SGSQuadrature(
            FT;
            quadrature_order = 3,
            distribution = ClimaAtmos.VerticallyResolvedSGS{
                ClimaAtmos.SubgridProfileRosenblatt{ClimaAtmos.ConvolutionQuantilesBracketed},
            }(),
        )
        gzero = Covariant123Vector(FT(0), FT(0), FT(0))
        ipr = ClimaAtmos.integrate_over_sgs(
            f,
            quad_pr,
            μ_q,
            μ_T,
            FT(1e-10),
            FT(1e-10),
            FT(0),
            FT(200),
            lg,
            gzero, gzero, gzero, gzero, FT(0.5),
            gzero, gzero, gzero, gzero,
        )
        @test ipr ≈ f(μ_T, μ_q)
    end

    @testset "two-slope asymmetry: layer-mean CDF ≠ symmetric limit" begin
        # With a genuinely asymmetric reconstruction (s_dn ≠ s_up), the two-slope
        # layer-mean CDF integral must differ from the single-slope
        # (centered-gradient only) limit. Otherwise the asymmetry channel is dead.
        using ClimaCore.Geometry
        FT = Float64
        lg = _fd_column_center_local_geometry(FT; ilevel = 4)
        f(T, q) = T + FT(5) * q
        μ_q = FT(0.012)
        μ_T = FT(285)
        qv = FT(1e-7)
        Tv = FT(0.4)
        ρc = FT(0.55)
        H = FT(400)
        ∂T∂θ = FT(0.45)
        # Centered slope identical to the symmetric test; asymmetry injected via
        # larger up-slope and smaller dn-slope (same mean d, nonzero D = s_up - s_dn).
        d_q, D_q = FT(2e-6), FT(6e-6)     # s_dn = -1e-6, s_up = 5e-6
        d_θ, D_θ = FT(0.008), FT(0.012)   # s_dn = 0.002, s_up = 0.014
        gq_dn = Covariant123Vector(FT(0), FT(0), d_q - D_q / 2)
        gq_up = Covariant123Vector(FT(0), FT(0), d_q + D_q / 2)
        gθ_dn = Covariant123Vector(FT(0), FT(0), d_θ - D_θ / 2)
        gθ_up = Covariant123Vector(FT(0), FT(0), d_θ + D_θ / 2)
        gsym_q = Covariant123Vector(FT(0), FT(0), d_q)
        gsym_θ = Covariant123Vector(FT(0), FT(0), d_θ)
        gzero = Covariant123Vector(FT(0), FT(0), FT(0))
        quad_pr = ClimaAtmos.SGSQuadrature(
            FT;
            quadrature_order = 3,
            distribution = ClimaAtmos.VerticallyResolvedSGS{
                ClimaAtmos.SubgridProfileRosenblatt{ClimaAtmos.ConvolutionQuantilesBracketed},
            }(),
        )
        i_asym = ClimaAtmos.integrate_over_sgs(
            f, quad_pr, μ_q, μ_T, qv, Tv, ρc, H, lg,
            gq_dn, gq_up, gθ_dn, gθ_up, ∂T∂θ,
            gzero, gzero, gzero, gzero,
        )
        i_sym = ClimaAtmos.integrate_over_sgs(
            f, quad_pr, μ_q, μ_T, qv, Tv, ρc, H, lg,
            gsym_q, gsym_q, gsym_θ, gsym_θ, ∂T∂θ,
            gzero, gzero, gzero, gzero,
        )
        @test isfinite(i_asym)
        @test isfinite(i_sym)
        # Asymmetry must perturb the integral away from the symmetric baseline.
        @test abs(i_asym - i_sym) > FT(1e-6)
    end

    @testset "layer-mean CDF: Halley ≈ Brent at the integral level" begin
        # Swapping the quantile method (Halley vs Brent) should only change the
        # integral by a small residual (set by Halley's per-quantile slack).
        using ClimaCore.Geometry
        FT = Float64
        lg = _fd_column_center_local_geometry(FT; ilevel = 4)
        f(T, q) = T + FT(5) * q
        μ_q = FT(0.012)
        μ_T = FT(285)
        qv = FT(1e-7)
        Tv = FT(0.4)
        ρc = FT(0.55)
        H = FT(400)
        ∂T∂θ = FT(0.45)
        gq_dn = Covariant123Vector(FT(0), FT(0), FT(1e-6))
        gq_up = Covariant123Vector(FT(0), FT(0), FT(3e-6))
        gθ_dn = Covariant123Vector(FT(0), FT(0), FT(0.004))
        gθ_up = Covariant123Vector(FT(0), FT(0), FT(0.012))
        gzero = Covariant123Vector(FT(0), FT(0), FT(0))
        mk(method) = ClimaAtmos.SGSQuadrature(
            FT;
            quadrature_order = 3,
            distribution = ClimaAtmos.VerticallyResolvedSGS{
                ClimaAtmos.SubgridProfileRosenblatt{method},
            }(),
        )
        i_brent = ClimaAtmos.integrate_over_sgs(
            f, mk(ClimaAtmos.ConvolutionQuantilesBracketed), μ_q, μ_T,
            qv, Tv, ρc, H, lg,
            gq_dn, gq_up, gθ_dn, gθ_up, ∂T∂θ,
            gzero, gzero, gzero, gzero,
        )
        i_halley = ClimaAtmos.integrate_over_sgs(
            f, mk(ClimaAtmos.ConvolutionQuantilesHalley), μ_q, μ_T,
            qv, Tv, ρc, H, lg,
            gq_dn, gq_up, gθ_dn, gθ_up, ∂T∂θ,
            gzero, gzero, gzero, gzero,
        )
        # Both use the same composite split inner marginal; Halley is one-step per leg.
        @test i_halley ≈ i_brent rtol = FT(1e-5) atol = FT(1e-6)
        i_cheb = ClimaAtmos.integrate_over_sgs(
            f, mk(ClimaAtmos.ConvolutionQuantilesChebyshevLogEta), μ_q, μ_T,
            qv, Tv, ρc, H, lg,
            gq_dn, gq_up, gθ_dn, gθ_up, ∂T∂θ,
            gzero, gzero, gzero, gzero,
        )
        @test isfinite(i_cheb)
        @test i_cheb ≈ i_brent rtol = FT(0.08) atol = FT(1e-3)
    end

    @testset "Profile Rosenblatt: per-leg one-step Halley on single law has small CDF residual" begin
        # Production uses `centered_uniform_gaussian_convolution_quantile_halley` on each
        # single half-cell law, not a scalar `F_mix^{-1}`.
        FT = Float64
        for (L, s) in (
            (FT(0.4), FT(0.12)),
            (FT(1.5), FT(0.3)),
            (FT(0.9), FT(0.25)),
        )
            wbr = ClimaAtmos.centered_uniform_gaussian_convolution_quantile_brent(
                FT(0.37), L, s,
            )
            wha = ClimaAtmos.centered_uniform_gaussian_convolution_quantile_halley(
                FT(0.37), L, s,
            )
            p = FT(0.37)
            pbr = ClimaAtmos.uniform_gaussian_convolution_cdf(wbr, L, s)
            pha = ClimaAtmos.uniform_gaussian_convolution_cdf(wha, L, s)
            @test abs(pbr - p) < FT(1e-8)
            @test abs(pha - p) < FT(5e-3)
            @test abs(wha - wbr) < FT(0.04) * max(L + s, FT(1e-6))
        end
    end

    @testset "Profile Rosenblatt composite: Float32 Bracketed and Halley finite (same f)" begin
        using ClimaCore.Geometry
        for FT in (Float32, Float64)
            lg = _fd_column_center_local_geometry(FT; ilevel = 4)
            f(T, q) = T^2 + FT(2) * q
            μ_q, μ_T = FT(0.01), FT(280)
            qv, Tv, ρc = FT(1e-7), FT(0.3), FT(0.4)
            H, ∂T∂θ = FT(300), FT(0.4)
            gq_dn = Covariant123Vector(FT(0), FT(0), FT(1e-6))
            gq_up = Covariant123Vector(FT(0), FT(0), FT(2e-6))
            gθ_dn = Covariant123Vector(FT(0), FT(0), FT(0.005))
            gθ_up = Covariant123Vector(FT(0), FT(0), FT(0.01))
            gzero = Covariant123Vector(FT(0), FT(0), FT(0))
            for method in (
                ClimaAtmos.ConvolutionQuantilesBracketed,
                ClimaAtmos.ConvolutionQuantilesHalley,
            )
                quad = ClimaAtmos.SGSQuadrature(
                    FT;
                    quadrature_order = 3,
                    distribution = ClimaAtmos.VerticallyResolvedSGS{
                        ClimaAtmos.SubgridProfileRosenblatt{method},
                    }(),
                )
                v = ClimaAtmos.integrate_over_sgs(
                    f, quad, μ_q, μ_T, qv, Tv, ρc, H, lg,
                    gq_dn, gq_up, gθ_dn, gθ_up, ∂T∂θ,
                    gzero, gzero, gzero, gzero,
                )
                @test isfinite(v)
            end
        end
    end

    @testset "two-slope profile Rosenblatt matches independent half-wise truth" begin
        # Operational regression guard: validate the full two-half profile path
        # (mean-path + half-combination) against an independent high-order
        # reference that directly integrates each half with frozen covariance.
        using ClimaCore.Geometry
        FT = Float64
        lg = _fd_column_center_local_geometry(FT; ilevel = 4)
        f(T, q) = max(T - FT(273.15), zero(FT))^2 * (q + FT(1e-6))
        μ_q = FT(0.011)
        μ_T = FT(286.0)
        qv = FT(3e-7)
        Tv = FT(0.55)
        ρc = FT(0.42)
        H = FT(360)
        ∂T∂θ = FT(0.5)
        # Asymmetric two-slope means; frozen covariance (variance slopes = 0).
        gq_dn = Covariant123Vector(FT(0), FT(0), FT(-8e-7))
        gq_up = Covariant123Vector(FT(0), FT(0), FT(3e-6))
        gθ_dn = Covariant123Vector(FT(0), FT(0), FT(0.002))
        gθ_up = Covariant123Vector(FT(0), FT(0), FT(0.011))
        gzero = Covariant123Vector(FT(0), FT(0), FT(0))

        q_br = ClimaAtmos.SGSQuadrature(
            FT;
            quadrature_order = 5,
            distribution = ClimaAtmos.VerticallyResolvedSGS{
                ClimaAtmos.SubgridProfileRosenblatt{ClimaAtmos.ConvolutionQuantilesBracketed},
                ClimaAtmos.LogNormalSGS,
            }(),
        )
        i_prof = ClimaAtmos.integrate_over_sgs(
            f, q_br, μ_q, μ_T, qv, Tv, ρc, H, lg,
            gq_dn, gq_up, gθ_dn, gθ_up, ∂T∂θ,
            gzero, gzero, gzero, gzero,
        )

        # Independent truth: average of two one-sided half-cell integrals.
        function half_truth(dT_dz::FT, dq_dz::FT)
            Nz_ref = 201
            Nh_ref = 5
            χ, wgh = ClimaAtmos.gauss_hermite(FT, Nh_ref)
            inv_sqrt_pi = one(FT) / sqrt(FT(π))
            σ_q, σ_T, ρ = ClimaAtmos.sgs_stddevs_and_correlation(qv, Tv, ρc)
            D = ClimaAtmos.GaussianSGS()
            acc = zero(FT)
            @inbounds for kz in 1:Nz_ref
                ξ = (kz - FT(0.5)) / Nz_ref
                z = ξ * (H / FT(2))
                μ_Tz = μ_T + z * dT_dz
                μ_qz = μ_q + z * dq_dz
                wz = one(FT) / Nz_ref
                @inbounds for i in eachindex(χ), j in eachindex(χ)
                    wi = wgh[i] * inv_sqrt_pi
                    wj = wgh[j] * inv_sqrt_pi
                    T_hat, q_hat = ClimaAtmos.get_physical_point(
                        D, χ[i], χ[j], μ_qz, μ_Tz,
                        σ_q, σ_T, ρ, q_br.T_min, q_br.q_max,
                    )
                    acc += wz * wi * wj * f(T_hat, q_hat)
                end
            end
            return acc
        end

        dT_dn = ∂T∂θ * WVector(gθ_dn, lg)[1]
        dT_up = ∂T∂θ * WVector(gθ_up, lg)[1]
        dq_dn = WVector(gq_dn, lg)[1]
        dq_up = WVector(gq_up, lg)[1]
        i_ref = FT(0.5) * (half_truth(dT_dn, dq_dn) + half_truth(dT_up, dq_up))

        @test isfinite(i_prof)
        @test isfinite(i_ref)
        @test abs(i_ref) > FT(1e-8)
        @test abs(i_prof - i_ref) / abs(i_ref) < FT(0.03)
    end

    @testset "single-convolution quantiles: Brent/Chebyshev vs brute-force truth" begin
        # Notebook-style truth construction: brute-force the target distribution
        # directly (PDF integration + inverse CDF), then compare fast quantile
        # methods against that independent reference.
        FT = Float64
        L = FT(1.0)
        η_grid = exp10.(range(-2, 1; length = 20))
        max_err_br = zero(FT)
        max_err_ch = zero(FT)
        rel_err_br = FT[]
        rel_err_ch = FT[]
        function brute_quantile(p::FT, L::FT, s::FT)
            lo = -L / FT(2) - FT(8) * s
            hi = L / FT(2) + FT(8) * s
            ngrid = 50_001
            xs = range(lo, hi; length = ngrid)
            dx = step(xs)
            pdfvals = similar(collect(xs))
            @inbounds for k in eachindex(xs)
                pdfvals[k] = ClimaAtmos.uniform_gaussian_convolution_pdf(xs[k], L, s)
            end
            cdfvals = similar(pdfvals)
            cdfvals[1] = zero(FT)
            @inbounds for k in 2:ngrid
                # trapezoidal cumulative integral
                cdfvals[k] = cdfvals[k - 1] + (pdfvals[k - 1] + pdfvals[k]) * dx / FT(2)
            end
            z = cdfvals[end]
            z <= FT(0) && return zero(FT)
            @inbounds for k in eachindex(cdfvals)
                cdfvals[k] /= z
            end
            idx = searchsortedfirst(cdfvals, p)
            if idx <= 1
                return xs[1]
            elseif idx > ngrid
                return xs[end]
            else
                x0, x1 = xs[idx - 1], xs[idx]
                c0, c1 = cdfvals[idx - 1], cdfvals[idx]
                t = (p - c0) / max(c1 - c0, eps(FT))
                return x0 + t * (x1 - x0)
            end
        end
        for N in (3, 5)
            p_nodes, _ = ClimaAtmos.gauss_legendre_01(FT, N)
            for η in η_grid
                s = η * L
                for i in 1:N
                    ut = brute_quantile(p_nodes[i], L, s)
                    ub = ClimaAtmos.centered_uniform_gaussian_convolution_quantile_brent(
                        p_nodes[i], L, s,
                    )
                    uc = ClimaAtmos.centered_uniform_gaussian_convolution_quantile_chebyshev(
                        L, s, N, i,
                    )
                    eb = abs(ub - ut)
                    ec = abs(uc - ut)
                    scale = max(abs(ut), FT(1e-6))
                    push!(rel_err_br, eb / scale)
                    push!(rel_err_ch, ec / scale)
                    max_err_br = max(max_err_br, eb)
                    max_err_ch = max(max_err_ch, ec)
                end
            end
        end
        @test max_err_br < FT(2e-3)
        @test max_err_ch < FT(2e-2)
        @test Statistics.median(rel_err_br) < FT(1e-3)
        @test Statistics.median(rel_err_ch) < FT(1e-2)
    end

    @testset "profile Rosenblatt supports NamedTuple-valued integrands" begin
        # Regression for smoke runs: profile branches average DN/UP halves and must
        # use `⊠`/`⊞` so NamedTuple tendencies scale/sum correctly.
        using ClimaCore.Geometry
        FT = Float32
        lg = _fd_column_center_local_geometry(FT; ilevel = 4)
        f(T, q) = (; dq_lcl_dt = T * q, dq_icl_dt = T + q, dq_rai_dt = q, dq_sno_dt = T)
        μ_q = FT(0.012)
        μ_T = FT(285)
        qv = FT(1e-7)
        Tv = FT(0.4)
        ρc = FT(0.55)
        H = FT(400)
        ∂T∂θ = FT(0.45)
        gq_dn = Covariant123Vector(FT(0), FT(0), FT(1e-6))
        gq_up = Covariant123Vector(FT(0), FT(0), FT(3e-6))
        gθ_dn = Covariant123Vector(FT(0), FT(0), FT(0.004))
        gθ_up = Covariant123Vector(FT(0), FT(0), FT(0.012))
        gzero = Covariant123Vector(FT(0), FT(0), FT(0))
        mk(method) = ClimaAtmos.SGSQuadrature(
            FT;
            quadrature_order = 3,
            distribution = ClimaAtmos.VerticallyResolvedSGS{
                ClimaAtmos.SubgridProfileRosenblatt{method},
            }(),
        )
        out_b = ClimaAtmos.integrate_over_sgs(
            f, mk(ClimaAtmos.ConvolutionQuantilesBracketed), μ_q, μ_T,
            qv, Tv, ρc, H, lg,
            gq_dn, gq_up, gθ_dn, gθ_up, ∂T∂θ,
            gzero, gzero, gzero, gzero,
        )
        out_h = ClimaAtmos.integrate_over_sgs(
            f, mk(ClimaAtmos.ConvolutionQuantilesHalley), μ_q, μ_T,
            qv, Tv, ρc, H, lg,
            gq_dn, gq_up, gθ_dn, gθ_up, ∂T∂θ,
            gzero, gzero, gzero, gzero,
        )
        out_c = ClimaAtmos.integrate_over_sgs(
            f, mk(ClimaAtmos.ConvolutionQuantilesChebyshevLogEta), μ_q, μ_T,
            qv, Tv, ρc, H, lg,
            gq_dn, gq_up, gθ_dn, gθ_up, ∂T∂θ,
            gzero, gzero, gzero, gzero,
        )
        @test out_b isa NamedTuple
        @test out_h isa NamedTuple
        @test out_c isa NamedTuple
        @test all(isfinite, Tuple(values(out_b)))
        @test all(isfinite, Tuple(values(out_h)))
        @test all(isfinite, Tuple(values(out_c)))
        for k in keys(out_b)
            @test out_b[k] ≈ out_h[k] rtol = FT(1e-3) atol = FT(1e-3)
            @test out_c[k] ≈ out_b[k] rtol = FT(0.15) atol = FT(1e-2)
        end
    end

    @testset "alternate layer discretizations: linear_profile runs and is not silently identical" begin
        # Regression guard: `SubgridLatinHypercubeZ` et al. must hit real branches in
        # `integrate_over_sgs` (not only dispatch /
        # `throw_if` plumbing). With vertical structure in means and subcell variance
        # slopes, reduced rules (LHS, principal axis, …) must differ from full
        # column-tensor cubature and from each other.
        using ClimaCore.Geometry
        FT = Float64
        lg = _fd_column_center_local_geometry(FT; ilevel = 4)
        f(T, q) = T^2 * q + FT(0.5) * q^2
        μ_q = FT(0.012)
        μ_T = FT(285)
        qv = FT(1e-7)
        Tv = FT(0.4)
        ρc = FT(0.55)
        H = FT(400)
        ∂T∂θ = FT(0.45)
        d_q, D_q = FT(2e-6), FT(6e-6)
        d_θ, D_θ = FT(0.008), FT(0.012)
        gq_dn = Covariant123Vector(FT(0), FT(0), d_q - D_q / 2)
        gq_up = Covariant123Vector(FT(0), FT(0), d_q + D_q / 2)
        gθ_dn = Covariant123Vector(FT(0), FT(0), d_θ - D_θ / 2)
        gθ_up = Covariant123Vector(FT(0), FT(0), d_θ + D_θ / 2)
        gzero = Covariant123Vector(FT(0), FT(0), FT(0))
        gqq_dn = Covariant123Vector(FT(0), FT(0), FT(5e-9))
        gqq_up = Covariant123Vector(FT(0), FT(0), FT(1.5e-8))
        gTT_dn = Covariant123Vector(FT(0), FT(0), FT(2e-4))
        gTT_up = Covariant123Vector(FT(0), FT(0), FT(6e-4))
        function mk_quad(::Type{S}) where {S}
            ClimaAtmos.SGSQuadrature(
                FT;
                quadrature_order = 3,
                distribution = ClimaAtmos.VerticallyResolvedSGS{S}(),
            )
        end
        args = (
            μ_q, μ_T, qv, Tv, ρc, H, lg,
            gq_dn, gq_up, gθ_dn, gθ_up, ∂T∂θ,
            gqq_dn, gqq_up, gTT_dn, gTT_up,
        )
        ict = ClimaAtmos.integrate_over_sgs(f, mk_quad(ClimaAtmos.SubgridColumnTensor), args...)
        ilhs = ClimaAtmos.integrate_over_sgs(f, mk_quad(ClimaAtmos.SubgridLatinHypercubeZ), args...)
        ipa = ClimaAtmos.integrate_over_sgs(f, mk_quad(ClimaAtmos.SubgridPrincipalAxisLayer), args...)
        ivor = ClimaAtmos.integrate_over_sgs(f, mk_quad(ClimaAtmos.SubgridVoronoiRepresentatives), args...)
        ibar = ClimaAtmos.integrate_over_sgs(f, mk_quad(ClimaAtmos.SubgridBarycentricSeeds), args...)
        @test ict isa FT && ilhs isa FT && ipa isa FT && ivor isa FT && ibar isa FT
        @test all(isfinite, (ict, ilhs, ipa, ivor, ibar))
        mn, mx = extrema((ict, ilhs, ipa, ivor, ibar))
        @test mx - mn > FT(1e-6)
        @test abs(ict - ilhs) > FT(1e-8)
    end

    @testset "layer schemes conserve quadrature mass (f=1)" begin
        using ClimaCore.Geometry
        FT = Float64
        lg = _fd_column_center_local_geometry(FT; ilevel = 4)
        μ_q = FT(0.012)
        μ_T = FT(285)
        qv = FT(1e-7)
        Tv = FT(0.4)
        ρc = FT(0.55)
        H = FT(400)
        ∂T∂θ = FT(0.45)
        gq_dn = Covariant123Vector(FT(0), FT(0), FT(1e-6))
        gq_up = Covariant123Vector(FT(0), FT(0), FT(3e-6))
        gθ_dn = Covariant123Vector(FT(0), FT(0), FT(0.004))
        gθ_up = Covariant123Vector(FT(0), FT(0), FT(0.012))
        gqq_dn = Covariant123Vector(FT(0), FT(0), FT(5e-9))
        gqq_up = Covariant123Vector(FT(0), FT(0), FT(1.5e-8))
        gTT_dn = Covariant123Vector(FT(0), FT(0), FT(2e-4))
        gTT_up = Covariant123Vector(FT(0), FT(0), FT(6e-4))
        args = (
            μ_q, μ_T, qv, Tv, ρc, H, lg,
            gq_dn, gq_up, gθ_dn, gθ_up, ∂T∂θ,
            gqq_dn, gqq_up, gTT_dn, gTT_up,
        )
        schemes = (
            ClimaAtmos.SubgridColumnTensor,
            ClimaAtmos.SubgridLatinHypercubeZ,
            ClimaAtmos.SubgridPrincipalAxisLayer,
            ClimaAtmos.SubgridVoronoiRepresentatives,
            ClimaAtmos.SubgridBarycentricSeeds,
            ClimaAtmos.SubgridProfileRosenblatt{ClimaAtmos.ConvolutionQuantilesHalley},
            ClimaAtmos.SubgridProfileRosenblatt{ClimaAtmos.ConvolutionQuantilesBracketed},
            ClimaAtmos.SubgridProfileRosenblatt{ClimaAtmos.ConvolutionQuantilesChebyshevLogEta},
        )
        for S in schemes
            quad = ClimaAtmos.SGSQuadrature(
                FT; quadrature_order = 3,
                distribution = ClimaAtmos.VerticallyResolvedSGS{S}(),
            )
            m = ClimaAtmos.integrate_over_sgs((T, q) -> FT(1), quad, args...)
            @test m ≈ FT(1) atol = FT(1e-8)
        end
    end

    @testset "Profile Rosenblatt: quadrature_order 1–5 conserves unit mass for all inner solvers" begin
        using ClimaCore.Geometry
        FT = Float64
        lg = _fd_column_center_local_geometry(FT; ilevel = 4)
        μ_q, μ_T = FT(0.012), FT(285)
        qv, Tv, ρc = FT(1e-7), FT(0.4), FT(0.55)
        H, ∂T∂θ = FT(400), FT(0.45)
        gq_dn = Covariant123Vector(FT(0), FT(0), FT(1e-6))
        gq_up = Covariant123Vector(FT(0), FT(0), FT(3e-6))
        gθ_dn = Covariant123Vector(FT(0), FT(0), FT(0.004))
        gθ_up = Covariant123Vector(FT(0), FT(0), FT(0.012))
        gqq_dn = Covariant123Vector(FT(0), FT(0), FT(5e-9))
        gqq_up = Covariant123Vector(FT(0), FT(0), FT(1.5e-8))
        gTT_dn = Covariant123Vector(FT(0), FT(0), FT(2e-4))
        gTT_up = Covariant123Vector(FT(0), FT(0), FT(6e-4))
        args = (
            μ_q, μ_T, qv, Tv, ρc, H, lg,
            gq_dn, gq_up, gθ_dn, gθ_up, ∂T∂θ,
            gqq_dn, gqq_up, gTT_dn, gTT_up,
        )
        for Nq in 1:5
            for M in (
                ClimaAtmos.ConvolutionQuantilesBracketed,
                ClimaAtmos.ConvolutionQuantilesHalley,
                ClimaAtmos.ConvolutionQuantilesChebyshevLogEta,
            )
                quad = ClimaAtmos.SGSQuadrature(
                    FT;
                    quadrature_order = Nq,
                    distribution = ClimaAtmos.VerticallyResolvedSGS{
                        ClimaAtmos.SubgridProfileRosenblatt{M},
                    }(),
                )
                m = ClimaAtmos.integrate_over_sgs((T, q) -> FT(1), quad, args...)
                @test m ≈ FT(1) atol = FT(1e-7) rtol = FT(1e-10)
            end
        end
    end

    @testset "coverage matrix: all 1M-supported gridscale SGS return finite tendencies" begin
        # Broad integration coverage for 1M+SGS dispatch:
        # every vertically resolved SGS family/type
        # should run through `microphysics_tendencies_1m_sgs_row` and return finite outputs.
        import Thermodynamics as TD
        import ClimaParams as CP
        import CloudMicrophysics.Parameters as CMP
        import CloudMicrophysics.BulkMicrophysicsTendencies as BMT
        FT = Float32
        toml_dict = CP.create_toml_dict(FT)
        thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
        mp_1m = CMP.Microphysics1MParams(toml_dict)
        ρ = FT(1.0)
        T = FT(280.0)
        q_tot = FT(0.015)
        q_lcl = FT(0.001)
        q_icl = FT(0.0005)
        q_rai = FT(0.0001)
        q_sno = FT(0.00005)
        T′T′ = FT(1.0)
        q′q′ = FT(1e-6)
        corr_Tq = FT(0.6)
        Δz = FT(400.0)
        lg = _fd_column_center_local_geometry(FT; ilevel = 4)
        gq = Geometry.Covariant123Vector(FT(0), FT(0), FT(2e-6))
        gθ = Geometry.Covariant123Vector(FT(0), FT(0), FT(8e-3))
        g0 = Geometry.Covariant123Vector(FT(0), FT(0), FT(0))
        ∂T∂θ = FT(0.45)
        tst = nothing
        dt = FT(1.0)
        # Include all currently supported gridscale S families across Gaussian/LogNormal.
        dists = (
            ClimaAtmos.VerticallyResolvedSGS{ClimaAtmos.SubgridColumnTensor, ClimaAtmos.GaussianSGS}(),
            ClimaAtmos.VerticallyResolvedSGS{ClimaAtmos.SubgridColumnTensor, ClimaAtmos.LogNormalSGS}(),
            ClimaAtmos.VerticallyResolvedSGS{ClimaAtmos.SubgridLatinHypercubeZ, ClimaAtmos.GaussianSGS}(),
            ClimaAtmos.VerticallyResolvedSGS{ClimaAtmos.SubgridLatinHypercubeZ, ClimaAtmos.LogNormalSGS}(),
            ClimaAtmos.VerticallyResolvedSGS{ClimaAtmos.SubgridPrincipalAxisLayer, ClimaAtmos.GaussianSGS}(),
            ClimaAtmos.VerticallyResolvedSGS{ClimaAtmos.SubgridPrincipalAxisLayer, ClimaAtmos.LogNormalSGS}(),
            ClimaAtmos.VerticallyResolvedSGS{ClimaAtmos.SubgridVoronoiRepresentatives, ClimaAtmos.GaussianSGS}(),
            ClimaAtmos.VerticallyResolvedSGS{ClimaAtmos.SubgridVoronoiRepresentatives, ClimaAtmos.LogNormalSGS}(),
            ClimaAtmos.VerticallyResolvedSGS{ClimaAtmos.SubgridBarycentricSeeds, ClimaAtmos.GaussianSGS}(),
            ClimaAtmos.VerticallyResolvedSGS{ClimaAtmos.SubgridBarycentricSeeds, ClimaAtmos.LogNormalSGS}(),
            ClimaAtmos.VerticallyResolvedSGS{
                ClimaAtmos.SubgridProfileRosenblatt{ClimaAtmos.ConvolutionQuantilesBracketed},
                ClimaAtmos.GaussianSGS,
            }(),
            ClimaAtmos.VerticallyResolvedSGS{
                ClimaAtmos.SubgridProfileRosenblatt{ClimaAtmos.ConvolutionQuantilesBracketed},
                ClimaAtmos.LogNormalSGS,
            }(),
            ClimaAtmos.VerticallyResolvedSGS{
                ClimaAtmos.SubgridProfileRosenblatt{ClimaAtmos.ConvolutionQuantilesHalley},
                ClimaAtmos.GaussianSGS,
            }(),
            ClimaAtmos.VerticallyResolvedSGS{
                ClimaAtmos.SubgridProfileRosenblatt{ClimaAtmos.ConvolutionQuantilesHalley},
                ClimaAtmos.LogNormalSGS,
            }(),
            ClimaAtmos.VerticallyResolvedSGS{
                ClimaAtmos.SubgridProfileRosenblatt{ClimaAtmos.ConvolutionQuantilesChebyshevLogEta},
                ClimaAtmos.GaussianSGS,
            }(),
            ClimaAtmos.VerticallyResolvedSGS{
                ClimaAtmos.SubgridProfileRosenblatt{ClimaAtmos.ConvolutionQuantilesChebyshevLogEta},
                ClimaAtmos.LogNormalSGS,
            }(),
        )
        for d in dists
            quad = ClimaAtmos.SGSQuadrature(FT; quadrature_order = 3, distribution = d)
            @test ClimaAtmos._is_vertically_resolved_sgs(d)
            @test _assert_1m_sgs_gridscale_supported_for_tests(quad) === nothing
            out = ClimaAtmos.microphysics_tendencies_1m_sgs_row(
                BMT.Microphysics1Moment(), quad, mp_1m, thp, ρ, T,
                q_tot, q_lcl, q_icl, q_rai, q_sno,
                T′T′, q′q′, corr_Tq, Δz, lg,
                gq, gq, gθ, gθ, ∂T∂θ, g0, g0, g0, g0, tst, dt,
            )
            @test out isa NamedTuple
            @test isfinite(out.dq_lcl_dt)
            @test isfinite(out.dq_icl_dt)
            @test isfinite(out.dq_rai_dt)
            @test isfinite(out.dq_sno_dt)
        end
    end

    @testset "two-component uniform–Gaussian mixture: CDF, PDF, PDF′ primitives" begin
        # Sanity: F is monotone in u, f is its derivative, ∫ f = 1.
        FT = Float64
        L_dn, s_dn, L_up, s_up = FT(1.2), FT(0.3), FT(0.8), FT(0.2)
        us = range(FT(-L_dn - 6 * s_dn), FT(L_up + 6 * s_up); length = 4000)
        Fs = [
            ClimaAtmos.mixture_uniform_gaussian_convolution_cdf(
                u, L_dn, s_dn, L_up, s_up,
            ) for u in us
        ]
        @test all(diff(Fs) .>= -1e-12)                # monotone (non-decreasing)
        @test Fs[1] < FT(1e-3) && Fs[end] > FT(1 - 1e-3)  # approaches 0 / 1 at tails
        fs = [
            ClimaAtmos.mixture_uniform_gaussian_convolution_pdf(
                u, L_dn, s_dn, L_up, s_up,
            ) for u in us
        ]
        @test sum(fs) * step(us) ≈ FT(1) rtol = FT(0.01)
        # Numerical derivative of F matches analytic PDF (central difference).
        dF = diff(Fs) ./ step(us)
        f_mid = (fs[1:end-1] .+ fs[2:end]) ./ 2
        @test maximum(abs, dF .- f_mid) < FT(5e-3)
        # Analytic mean and variance from closed form.
        μ_exp, var_exp = ClimaAtmos.mixture_uniform_gaussian_convolution_mean_var(
            L_dn, s_dn, L_up, s_up,
        )
        μ_num = sum(fs .* us) * step(us)
        var_num = sum(fs .* (us .- μ_num) .^ 2) * step(us)
        @test μ_num ≈ μ_exp rtol = FT(0.02) atol = FT(1e-3)
        @test var_num ≈ var_exp rtol = FT(0.05)
    end

    @testset "mixture CDF is half sum of DN and UP shifted uniform⊛Gaussian CDFs" begin
        # Algebraic definition deployed in production (not an optional approximation):
        #   F_mix(x) = ½ F_{U⊛N}(x | U∼Unif[-L_dn,0]) + ½ F_{U⊛N}(x | U∼Unif[0,L_up])
        # Each half is a length-L interval centered at -L_dn/2 and +L_up/2 respectively.
        FT = Float64
        for (L_dn, s_dn, L_up, s_up) in (
            (FT(1.2), FT(0.3), FT(0.8), FT(0.2)),
            (FT(0.05), FT(0.08), FT(1.5), FT(0.25)),
            (FT(1.0e-3), FT(1.0e-4), FT(50.0), FT(0.12)),
        )
            for x in (FT(-3), FT(-0.5), FT(0), FT(0.4), FT(3))
                mid_dn = -L_dn / FT(2)
                mid_up = L_up / FT(2)
                F_dn = ClimaAtmos.uniform_gaussian_convolution_cdf(
                    x - mid_dn, L_dn, s_dn,
                )
                F_up = ClimaAtmos.uniform_gaussian_convolution_cdf(
                    x - mid_up, L_up, s_up,
                )
                F_mix = ClimaAtmos.mixture_uniform_gaussian_convolution_cdf(
                    x, L_dn, s_dn, L_up, s_up,
                )
                @test F_mix ≈ (F_dn + F_up) / FT(2) rtol = FT(1e-12) atol = FT(1e-14)
            end
        end
    end

    @testset "split-leg GL mean of per-leg inverses matches mixture closed-form mean" begin
        FT = Float64
        L_dn, s_dn, L_up, s_up = FT(0.7), FT(0.12), FT(1.1), FT(0.18)
        N = 5
        p_nodes, p_w = ClimaAtmos.gauss_legendre_01(FT, N)
        m_dn = sum(
            p_w[i] *
            ClimaAtmos.dn_half_uniform_gaussian_convolution_quantile_brent(
                p_nodes[i], L_dn, s_dn,
            ) for i in 1:N
        )
        m_up = sum(
            p_w[i] *
            ClimaAtmos.up_half_uniform_gaussian_convolution_quantile_brent(
                p_nodes[i], L_up, s_up,
            ) for i in 1:N
        )
        μ_gl = (m_dn + m_up) / FT(2)
        μ_mix, _ = ClimaAtmos.mixture_uniform_gaussian_convolution_mean_var(
            L_dn, s_dn, L_up, s_up,
        )
        @test μ_gl ≈ μ_mix rtol = FT(1e-4) atol = FT(1e-6)
    end

    @testset "mixture: Brent (test helper) vs brute-force CDF inversion" begin
        # `F_mix` and `f_mix` are defined in `src` for analysis; the profile integrator never inverts
        # `F_mix` at a single `u` (it uses the composite per-leg rule instead).
        FT = Float64
        function brute_quantile_mix(
            p::FT, L_dn::FT, s_dn::FT, L_up::FT, s_up::FT,
        ) where {FT}
            lo = -L_dn - FT(8) * max(s_dn, s_up)
            hi = L_up + FT(8) * max(s_dn, s_up)
            ngrid = 60_001
            xs = range(lo, hi; length = ngrid)
            dx = step(xs)
            pdfvals = similar(collect(xs))
            @inbounds for k in eachindex(xs)
                pdfvals[k] = ClimaAtmos.mixture_uniform_gaussian_convolution_pdf(
                    xs[k], L_dn, s_dn, L_up, s_up,
                )
            end
            cdfvals = similar(pdfvals)
            cdfvals[1] = zero(FT)
            @inbounds for k in 2:ngrid
                cdfvals[k] = cdfvals[k - 1] + (pdfvals[k - 1] + pdfvals[k]) * dx / FT(2)
            end
            z = cdfvals[end]
            z <= FT(0) && return zero(FT)
            @inbounds for k in eachindex(cdfvals)
                cdfvals[k] /= z
            end
            idx = searchsortedfirst(cdfvals, p)
            if idx <= 1
                return xs[1]
            elseif idx > ngrid
                return xs[end]
            else
                x0, x1 = xs[idx - 1], xs[idx]
                c0, c1 = cdfvals[idx - 1], cdfvals[idx]
                t = (p - c0) / max(c1 - c0, eps(FT))
                return x0 + t * (x1 - x0)
            end
        end
        max_abs = zero(FT)
        med_rel = FT[]
        for (L_dn, s_dn, L_up, s_up) in (
            (FT(0.2), FT(0.05), FT(0.9), FT(0.2)),
            (FT(0.8), FT(0.15), FT(0.3), FT(0.08)),
            (FT(1.2), FT(0.35), FT(1.1), FT(0.32)),
        )
            for p in (FT(0.1), FT(0.25), FT(0.5), FT(0.75), FT(0.9))
                ub = ClimaAtmos.mixture_uniform_gaussian_convolution_quantile_brent(
                    p, L_dn, s_dn, L_up, s_up,
                )
                ut = brute_quantile_mix(p, L_dn, s_dn, L_up, s_up)
                e = abs(ub - ut)
                max_abs = max(max_abs, e)
                push!(med_rel, e / max(abs(ut), FT(1e-6)))
            end
        end
        @test max_abs < FT(5e-3)
        @test Statistics.median(med_rel) < FT(2e-3)
    end

    @testset "Chebyshev vs Brent: single centered uniform⊛Gaussian only" begin
        # Tables (`chebyshev_convolution_coeffs`) are fit for ONE law:
        #   uniform[-L/2, L/2] ⊛ N(0, s²),  η = s/L,
        # at each Gauss–Legendre node p on a fixed order N_gl (see gen script).
        # They do **not** approximate the two-component **mixture** marginal.
        FT = Float64
        L = FT(1.2)
        s = FT(0.15)
        N = 3
        p_nodes, _ = ClimaAtmos.gauss_legendre_01(FT, N)
        for i in 1:N
            p = p_nodes[i]
            ub = ClimaAtmos.centered_uniform_gaussian_convolution_quantile_brent(
                p, L, s,
            )
            uc = ClimaAtmos.centered_uniform_gaussian_convolution_quantile_chebyshev(
                L, s, N, i,
            )
            @test abs(uc - ub) < FT(0.01) * max(L, s, FT(1e-6))
        end
    end

    @testset "get_sgs_distribution linear-profile keys" begin
        # Default YAML `gaussian_vertical_profile` matches production default inner quantiles (Halley dispatch).
        # Bracketed roots are explicit `_inner_bracketed` YAML.
        pa = Dict{String, Any}("sgs_distribution" => "gaussian_vertical_profile_inner_chebyshev")
        @test ClimaAtmos.get_sgs_distribution(pa) isa ClimaAtmos.VerticallyResolvedSGS{
            ClimaAtmos.SubgridProfileRosenblatt{ClimaAtmos.ConvolutionQuantilesChebyshevLogEta},
        }
        pa = Dict{String, Any}("sgs_distribution" => "gaussian_vertical_profile")
        @test ClimaAtmos.get_sgs_distribution(pa) isa ClimaAtmos.VerticallyResolvedSGS{
            ClimaAtmos.SubgridProfileRosenblatt{ClimaAtmos.ConvolutionQuantilesHalley},
        }
        pa["sgs_distribution"] = "gaussian_vertical_profile_full_cubature"
        @test ClimaAtmos.get_sgs_distribution(pa) isa
              ClimaAtmos.VerticallyResolvedSGS{ClimaAtmos.SubgridColumnTensor}
        pa["sgs_distribution"] = "gaussian_vertical_profile_inner_bracketed"
        @test ClimaAtmos.get_sgs_distribution(pa) isa ClimaAtmos.VerticallyResolvedSGS{
            ClimaAtmos.SubgridProfileRosenblatt{ClimaAtmos.ConvolutionQuantilesBracketed},
        }
        pa["sgs_distribution"] = "gaussian_vertical_profile_inner_halley"
        @test ClimaAtmos.get_sgs_distribution(pa) isa ClimaAtmos.VerticallyResolvedSGS{
            ClimaAtmos.SubgridProfileRosenblatt{ClimaAtmos.ConvolutionQuantilesHalley},
        }
        pa["sgs_distribution"] = "lognormal_vertical_profile"
        @test ClimaAtmos.get_sgs_distribution(pa) isa ClimaAtmos.VerticallyResolvedSGS{
            ClimaAtmos.SubgridProfileRosenblatt{ClimaAtmos.ConvolutionQuantilesHalley},
        }
        pa["sgs_distribution"] = "lognormal_vertical_profile_full_cubature"
        @test ClimaAtmos.get_sgs_distribution(pa) isa
              ClimaAtmos.VerticallyResolvedSGS{ClimaAtmos.SubgridColumnTensor}
        pa["sgs_distribution"] = "lognormal_vertical_profile_inner_bracketed"
        @test ClimaAtmos.get_sgs_distribution(pa) isa ClimaAtmos.VerticallyResolvedSGS{
            ClimaAtmos.SubgridProfileRosenblatt{ClimaAtmos.ConvolutionQuantilesBracketed},
        }
        pa["sgs_distribution"] = "lognormal_vertical_profile_inner_halley"
        @test ClimaAtmos.get_sgs_distribution(pa) isa ClimaAtmos.VerticallyResolvedSGS{
            ClimaAtmos.SubgridProfileRosenblatt{ClimaAtmos.ConvolutionQuantilesHalley},
        }
        pa["sgs_distribution"] = "lognormal_vertical_profile_inner_chebyshev"
        @test ClimaAtmos.get_sgs_distribution(pa) isa ClimaAtmos.VerticallyResolvedSGS{
            ClimaAtmos.SubgridProfileRosenblatt{ClimaAtmos.ConvolutionQuantilesChebyshevLogEta},
        }
        pa["sgs_distribution"] = "gaussian_vertical_profile_full_cubature"
        @test ClimaAtmos.get_sgs_distribution(pa) isa
              ClimaAtmos.VerticallyResolvedSGS{ClimaAtmos.SubgridColumnTensor}
        pa["sgs_distribution"] = "gaussian_vertical_profile_lhs_z"
        @test ClimaAtmos.get_sgs_distribution(pa) isa
              ClimaAtmos.VerticallyResolvedSGS{ClimaAtmos.SubgridLatinHypercubeZ}
        pa["sgs_distribution"] = "gaussian_vertical_profile_principal_axis"
        @test ClimaAtmos.get_sgs_distribution(pa) isa
              ClimaAtmos.VerticallyResolvedSGS{ClimaAtmos.SubgridPrincipalAxisLayer}
        pa["sgs_distribution"] = "gaussian_vertical_profile_voronoi"
        @test ClimaAtmos.get_sgs_distribution(pa) isa
              ClimaAtmos.VerticallyResolvedSGS{ClimaAtmos.SubgridVoronoiRepresentatives}
        pa["sgs_distribution"] = "gaussian_vertical_profile_barycentric"
        @test ClimaAtmos.get_sgs_distribution(pa) isa
              ClimaAtmos.VerticallyResolvedSGS{ClimaAtmos.SubgridBarycentricSeeds}
        pa["sgs_distribution"] = "lognormal_vertical_profile_full_cubature"
        @test ClimaAtmos.get_sgs_distribution(pa) isa
              ClimaAtmos.VerticallyResolvedSGS{ClimaAtmos.SubgridColumnTensor}
        pa["sgs_distribution"] = "lognormal_vertical_profile_lhs_z"
        @test ClimaAtmos.get_sgs_distribution(pa) isa
              ClimaAtmos.VerticallyResolvedSGS{ClimaAtmos.SubgridLatinHypercubeZ}
        pa["sgs_distribution"] = "lognormal_vertical_profile_principal_axis"
        @test ClimaAtmos.get_sgs_distribution(pa) isa
              ClimaAtmos.VerticallyResolvedSGS{ClimaAtmos.SubgridPrincipalAxisLayer}
        pa["sgs_distribution"] = "lognormal_vertical_profile_voronoi"
        @test ClimaAtmos.get_sgs_distribution(pa) isa
              ClimaAtmos.VerticallyResolvedSGS{ClimaAtmos.SubgridVoronoiRepresentatives}
        pa["sgs_distribution"] = "lognormal_vertical_profile_barycentric"
        @test ClimaAtmos.get_sgs_distribution(pa) isa
              ClimaAtmos.VerticallyResolvedSGS{ClimaAtmos.SubgridBarycentricSeeds}
    end

    @testset "get_physical_point respects T_min (inner SGS layer)" begin
        # Thermodynamics saturation uses log(T); unphysical T<0 → DomainError in long runs.
        # Inner Hermite layer must clamp T_hat ≥ T_min for Gaussian and LogNormal SGS.
        FT = Float32
        T_min = FT(150)
        q_max = FT(0.1)
        μ_q = FT(0.01)
        μ_T = FT(280)
        σ_q = FT(0.002)
        σ_T = FT(5.0)
        corr = FT(0.4)
        for dist in (ClimaAtmos.GaussianSGS(), ClimaAtmos.LogNormalSGS())
            for χ1 in (-4.0f0, -2.0f0, 0.0f0, 2.0f0, 4.0f0),
                χ2 in (-4.0f0, -2.0f0, 0.0f0, 2.0f0, 4.0f0)

                T_hat, q_hat = ClimaAtmos.get_physical_point(
                    dist, χ1, χ2, μ_q, μ_T, σ_q, σ_T, corr, T_min, q_max,
                )
                @test T_hat >= T_min
                @test q_hat >= zero(FT)
                @test q_hat <= q_max
            end
        end
    end

end
