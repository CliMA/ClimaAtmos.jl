using Test
import ClimaAtmos as CA

const BeresSourceParams = CA.BeresSourceParams
const wave_source = CA.wave_source

# NOTE: The wave_source / Beres spectrum computation is EDMF-mode-agnostic.
# It takes (Q0, h, u_heat, N_source) directly, so these unit tests apply
# identically whether the upstream EDMF is diagnostic or prognostic.
# The prognostic EDMF integration path is covered by the conv_gw aquaplanet
# longrun (longrun_aquaplanet_allsky_progedmf_0M_conv_gw.yml).

@testset "Beres (2004) source spectrum" begin
    FT = Float64

    # Phase speed bins: -100 to +100 m/s in steps of 2 m/s
    dc = FT(4.0)
    cmax = FT(100.0)
    nc = Int(2 * cmax / dc + 1)  # 101 bins
    c = ntuple(n -> FT((n - 1) * dc - cmax), Val(nc))

    N_source = FT(0.01)

    beres = BeresSourceParams{FT}(;
        Q0_threshold = FT(1.157e-4),
        beres_scale_factor = FT(1.0),
        σ_x = FT(4000.0),
        ν_min = FT(2π / (120 * 60)),
        ν_max = FT(2π / (10 * 60)),
        n_ν = 9,
    )

    @testset "Output structure" begin
        Q0 = FT(10.0 / 86400.0)
        h = FT(10000.0)
        u_heat = FT(0.0)

        B = wave_source(c, u_heat, Q0, h, N_source, beres, Val(nc))

        @test B isa NTuple{nc, FT}
        @test length(B) == nc
        @test all(isfinite, B)
        @test !all(iszero, B)
    end

    @testset "Symmetry with U=0" begin
        Q0 = FT(10.0 / 86400.0)
        h = FT(10000.0)
        u_heat = FT(0.0)

        B = wave_source(c, u_heat, Q0, h, N_source, beres, Val(nc))

        mid = (nc + 1) ÷ 2
        for i in 1:(mid - 1)
            @test B[mid - i] ≈ -B[mid + i] atol = 1e-30 rtol = 1e-10
        end
        @test B[mid] == FT(0)
    end

    @testset "Q0² amplitude scaling" begin
        h = FT(10000.0)
        u_heat = FT(0.0)

        Q0_1 = FT(5.0 / 86400.0)
        Q0_2 = FT(10.0 / 86400.0)

        B1 = wave_source(c, u_heat, Q0_1, h, N_source, beres, Val(nc))
        B2 = wave_source(c, u_heat, Q0_2, h, N_source, beres, Val(nc))

        # Sample indices spread across the spectrum (must be ≤ nc)
        test_indices = filter(i -> i <= nc, [5, 10, 15, 20, div(nc, 2) - 5])
        for i in test_indices
            if abs(B1[i]) > 1e-40
                ratio = B2[i] / B1[i]
                @test ratio ≈ 4.0 rtol = 1e-10
            end
        end
    end

    @testset "Scale factor linearity" begin
        Q0 = FT(10.0 / 86400.0)
        h = FT(10000.0)
        u_heat = FT(0.0)

        beres_1x = BeresSourceParams{FT}(;
            beres.Q0_threshold, beres_scale_factor = FT(1.0),
            beres.σ_x, beres.ν_min, beres.ν_max, beres.n_ν,
        )
        beres_3x = BeresSourceParams{FT}(;
            beres.Q0_threshold, beres_scale_factor = FT(3.0),
            beres.σ_x, beres.ν_min, beres.ν_max, beres.n_ν,
        )

        B1 = wave_source(c, u_heat, Q0, h, N_source, beres_1x, Val(nc))
        B2 = wave_source(c, u_heat, Q0, h, N_source, beres_3x, Val(nc))

        for i in 1:nc
            @test B2[i] ≈ 3.0 * B1[i] atol = 1e-40
        end
    end

    @testset "Doppler shift with nonzero U" begin
        Q0 = FT(10.0 / 86400.0)
        h = FT(10000.0)
        u_heat = FT(10.0)

        B = wave_source(c, u_heat, Q0, h, N_source, beres, Val(nc))

        @test all(isfinite, B)

        positive_max = FT(0)
        negative_min = FT(0)
        for i in 1:nc
            if B[i] > positive_max
                positive_max = B[i]
            end
            if B[i] < negative_min
                negative_min = B[i]
            end
        end
        @test positive_max > 0
        @test negative_min < 0
    end

    @testset "Zero Q0 gives zero spectrum" begin
        Q0 = FT(0.0)
        h = FT(10000.0)
        u_heat = FT(0.0)

        B = wave_source(c, u_heat, Q0, h, N_source, beres, Val(nc))

        @test all(iszero, B)
    end

    @testset "Beres 2004 Figure 1 reference (single frequency)" begin
        # Figure 1b: σ_x = 2.5 km, h = 5 km, single freq ν = 0.02 cyc/min
        # J₀ = 0.004 K/s, N = 0.012 s⁻¹, U = 0, L = 1000 km, τ = 1 s
        ν_single = FT(2π * 0.02 / 60)
        Δν_narrow = FT(ν_single * 0.01)

        beres_fig1 = BeresSourceParams{FT}(;
            Q0_threshold = FT(0),
            beres_scale_factor = FT(1.0),
            σ_x = FT(2500.0),
            ν_min = ν_single - Δν_narrow,
            ν_max = ν_single + Δν_narrow,
            n_ν = 5,
        )

        N_fig1 = FT(0.012)
        Q0_fig1 = FT(0.004)
        h_fig1 = FT(5000.0)
        u_heat_fig1 = FT(0.0)

        dc_fine = FT(2.0)
        cmax_fine = FT(60.0)
        nc_fine = Int(2 * cmax_fine / dc_fine + 1)
        c_fine = ntuple(n -> FT((n - 1) * dc_fine - cmax_fine), Val(nc_fine))

        B_fig1 = wave_source(
            c_fine, u_heat_fig1, Q0_fig1, h_fig1, N_fig1,
            beres_fig1, Val(nc_fine),
        )

        @test all(isfinite, B_fig1)

        # With U=0, spectrum should be antisymmetric
        mid = (nc_fine + 1) ÷ 2
        for i in 1:(mid - 1)
            @test B_fig1[mid - i] ≈ -B_fig1[mid + i] atol = 1e-30 rtol = 1e-8
        end

        # Find peak phase speed and value
        max_val = FT(0)
        peak_c = FT(0)
        for i in (mid + 1):nc_fine
            if B_fig1[i] > max_val
                max_val = B_fig1[i]
                peak_c = c_fine[i]
            end
        end

        println("Fig 1b reference test:")
        println("  Peak phase speed: $(peak_c) m/s (paper: ~15 m/s)")
        println("  Peak value: $(max_val)")

        @test 5.0 < peak_c < 40.0
        @test max_val > 0
    end

    @testset "Magnitude comparison with AD spectrum" begin
        Q0 = FT(10.0 / 86400.0)
        h = FT(12000.0)
        u_heat = FT(0.0)

        B = wave_source(c, u_heat, Q0, h, N_source, beres, Val(nc))

        max_B = maximum(abs, B)
        @test max_B > 0
        @test max_B < 1e10

        println("Beres spectrum max amplitude: ", max_B)
    end

    @testset "Different n_ν values (convergence)" begin
        Q0 = FT(10.0 / 86400.0)
        h = FT(10000.0)
        u_heat = FT(0.0)

        beres_5 = BeresSourceParams{FT}(;
            beres.Q0_threshold, beres.beres_scale_factor,
            beres.σ_x, beres.ν_min, beres.ν_max, n_ν = 5,
        )
        beres_9 = beres
        beres_13 = BeresSourceParams{FT}(;
            beres.Q0_threshold, beres.beres_scale_factor,
            beres.σ_x, beres.ν_min, beres.ν_max, n_ν = 13,
        )

        B5 = wave_source(c, u_heat, Q0, h, N_source, beres_5, Val(nc))
        B9 = wave_source(c, u_heat, Q0, h, N_source, beres_9, Val(nc))
        B13 = wave_source(c, u_heat, Q0, h, N_source, beres_13, Val(nc))

        diff_5_9 = sum(abs, B5 .- B9)
        diff_9_13 = sum(abs, B9 .- B13)

        println("Convergence: |B5-B9| = $diff_5_9, |B9-B13| = $diff_9_13")
        if diff_5_9 > 1e-40
            @test diff_9_13 < diff_5_9
        end
    end

    @testset "Invalid n_ν rejected" begin
        @test_throws ErrorException BeresSourceParams{FT}(;
            Q0_threshold = FT(1e-4),
            beres_scale_factor = FT(1.0),
            σ_x = FT(4000.0),
            ν_min = FT(2π / (120 * 60)),
            ν_max = FT(2π / (10 * 60)),
            n_ν = 6,
        )
        @test_throws ErrorException BeresSourceParams{FT}(;
            Q0_threshold = FT(1e-4),
            beres_scale_factor = FT(1.0),
            σ_x = FT(4000.0),
            ν_min = FT(2π / (120 * 60)),
            ν_max = FT(2π / (10 * 60)),
            n_ν = 10,
        )
    end

    @testset "Sign convention: sgn(ĉ) consistent with propagator" begin
        # Test that the Beres source spectrum produces the correct sign
        # pattern for the CAM/WACCM convention: sgn(c_hat) outside the
        # integral, where c_hat = c - u_heat.
        #
        # With u_heat = 5 m/s:
        #   c > u_heat  →  c_hat > 0  →  B(c) > 0  (eastward flux)
        #   c < u_heat  →  c_hat < 0  →  B(c) < 0  (westward flux)
        #   c ≈ u_heat  →  c_hat ≈ 0  →  B(c) = 0  (critical level)
        Q0 = FT(10.0 / 86400.0)
        h = FT(10000.0)
        u_heat = FT(5.0)

        B = wave_source(c, u_heat, Q0, h, N_source, beres, Val(nc))

        mid = (nc + 1) ÷ 2  # c = 0 bin
        # Find the bin closest to c = u_heat
        c_hat_idx = findfirst(i -> c[i] >= u_heat, 1:nc)

        # All bins well above u_heat should have B > 0
        for i in (c_hat_idx + 2):nc
            if abs(B[i]) > 1e-40
                @test B[i] > 0 "Expected B[$(i)] > 0 for c=$(c[i]) > u_heat=$(u_heat), got $(B[i])"
            end
        end

        # All bins well below u_heat should have B < 0
        for i in 1:(c_hat_idx - 2)
            if abs(B[i]) > 1e-40
                @test B[i] < 0 "Expected B[$(i)] < 0 for c=$(c[i]) < u_heat=$(u_heat), got $(B[i])"
            end
        end

        # Verify antisymmetry is broken: spectrum should NOT be symmetric
        # about c=0 when u_heat ≠ 0 (Doppler shift breaks symmetry)
        asymmetry = sum(abs, B[i] + B[nc + 1 - i] for i in 1:mid)
        total = sum(abs, B)
        @test asymmetry / total > 1e-4 "Spectrum should be asymmetric with nonzero u_heat"
    end
end
