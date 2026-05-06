using Test
import ClimaAtmos as CA

const BeresSourceParams = CA.BeresSourceParams
const wave_source = CA.wave_source

# NOTE: The wave_source / Beres spectrum computation is EDMF-mode-agnostic.
# It takes (Q0, h, u_heat, N_source) directly, so these unit tests apply
# identically whether the upstream EDMF is diagnostic or prognostic.
# The prognostic EDMF integration path is covered by the aquaplanet
# longrun (longrun_aquaplanet_allsky_progedmf_0M.yml).

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
                @test B[i] > 0
            end
        end

        # All bins well below u_heat should have B < 0
        for i in 1:(c_hat_idx - 2)
            if abs(B[i]) > 1e-40
                @test B[i] < 0
            end
        end

        # Verify antisymmetry is broken: spectrum should NOT be symmetric
        # about c=0 when u_heat ≠ 0 (Doppler shift breaks symmetry)
        asymmetry = sum(abs, B[i] + B[nc + 1 - i] for i in 1:mid)
        total = sum(abs, B)
        @test asymmetry / total > 1e-4
    end
end

# ============================================================================
# Beres (2004) §4 squall-line spectrum — asymmetric case with u_heat ≠ 0
#
# Uses the Beres (2004) Section 4 squall-line parameters (σ_x = 2500 m,
# h = 5000 m, u_heat = −5 m/s, N = 0.012 s⁻¹, Q₀ = 0.004 K/s) but note
# that our implementation uses a WHITE-NOISE time spectrum, not the CRM-derived
# red-noise spectrum in Beres' Figs. 12–13.  This means:
#   - Peak locations are roughly symmetric in |ĉ| about u_heat, at
#     |ĉ_peak| ≈ 14 m/s (from the U=0 Fig. 1b test above).  Beres' Fig. 13
#     shows asymmetric peaks at ĉ ≈ +25 and −6 because his red-noise spectrum
#     weights lower frequencies more heavily.
#   - The spectrum decays slowly at large |c| instead of cutting off sharply,
#     because white noise excites all frequencies.
#
# Assertions below test the white-noise implementation against physics that
# hold regardless of the time spectrum: sign convention, critical level,
# Doppler-shifted peak symmetry, east > west asymmetry from σ_x.
#
# Does NOT test: vertical propagation/drag (see test_beres_column_drag.jl),
# or full-simulation wiring (see test_beres_squall_line_integration.jl).
# ============================================================================
@testset "Beres 2004 §4 squall-line spectrum (white-noise)" begin
    FT = Float64

    # --- Beres (2004) Section 4 squall-line parameters ---
    σ_x     = FT(2500.0)   # m, narrow convective cell (Fig. 13 / Fig. 1b)
    h       = FT(5000.0)   # m, heating depth
    u_heat  = FT(-5.0)     # m/s, mean wind in heating region
    N       = FT(0.012)    # s⁻¹, buoyancy frequency
    Q0      = FT(0.004)    # K/s, peak heating rate

    # Fine phase-speed grid covering ±60 m/s (dc = 2 m/s, nc = 61)
    # dc=2 keeps nc small enough to avoid ntuple/Val compile-time blowup.
    dc_sq = FT(2.0)
    cmax_sq = FT(60.0)
    nc_sq = Int(2 * cmax_sq / dc_sq + 1)  # 61
    c_sq = ntuple(n -> FT((n - 1) * dc_sq - cmax_sq), Val(nc_sq))

    beres_sq = BeresSourceParams{FT}(;
        Q0_threshold = FT(0),
        beres_scale_factor = FT(1.0),
        σ_x = σ_x,
        ν_min = FT(2π / (120 * 60)),   # period 120 min
        ν_max = FT(2π / (10 * 60)),    # period 10 min
        n_ν = 9,
    )

    B_sq = wave_source(c_sq, u_heat, Q0, h, N, beres_sq, Val(nc_sq))

    # Helper: find index of c closest to a target value
    function closest_idx(target)
        _, idx = findmin(n -> abs(c_sq[n] - target), 1:nc_sq)
        return idx
    end

    # Find eastward peak (max positive B, restricted to c > u_heat)
    i_uheat = closest_idx(u_heat)
    east_max_val = FT(0)
    east_peak_c = FT(0)
    for i in (i_uheat + 1):nc_sq
        if B_sq[i] > east_max_val
            east_max_val = B_sq[i]
            east_peak_c = c_sq[i]
        end
    end

    # Find westward peak (most negative B, restricted to c < u_heat)
    west_min_val = FT(0)
    west_peak_c = FT(0)
    for i in 1:(i_uheat - 1)
        if B_sq[i] < west_min_val
            west_min_val = B_sq[i]
            west_peak_c = c_sq[i]
        end
    end

    # Intrinsic phase speeds at peaks
    east_chat = east_peak_c - u_heat
    west_chat = west_peak_c - u_heat

    println("Squall-line spectrum (white-noise):")
    println("  Eastward peak: c = $(east_peak_c) m/s (ĉ = $(east_chat)), B = $(east_max_val)")
    println("  Westward peak: c = $(west_peak_c) m/s (ĉ = $(west_chat)), B = $(west_min_val)")

    @testset "Peak locations: Doppler-shifted from U=0 reference" begin
        # With U=0, the Fig. 1b test finds peak at c ≈ 14 m/s.  With
        # u_heat = -5, peaks should Doppler-shift to c ≈ u_heat ± 14,
        # i.e. eastward peak near +9, westward near -19.
        # Tolerance: ±5 m/s for quadrature discretisation.
        @test 4.0 < east_peak_c < 16.0
        @test -24.0 < west_peak_c < -14.0
    end

    @testset "Intrinsic peak symmetry (white-noise → |ĉ| symmetric)" begin
        # White-noise time spectrum means the integrand is symmetric in
        # |ν̂|, so the magnitude spectrum |B_0| should be roughly
        # symmetric about u_heat in intrinsic phase speed.
        @test abs(east_chat) ≈ abs(west_chat) atol = 4.0
    end

    @testset "East-west asymmetry (σ_x = 2.5 km)" begin
        # With σ_x = 2.5 km (narrow cell) and u_heat = -5 m/s, the
        # eastward lobe is larger than the westward lobe.  This asymmetry
        # comes from the Gaussian spatial spectrum G_k: eastward waves
        # (c > 0) map to smaller |k| = ν/c than westward waves (c < 0),
        # and the Gaussian G_k ∝ exp(-k²σ_x²/2) favours smaller |k|.
        @test east_max_val > abs(west_min_val)
    end

    @testset "Critical level: B small near c = u_heat" begin
        # At c = u_heat, ĉ = 0 and B_0 = 0 exactly.  With dc = 2 m/s,
        # u_heat = -5 doesn't land on a grid point — the nearest point
        # has |ĉ| = 1 m/s, so B is small but nonzero.  Check that it's
        # negligible relative to the peak (< 1%).
        B_at_uheat = B_sq[closest_idx(u_heat)]
        max_B = maximum(abs, B_sq)
        @test abs(B_at_uheat) < 0.01 * max_B
    end

    @testset "Spectrum decays at large |c|" begin
        # White noise means the spectrum doesn't cut off sharply, but it
        # should still decay substantially at large |c|.  Check that the
        # amplitude at the grid edges (|c| = 60) is < 1% of peak.
        max_B = maximum(abs, B_sq)
        edge_B = max(abs(B_sq[1]), abs(B_sq[nc_sq]))
        println("  Edge/peak ratio: $(edge_B / max_B)")
        @test edge_B < 0.01 * max_B
    end

    @testset "Sign convention: sgn(ĉ) consistency" begin
        # B > 0 for c > u_heat (eastward intrinsic propagation)
        # B < 0 for c < u_heat (westward intrinsic propagation)
        noise_floor = 1e-3 * maximum(abs, B_sq)
        for i in 1:nc_sq
            if abs(B_sq[i]) > noise_floor
                c_hat = c_sq[i] - u_heat
                if c_hat > FT(1e-6)
                    @test B_sq[i] > 0
                elseif c_hat < -FT(1e-6)
                    @test B_sq[i] < 0
                end
            end
        end
    end
end
