# ============================================================================
# Beres (2004) NOGW — pure-function unit tests (no simulation build)
#
# This file groups the Beres source tests that call the source/heating kernels
# directly, with no `get_simulation`. They are cheap, CPU-only, and run as part
# of the package test suite (`parameterizations` TEST_GROUP). Three blocks:
#
#   1. Source spectrum            — CA.wave_source (transient Beres spectrum)
#   2. Steady (ν=0) component     — CA.wave_source c≈0 bin + _beres_steady_*
#   3. Canonical latent heating   — CA._beres_latent_heating dual-formulation oracle
#
# Simulation-based Beres coverage lives in test_beres_single_column.jl
# (single-column build) and test_beres_sphere_integration.jl (sphere build).
# ============================================================================

using Test
import ClimaAtmos as CA
import ClimaParams as CP
# Reach Thermodynamics / CloudMicrophysics through ClimaAtmos's own bindings
# instead of direct imports, so this file runs from any environment that loads
# ClimaAtmos (e.g. --project=.buildkite, where neither is a direct dependency).
const TD = CA.TD
const CMP = CA.CMP
const BMT = CA.BMT

const BeresSourceParams = CA.BeresSourceParams
const wave_source = CA.wave_source
const V_hs_sq = CA.V_hs_sq
const _beres_steady_flux = CA._beres_steady_flux
const _beres_steady_horizontal_const = CA._beres_steady_horizontal_const
const _beres_gs = CA._beres_gs
const _beres_launch_spectrum = CA._beres_launch_spectrum
const _beres_mech_flux = CA._beres_mech_flux
const _beres_b0_band = CA._beres_b0_band
# A profile scaled to unit peak, for the spectrum tests below that compare shapes.
unit_peak(t) = t ./ maximum(t)

# Unit half-sine sin(π·(i−1)/(ng−1)) sampled at `ng` points, as a CONCRETE
# NTuple{ng,FT}. `Val{ng}` keeps the length in the type so `_beres_gs` /
# `_beres_launch_spectrum` dispatch is type-stable and compiles quickly (a runtime
# `ntuple(f, ng)` returns a non-concrete tuple → type-unstable, pathological
# compile). Placement onto [z_bot, z_bot+span] is done by `_beres_gs` via its
# z_bot/span args, so the shape alone lives here.
_halfsine_profile(::Val{ng}, FT) where {ng} =
    ntuple(i -> FT(sin(π * FT(i - 1) / FT(ng - 1))), Val(ng))

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

        @test 12 <= peak_c <= 18

        # The bin at c = 14 m/s against an independent quadrature of Beres et al.
        # (2004) Eqs. 23 and 30, written from the paper rather than from the code:
        #   B_kν = (k²/ν̂²)·G_k·Q0·Q_t·(π/(m h))·sin(m h)/(m² − π²/h²),
        #   F_kν = (1/√(2π))·(ρ0/(L τ))·sgn(ĉ)·√(N²/ν̂² − 1)·|B_kν|²,
        # with G_k² = (σ_x²/2)·exp(−k²σ_x²/2) the transform of the Gaussian cell,
        # m = |k|·√(N²/ν̂² − 1), k = ν/c, Q_t = 1 and ρ0/(Lτ) = 1 (both carried by
        # the scale factor, here 1). The bin holds dc·∫ F_kν·(ν/c²) dν over the
        # frequency band, taken here by the trapezoid rule on 4001 points.
        function beres2004_bin(c, ν_lo, ν_hi; N, h, σ_x, Q0, U, dc, n = 4001)
            integral = zero(c)
            for i in 1:n
                ν = ν_lo + (ν_hi - ν_lo) * (i - 1) / (n - 1)
                weight = (i == 1 || i == n ? 1 / 2 : 1) * (ν_hi - ν_lo) / (n - 1)
                k = ν / c
                ν̂ = ν - k * U
                0 < abs(ν̂) < N || continue
                m = abs(k) * sqrt(N^2 / ν̂^2 - 1)
                G_k = sqrt(σ_x^2 / 2 * exp(-k^2 * σ_x^2 / 2))
                B = (k^2 / ν̂^2) * G_k * Q0 * (π / (m * h)) * sin(m * h) /
                    (m^2 - π^2 / h^2)
                F = 1 / sqrt(2π) * sqrt(N^2 / ν̂^2 - 1) * abs(B)^2
                integral += weight * F * ν / c^2
            end
            return sign(c - U) * dc * integral
        end
        n14 = findfirst(==(FT(14)), c_fine)
        B14_reference = beres2004_bin(
            FT(14), ν_single - Δν_narrow, ν_single + Δν_narrow;
            N = N_fig1, h = h_fig1, σ_x = FT(2500), Q0 = Q0_fig1, U = u_heat_fig1,
            dc = dc_fine,
        )
        println("  B0(c = 14 m/s): $(B_fig1[n14]) (independent quadrature: $B14_reference)")
        # The quadrature itself, as computed when this test was written.
        @test B14_reference ≈ 10.4367 rtol = 1e-4
        @test B_fig1[n14] ≈ B14_reference rtol = 1e-6
        @test max_val == B_fig1[n14]
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

    @testset "Depth averaging (n_h_avg > 1) of the transient half-sine bins" begin
        # With n_h_avg = 3 and Δh_frac = 0.1 every transient bin is the mean of the
        # bins for the depths 0.9h, h and 1.1h. The steady bin is switched off, so
        # only the transient path is compared.
        Q0 = FT(10.0 / 86400.0)
        h = FT(8000.0)
        u_heat = FT(3.0)
        params_with(; kw...) = BeresSourceParams{FT}(;
            (n => getfield(beres, n) for n in fieldnames(typeof(beres)))...,
            beres_steady_source = false,
            kw...,
        )
        averaged = wave_source(
            c, u_heat, Q0, h, N_source, params_with(; n_h_avg = 3, Δh_frac = FT(0.1)),
            Val(nc),
        )
        single(h_i) = wave_source(c, u_heat, Q0, h_i, N_source, params_with(), Val(nc))
        mean_of_three =
            (
                collect(single(FT(0.9) * h)) .+ collect(single(h)) .+
                collect(single(FT(1.1) * h))
            ) ./ 3
        @test collect(averaged) ≈ mean_of_three rtol = 1e-10
        @test !(collect(averaged) ≈ collect(single(h)))   # the averaging matters
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
# Does NOT test: vertical propagation/drag (see test_beres_single_column.jl),
# or full-simulation wiring (see test_beres_sphere_integration.jl).
# ============================================================================
@testset "Beres 2004 §4 squall-line spectrum (white-noise)" begin
    FT = Float64

    # --- Beres (2004) Section 4 squall-line parameters ---
    σ_x = FT(2500.0)   # m, narrow convective cell (Fig. 13 / Fig. 1b)
    h = FT(5000.0)   # m, heating depth
    u_heat = FT(-5.0)     # m/s, mean wind in heating region
    N = FT(0.012)    # s⁻¹, buoyancy frequency
    Q0 = FT(0.004)    # K/s, peak heating rate

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
    println(
        "  Eastward peak: c = $(east_peak_c) m/s (ĉ = $(east_chat)), B = $(east_max_val)",
    )
    println(
        "  Westward peak: c = $(west_peak_c) m/s (ĉ = $(west_chat)), B = $(west_min_val)",
    )

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

# ============================================================================
# Steady (ν=0) stationary-mechanical component of the Beres source
# (Beres 2004 Eqs. 31–34), normalization per the (30)→(32) rederivation. The
# reference grid and H value hard-coded below (V_hs_sq, β, one-sided H, F_T
# continuous/discrete, R table, L-invariance) were computed offline from the
# closed-form expressions in the docstrings of `_beres_steady_flux` and
# `_beres_steady_horizontal_const` with H counting k > 0 only; H now counts
# both signs of k, so they are multiplied by 2 here.
# ============================================================================
@testset "Beres steady (ν=0) source" begin
    FT = Float64

    dc = FT(4.0)
    cmax = FT(100.0)
    nc = Int(2 * cmax / dc + 1)          # 51 bins, exact c=0 bin present
    c = ntuple(n -> FT((n - 1) * dc - cmax), Val(nc))
    n_zero = clamp(round(Int, cmax / dc) + 1, 1, nc)
    @test c[n_zero] == FT(0)             # grid has an exact c=0 bin

    N_source = FT(0.012)

    # Steady-source constants (code defaults).
    σx = FT(4000.0)
    Lsys = FT(1.0e6)
    dcf = FT(1.0)
    νmin = FT(8.727e-4)
    sf = FT(2.0e-6)

    # Identical params except the steady flag.
    kw = (;
        Q0_threshold = FT(1.157e-4),
        beres_scale_factor = sf,
        σ_x = σx,
        ν_min = νmin,
        ν_max = FT(1.047e-2),
        n_ν = 9,
    )
    beres_off = BeresSourceParams{FT}(; kw..., beres_steady_source = false)
    beres_on = BeresSourceParams{FT}(;
        kw...,
        beres_steady_source = true,
        beres_steady_dc_frac = dcf,
        beres_L_system = Lsys,
    )

    # Convenience wrapper matching beres_on's settings.
    sflux(U, h, Q0; nha = 1) =
        _beres_steady_flux(U, N_source, h, Q0, sf, nha, FT(0.1), σx, Lsys, dcf, νmin)

    @testset "V_hs_sq isolated shape factor" begin
        h = FT(6000.0)
        # Finite at the resonance m = π/h, equal to h²/4.
        @test V_hs_sq(FT(π) / h, h) ≈ h^2 / 4 rtol = 1e-12
        # No NaN/Inf across a sweep through resonance.
        for m in range(FT(0.3) * FT(π) / h, FT(3) * FT(π) / h; length = 200)
            @test isfinite(V_hs_sq(m, h))
        end
        # Identity to the transient amplitude R:
        #   R² == V_hs_sq(m,h) · m² / (N² − ν̂²)²
        N2 = N_source^2
        ν_hat = FT(0.5) * N_source
        for m in range(FT(0.3) * FT(π) / h, FT(3) * FT(π) / h; length = 50)
            abs(m - FT(π) / h) < 1e-6 && continue
            m_h = m * h
            δ = m_h - FT(π)
            sinc_δ = abs(δ) < FT(1e-10) ? FT(1) : sin(δ) / δ
            R = FT(π) * m * h * (-sinc_δ / (m_h + FT(π))) / (N2 - ν_hat^2)
            @test R^2 ≈ V_hs_sq(m, h) * m^2 / (N2 - ν_hat^2)^2 rtol = 1e-12
        end
    end

    @testset "Two-sided horizontal constant H" begin
        # Closed-form H matches the offline-computed quadrature reference value
        # (one-sided, times 2).
        H = _beres_steady_horizontal_const(σx, Lsys)
        @test H > 0
        @test H ≈ 2 * 2.993366e7 rtol = 1e-5
        # L-invariance bug-detector: H grows only logarithmically with L (×~2.23
        # over ×100 in L), and the steady flux's ENTIRE L-dependence is via H.
        H100 = _beres_steady_horizontal_const(σx, FT(100) * Lsys)
        @test H100 / H < 3.0                       # logarithmic, not power-law
        @test H100 / H ≈ 2.2307 rtol = 1e-3
        f1 = sflux(FT(15.0), FT(5000.0), FT(5e-5))
        f100 = _beres_steady_flux(
            FT(15.0), N_source, FT(5000.0), FT(5e-5), sf, 1, FT(0.1),
            σx, FT(100) * Lsys, dcf, νmin,
        )
        @test f100 / f1 ≈ H100 / H rtol = 1e-12    # no second L beyond H
    end

    @testset "Bit-identical when flag OFF; steady only touches c≈0 bin" begin
        for (Q0, h, u_heat) in [
            (FT(5e-5), FT(6000.0), FT(15.0)),
            (FT(2e-5), FT(8000.0), FT(-12.0)),
            (FT(1e-4), FT(5000.0), FT(0.0)),
            (FT(3e-5), FT(10000.0), FT(25.0)),
        ]
            B_off = wave_source(c, u_heat, Q0, h, N_source, beres_off, Val(nc))
            B_on = wave_source(c, u_heat, Q0, h, N_source, beres_on, Val(nc))
            # Transient bins are bit-identical away from c=0 (exact ==).
            for n in 1:nc
                n == n_zero && continue
                @test B_on[n] == B_off[n]
            end
            # The c=0 bin is transient-zero, and ON adds the steady flux there.
            @test B_off[n_zero] == FT(0)
            @test B_on[n_zero] == sflux(u_heat, h, Q0)
        end
    end

    @testset "Graceful no-op when the grid has no exact c=0 bin" begin
        # cmax/dc not an integer ⇒ no c[n] == 0. The steady source has no config
        # switch and defaults ON, so its deposition is gated by the grid alone:
        # on such a grid it must silently produce nothing extra (not corrupt a
        # nonzero edge bin). beres_on here is identical to the c=0-bin case above.
        dc_bad = FT(4.0)
        cmax_bad = FT(98.0)                  # 98/4 = 24.5 ⇒ no c=0 bin
        nc_bad = Int(floor(2 * cmax_bad / dc_bad + 1))
        c_bad = ntuple(n -> FT((n - 1) * dc_bad - cmax_bad), Val(nc_bad))
        @test all(!=(FT(0)), c_bad)          # confirm there is genuinely no c=0 bin
        for (Q0, h, u_heat) in [
            (FT(5e-5), FT(6000.0), FT(15.0)),
            (FT(1e-4), FT(5000.0), FT(-20.0)),
        ]
            B_off = wave_source(c_bad, u_heat, Q0, h, N_source, beres_off, Val(nc_bad))
            B_on = wave_source(c_bad, u_heat, Q0, h, N_source, beres_on, Val(nc_bad))
            # Steady-ON is bit-identical to steady-OFF everywhere: no deposition.
            for n in 1:nc_bad
                @test B_on[n] == B_off[n]
            end
        end
    end

    @testset "Reference grid match (Eqs 31–34, two-sided H + DC weight)" begin
        # N=0.012, Q0=5e-5, σ_x=4000, sf=2e-6, L=1e6, dc_frac=1, ν_min=8.727e-4, n_h_avg=1.
        Q0 = FT(5e-5)
        ref = [          # (U, h, F_steady_bin) computed offline from Eqs 31–34
            (FT(5.0), FT(6000.0), -2.951007969339e-07),
            (FT(10.0), FT(6000.0), -5.517182327405e-07),
            (FT(15.0), FT(6000.0), -2.615270829043e-06),
            (FT(20.0), FT(6000.0), -3.954343639577e-06),
            (FT(30.0), FT(6000.0), -1.543708984643e-06),
            (FT(-15.0), FT(6000.0), 2.615270829043e-06),
            (FT(15.0), FT(5000.0), -4.838131967350e-06),
        ]
        for (U, h, fref) in ref
            @test sflux(U, h, Q0) ≈ 2 * FT(fref) rtol = 1e-10
        end
    end

    @testset "Steady flux = ν → 0 limit of the transient integrand, both signs of k" begin
        # The steady term carries the code's own transient spectral density down to
        # ν = 0 over the width dc_frac·ν_min. At ν → 0 the intrinsic frequency is
        # ν̂ = −k·U, and the density (as in `_beres_spectrum_single_h`, before the
        # change of variables to c), integrated over |k| > k_min = 2π/L on both
        # sides of k = 0, must equal the steady flux. With the hydrostatic
        # m₀ = N/|U| (and ν̂ ≪ N) that the steady term takes, they agree to the
        # quadrature error; with the exact m² = N²/U² − k², to within a few per cent.
        for (U, h) in ((FT(3.0), FT(8000.0)), (FT(-3.0), FT(6000.0)))
            Q0 = FT(5e-5)
            N = N_source
            k_min = 2 * FT(π) / Lsys
            function density(k)
                ν̂ = -k * U
                abs(ν̂) >= N && return FT(0)
                m² = k^2 * (N^2 / ν̂^2 - 1)
                m = sqrt(m²)
                Gk² = σx^2 / 2 * exp(-k^2 * σx^2 / 2)
                R² = V_hs_sq(m, h) * m² / (N^2 - ν̂^2)^2
                return 1 / sqrt(2 * FT(π)) * sqrt(N^2 - ν̂^2) / abs(ν̂) * Q0^2 * Gk² * R²
            end
            density_hydrostatic(k) =
                1 / sqrt(2 * FT(π)) * Q0^2 * σx^2 / 2 * exp(-k^2 * σx^2 / 2) *
                V_hs_sq(N / abs(U), h) / (N * abs(k) * abs(U)^3)
            # Integrate in log k over k_min < |k| < 20/σx, on both sides of k = 0.
            lk = range(log(k_min), log(20 / σx); length = 40001)
            integral(f) =
                sum(exp(l) * (f(exp(l)) + f(-exp(l))) for l in lk) * step(lk)
            one_side = sum(exp(l) * density(exp(l)) for l in lk) * step(lk)
            other_side = sum(exp(l) * density(-exp(l)) for l in lk) * step(lk)
            @test other_side ≈ one_side rtol = 1e-10       # both signs carry the same
            weight = sf * dcf * νmin
            @test abs(sflux(U, h, Q0)) ≈ weight * integral(density_hydrostatic) rtol = 1e-3
            @test abs(sflux(U, h, Q0)) ≈ weight * integral(density) rtol = 0.06
        end
    end

    @testset "U→0 limit: flux→0, finite, no NaN" begin
        h = FT(6000.0)
        Q0 = FT(5e-5)
        # Hard guard at |U|<1e-6.
        @test sflux(FT(1e-7), h, Q0) == FT(0)
        @test sflux(FT(0.0), h, Q0) == FT(0)
        # Decreasing magnitude toward 0 as U shrinks; always finite.
        mags = FT[]
        for U in (FT(10.0), FT(1.0), FT(1e-2), FT(1e-4))
            f = sflux(U, h, Q0)
            @test isfinite(f)
            push!(mags, abs(f))
        end
        @test issorted(mags; rev = true)
    end

    @testset "Near-resonance is finite (n_h_avg = 1 and 5)" begin
        h = FT(6000.0)
        Q0 = FT(5e-5)
        U_res = N_source * h / FT(π)   # m₀ = N/|U| = π/h exactly
        @test isfinite(sflux(U_res, h, Q0; nha = 1))
        @test isfinite(sflux(U_res, h, Q0; nha = 5))
    end

    @testset "Sign: steady flux decelerates U" begin
        h = FT(6000.0)
        Q0 = FT(5e-5)
        # ON at c=0 must be negative for U>0 (opposes eastward flow), positive for U<0.
        for U in (FT(20.0), FT(8.0))
            B_on = wave_source(c, U, Q0, h, N_source, beres_on, Val(nc))
            @test B_on[n_zero] < 0
        end
        for U in (FT(-20.0), FT(-8.0))
            B_on = wave_source(c, U, Q0, h, N_source, beres_on, Val(nc))
            @test B_on[n_zero] > 0
        end
    end
end

# ============================================================================
# Beres canonical latent-heating — pointwise dual-formulation oracle
#
# The shipped in-cloud heating (when `nogw_beres_heating_latent: true`) is
#   Q_lat = (1/cp⁽ʲ⁾) [ L_v·(dq_lcl + dq_rai) + L_s·(dq_icl + dq_sno) ]
# i.e. CA._beres_latent_heating, built from the CloudMicrophysics 1-moment
# *aggregated* phase-change tendencies (the tested `_aggregate_tendencies`). In the
# model those are the tendencies it applies to the draft: the bulk set averaged over
# the step and the implicit set at its instantaneous rate. Here both come from
# instantaneous rates of the same state (`shipped_latent`), so the oracle below,
# built from the instantaneous per-process terms, must match.
#
# An integral-only check cannot localize a sign/coefficient bug in a small
# melt/freeze term — such a bug preserves the column integral while corrupting
# the vertical SHAPE, which is the entire reason for the change (Beres-G
# consumes the profile). So this test computes Q_lat a SECOND, independent way —
# the explicit per-process Σ_p L_p R_p sum over the verbose source terms — and
# requires agreement at EVERY state to ~machine precision.
#
# The two forms are algebraically identical (the reservoir grouping is just the
# explicit sum regrouped), so they must match to round-off. A mismatch localizes
# a sign or selection error in the explicit oracle below; a clean match confirms
# the shipped reservoir form picks up every phase change with the right sign —
# including the freezing terms (S_accr_lcl_sno_cold, S_accr_rai_sno_cold) and the
# warm-arm melt (S_accr_rai_sno_warm) that a naive hand-selection omits.
# ============================================================================

# The shipped heating for one state: CA._beres_latent_heating applied to the
# instantaneous 1M tendencies of the bulk and the implicit parameter sets, with the
# condensate floored at zero as the model does before calling CM BMT.
function shipped_latent(mp_bulk, mp_imp, thp, ρ, T, w, q_tot, q_lcl, q_icl, q_rai, q_sno)
    FT = typeof(ρ)
    q = max.(zero(FT), (q_lcl, q_icl, q_rai, q_sno))
    q_tot = max(zero(FT), q_tot, sum(q))
    rates(mp) = BMT.bulk_microphysics_tendencies(
        BMT.Instantaneous(), BMT.Microphysics1Moment(), mp, thp, ρ, T, w, q_tot, q...,
    )
    return CA._beres_latent_heating(thp, q_tot, q..., rates(mp_bulk), rates(mp_imp))
end

# Independent oracle: the explicit canonical sum Q_lat = (1/cp) Σ_p L_p R_p,
# assembled from the individual verbose source terms with hand-assigned signs
# and the COMPLETE latent-heat-releasing selection (audited against
# CloudMicrophysics `_aggregate_tendencies`). L_f is derived as L_s − L_v so the
# two forms share identical constants and must agree to round-off.
function Q_lat_explicit(
    mp, thp, ρ, T, q_tot, q_lcl, q_icl, q_rai, q_sno;
    w = zero(ρ),
)
    FT = typeof(ρ)
    src = BMT.bulk_microphysics_tendencies(
        BMT.InstantaneousVerbose(),
        BMT.Microphysics1Moment(),
        mp,
        thp,
        ρ,
        T,
        w,
        q_tot,
        q_lcl,
        q_icl,
        q_rai,
        q_sno,
    )
    Lv = TD.Parameters.LH_v0(thp)
    Ls = TD.Parameters.LH_s0(thp)
    Lf = Ls - Lv
    cp = TD.cp_m(thp, q_tot, q_lcl, q_icl)

    # vapor ↔ liquid (condensation/evaporation), +L_v
    q_vap_liq = Lv * (src.S_phase_change_vap_lcl + src.S_phase_change_vap_rai)
    # vapor ↔ ice (deposition/sublimation), +L_s
    q_vap_ice = Ls * (src.S_phase_change_vap_icl + src.S_phase_change_vap_sno)
    # liquid → ice (freezing), +L_f
    freeze =
        src.S_accr_lcl_sno_cold +
        src.S_accr_freeze_icl_rai +
        src.S_accr_rai_sno_cold
    # ice → liquid (melting), −L_f
    melt =
        src.S_melt_icl_lcl +
        src.S_melt_sno_rai +
        src.S_accr_melt_lcl_sno +
        src.S_accr_melt_rai_sno +
        src.S_accr_rai_sno_warm
    q_liq_ice = Lf * (freeze - melt)

    return (q_vap_liq + q_vap_ice + q_liq_ice) / cp
end

# Source terms that must be exercised (nonzero in at least one swept state) for
# the oracle to actually cover the freeze/melt sign handling.
const FREEZE_MELT_TERMS = (
    :S_accr_lcl_sno_cold,
    :S_accr_freeze_icl_rai,
    :S_accr_rai_sno_cold,
    :S_melt_icl_lcl,
    :S_melt_sno_rai,
    :S_accr_melt_lcl_sno,
    :S_accr_melt_rai_sno,
    :S_accr_rai_sno_warm,
)

@testset "Beres canonical latent heating: dual-formulation oracle" begin
    for FT in (Float32, Float64)
        @testset "FT = $FT" begin
            toml_dict = CP.create_toml_dict(FT)
            mp = CMP.Microphysics1MParams(
                toml_dict;
                rain_autoconversion = CMP.PrescribedNd(toml_dict),
            )
            # `_beres_latent_heating` sums the tendencies of the bulk and
            # implicit 1M parameter sets. Here we exercise the "no implicit
            # split" case: all processes live on `mp`, and `mp_imp` has every
            # process nulled so its BMT contribution is zero.
            mp_imp = CMP.Microphysics1MParams(
                toml_dict;
                cloud_liquid_formation = nothing,
                cloud_ice_formation = nothing,
                cloud_ice_melt = nothing,
                cloud_liquid_freezing = nothing,
                rain_autoconversion = nothing,
                snow_autoconversion = nothing,
                rain_condensation_evaporation = nothing,
                snow_deposition_sublimation = nothing,
                snow_melt = nothing,
                cloud_liquid_rain_accretion = nothing,
                cloud_liquid_snow_accretion = nothing,
                cloud_ice_rain_accretion = nothing,
                cloud_ice_snow_accretion = nothing,
                rain_snow_accretion = nothing,
            )
            thp = TD.Parameters.ThermodynamicsParameters(toml_dict)

            ρ = FT(0.7)  # kg/m³, mid-troposphere
            # Sweep temperature across the melting layer so cold (freezing),
            # warm (melting) and mixed-phase arms all activate, with all four
            # condensate species present so accretion/melt terms are nonzero.
            Ts = FT.((248, 258, 268, 271, 273, 275, 285, 298))
            qs = (
                # (q_tot, q_lcl, q_icl, q_rai, q_sno)
                (FT(1.5e-2), FT(1.0e-3), FT(8.0e-4), FT(6.0e-4), FT(5.0e-4)),
                (FT(2.0e-2), FT(2.0e-3), FT(2.0e-4), FT(1.0e-3), FT(1.0e-3)),
                (FT(8.0e-3), FT(5.0e-4), FT(1.2e-3), FT(2.0e-4), FT(9.0e-4)),
                (FT(2.5e-2), FT(3.0e-3), FT(0.0), FT(1.5e-3), FT(2.0e-4)),
            )

            exercised = Dict(s => false for s in FREEZE_MELT_TERMS)
            maxrelerr = zero(FT)
            rtol = FT === Float32 ? FT(2e-4) : FT(1e-6)
            atol = FT === Float32 ? FT(1e-9) : FT(1e-10)

            for T in Ts, (q_tot, q_lcl, q_icl, q_rai, q_sno) in qs
                shipped = shipped_latent(
                    mp,
                    mp_imp,
                    thp,
                    ρ,
                    T,
                    FT(0),
                    q_tot,
                    q_lcl,
                    q_icl,
                    q_rai,
                    q_sno,
                )
                oracle =
                    Q_lat_explicit(mp, thp, ρ, T, q_tot, q_lcl, q_icl, q_rai, q_sno)

                # The load-bearing assertion: pointwise agreement to round-off.
                @test isapprox(shipped, oracle; rtol = rtol, atol = atol)
                if abs(oracle) > atol
                    maxrelerr =
                        max(maxrelerr, abs(shipped - oracle) / abs(oracle))
                end

                # Track which freeze/melt terms the sweep actually exercises.
                src = BMT.bulk_microphysics_tendencies(
                    BMT.InstantaneousVerbose(),
                    BMT.Microphysics1Moment(),
                    mp,
                    thp,
                    ρ,
                    T,
                    FT(0),
                    q_tot,
                    q_lcl,
                    q_icl,
                    q_rai,
                    q_sno,
                )
                for s in FREEZE_MELT_TERMS
                    abs(getfield(src, s)) > atol && (exercised[s] = true)
                end
            end

            @info "Q_lat oracle agreement (FT=$FT): max relative error = $maxrelerr"

            # Coverage: the sweep must actually exercise freezing AND melting,
            # otherwise the oracle silently passes without testing those signs.
            n_exercised = count(values(exercised))
            @info "freeze/melt terms exercised (FT=$FT): $n_exercised / $(length(FREEZE_MELT_TERMS))" exercised
            @test exercised[:S_melt_icl_lcl] || exercised[:S_melt_sno_rai]      # some melting
            @test exercised[:S_accr_lcl_sno_cold] ||
                  exercised[:S_accr_rai_sno_cold] ||
                  exercised[:S_accr_freeze_icl_rai]                            # some freezing
        end
    end
end

# The model splits the 1M processes between two parameter sets: the implicit set
# carries condensation and deposition (cloud liquid and ice formation, rain
# evaporation, snow deposition and sublimation), the bulk set everything else, and
# `_beres_latent_heating` sums the two. The oracle above nulls the implicit set, so
# it never checks that sum; here both sets are live, as in the model, the updraft
# rises (w > 0), and the sum must match the explicit oracle on the full set.
const CONDENSATION_PROCESSES = (
    :cloud_liquid_formation,
    :cloud_ice_formation,
    :rain_condensation_evaporation,
    :snow_deposition_sublimation,
)
const OTHER_1M_PROCESSES = (
    :cloud_ice_melt,
    :cloud_liquid_freezing,
    :rain_autoconversion,
    :snow_autoconversion,
    :snow_melt,
    :cloud_liquid_rain_accretion,
    :cloud_liquid_snow_accretion,
    :cloud_ice_rain_accretion,
    :cloud_ice_snow_accretion,
    :rain_snow_accretion,
)

@testset "Beres canonical latent heating: live implicit set, rising updraft" begin
    for FT in (Float32, Float64)
        toml_dict = CP.create_toml_dict(FT)
        autoconversion = CMP.PrescribedNd(toml_dict)
        mp_full = CMP.Microphysics1MParams(toml_dict; rain_autoconversion = autoconversion)
        mp_bulk = CMP.Microphysics1MParams(
            toml_dict;
            rain_autoconversion = autoconversion,
            (p => nothing for p in CONDENSATION_PROCESSES)...,
        )
        mp_imp = CMP.Microphysics1MParams(
            toml_dict;
            (p => nothing for p in OTHER_1M_PROCESSES)...,
        )
        mp_none = CMP.Microphysics1MParams(
            toml_dict;
            (p => nothing for p in (CONDENSATION_PROCESSES..., OTHER_1M_PROCESSES...))...,
        )
        thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
        ρ = FT(0.7)
        rtol = FT === Float32 ? FT(2e-4) : FT(1e-6)
        atol = FT === Float32 ? FT(1e-9) : FT(1e-10)
        implicit_share = zero(FT)
        for w in FT.((0.5, 3)),
            T in FT.((258, 271, 275, 290)),
            q_tot_over_sat in FT.((0.9, 1.05))

            q_sat = TD.q_vap_saturation(thp, T, ρ)
            q = (FT(1e-3), FT(2e-4), FT(5e-4), FT(3e-4))   # q_lcl, q_icl, q_rai, q_sno
            q_tot = q_tot_over_sat * q_sat + sum(q)
            shipped = shipped_latent(mp_bulk, mp_imp, thp, ρ, T, w, q_tot, q...)
            oracle = Q_lat_explicit(mp_full, thp, ρ, T, q_tot, q...; w)
            @test isapprox(shipped, oracle; rtol, atol)
            bulk_only = shipped_latent(mp_bulk, mp_none, thp, ρ, T, w, q_tot, q...)
            implicit_share = max(implicit_share, abs(shipped - bulk_only))
        end
        # Setup: the implicit set carries a real part of the heating.
        @test implicit_share > 1000 * atol
    end
end

@testset "Beres canonical latent heating: physical sign sanity" begin
    FT = Float64
    toml_dict = CP.create_toml_dict(FT)
    mp = CMP.Microphysics1MParams(
        toml_dict;
        rain_autoconversion = CMP.PrescribedNd(toml_dict),
    )
    # Null implicit set; the single-mp sum collapses to `mp`'s contribution.
    mp_imp = CMP.Microphysics1MParams(
        toml_dict;
        cloud_liquid_formation = nothing,
        cloud_ice_formation = nothing,
        cloud_ice_melt = nothing,
        cloud_liquid_freezing = nothing,
        rain_autoconversion = nothing,
        snow_autoconversion = nothing,
        rain_condensation_evaporation = nothing,
        snow_deposition_sublimation = nothing,
        snow_melt = nothing,
        cloud_liquid_rain_accretion = nothing,
        cloud_liquid_snow_accretion = nothing,
        cloud_ice_rain_accretion = nothing,
        cloud_ice_snow_accretion = nothing,
        rain_snow_accretion = nothing,
    )
    thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
    ρ = FT(0.7)

    # Warm, strongly supersaturated, no ice → net condensation → warming.
    T_warm = FT(295)
    q_sat_warm = TD.q_vap_saturation(thp, T_warm, ρ)
    Q_cond = shipped_latent(
        mp,
        mp_imp,
        thp,
        ρ,
        T_warm,
        FT(0),               # w
        FT(2) * q_sat_warm,  # q_tot well above saturation
        FT(1e-3),
        FT(0),
        FT(5e-4),
        FT(0),
    )
    @test Q_cond > 0  # condensation releases latent heat

    # Warm, strongly subsaturated, rain present → net evaporation → cooling.
    Q_evap = shipped_latent(
        mp,
        mp_imp,
        thp,
        ρ,
        T_warm,
        FT(0),                 # w
        FT(0.1) * q_sat_warm,  # very dry
        FT(0),
        FT(0),
        FT(1e-3),
        FT(0),
    )
    @test Q_evap < 0  # evaporation absorbs latent heat
end

@testset "Beres canonical latent heating: construction gate" begin
    # The getter must reject nogw_beres_heating_latent unless the run has BOTH
    # 1-moment microphysics (per-phase rates) AND prognostic_edmfx (per-draft
    # state). The validation reads parsed_args only and throws before `params`
    # is dereferenced, so `nothing` params is safe for the rejection paths.
    base = Dict{String, Any}(
        "non_orographic_gravity_wave" => true,
        "nogw_beres_source" => true,
        "nogw_beres_heating_latent" => true,
        "turbconv" => "prognostic_edmfx",
        "microphysics_model" => "1M",
    )

    # 0M microphysics → reject (no explicit per-phase rates).
    args_0m = merge(base, Dict("microphysics_model" => "0M"))
    @test_throws ErrorException CA.get_non_orographic_gravity_wave_model(
        args_0m,
        nothing,
        Float64,
    )

    # edonly_edmfx → reject (no per-draft prognostic condensate state).
    args_diag = merge(base, Dict("turbconv" => "edonly_edmfx"))
    @test_throws ErrorException CA.get_non_orographic_gravity_wave_model(
        args_diag,
        nothing,
        Float64,
    )

    # The Beres source itself needs the prognostic EDMF drafts: any other
    # turbconv value, or no updraft, is rejected with a clear error.
    beres_base = Dict{String, Any}(
        "non_orographic_gravity_wave" => true,
        "nogw_beres_source" => true,
        "turbconv" => "prognostic_edmfx",
        "updraft_number" => 1,
    )
    for bad in (
        Dict("turbconv" => "edmfx"),
        Dict("turbconv" => "edonly_edmfx"),
        Dict("turbconv" => nothing),
        Dict("updraft_number" => 0),
    )
        @test_throws ErrorException CA.get_non_orographic_gravity_wave_model(
            merge(beres_base, bad),
            nothing,
            Float64,
        )
    end

    # The struct itself defaults heating_latent to false (inert path).
    bsp = CA.BeresSourceParams{Float64}(;
        Q0_threshold = 1.0e-5,
        beres_scale_factor = 2.0e-6,
        σ_x = 4000.0,
        ν_min = 8.727e-4,
        ν_max = 1.047e-2,
        n_ν = 9,
    )
    @test bsp.heating_latent == false
end

# ============================================================================
# Beres-G Extension 1 — generalized vertical shape factor V_G = |gs(m)|²
#
# The sine-transform shape factor replaces the half-sine V_hs at the SAME call
# site (R² = shape_sq·m²/(N²−ν̂²)²), pinned by the algebraic parity
# V_hs_sq(m,h) = |gs_halfsine(m)|² (offline gate 1 = 6.5e-14). Here we test the
# shipped Julia primitives: O(dz²) convergence of |gs|² to V_hs_sq for a sampled
# half-sine, and that the full launch spectrum in sine_transform mode reproduces
# the half-sine spectrum to the quadrature tolerance.
# ============================================================================
@testset "Beres-G Ext 1: V_G sine-transform shape factor" begin
    FT = Float64
    h = FT(5000.0)

    @testset "Gate 2: O(dz²) convergence of |gs|² to V_hs_sq (half-sine)" begin
        # `_beres_gs` is a plain loop over a runtime-indexed NTuple, so specializing
        # it for a large tuple is a slow compile; keep ng modest (≤ 129) — enough
        # to show the O(dz²) ratio. Note the ng values must be a compile-time
        # literal tuple (each ng specializes _beres_gs on NTuple{ng}).
        m_test = FT(1.7e-3)               # away from resonance π/h
        Vref = V_hs_sq(m_test, h)
        e33 =
            abs(_beres_gs(_halfsine_profile(Val(33), FT), FT(0), h, m_test)^2 - Vref) / Vref
        e65 =
            abs(_beres_gs(_halfsine_profile(Val(65), FT), FT(0), h, m_test)^2 - Vref) / Vref
        e129 =
            abs(_beres_gs(_halfsine_profile(Val(129), FT), FT(0), h, m_test)^2 - Vref) /
            Vref
        # Each halving of dz quarters the error (O(dz²) ⇒ ratio → 4).
        @test 3.5 < e33 / e65 < 4.5
        @test 3.5 < e65 / e129 < 4.5
    end

    @testset "Gate 1 (in-source): sine_transform reduces to half-sine" begin
        # Feed a half-sine on [0,h] at the PRODUCTION resolution (N_BERES_PROFILE);
        # the full launch spectrum in sine_transform mode must match the half-sine
        # spectrum to the quadrature tolerance (the algebraic NORMALIZATION parity
        # is exact, 6.5e-14 offline; the residual here is the O(dz²) profile
        # discretization at ng=64, ~4e-4). Use the production ng — a large ng would
        # make specializing `_beres_launch_spectrum` on NTuple{ng} a slow compile.
        dc = FT(4.0)
        cmax = FT(100.0)
        nc = Int(2 * cmax / dc + 1)
        c = ntuple(n -> FT((n - 1) * dc - cmax), Val(nc))
        N_source = FT(0.012)
        Q0 = FT(10.0 / 86400.0)
        u_heat = FT(0.0)

        # Beres-G transforms the heating itself, Q0 times the unit half-sine.
        g = Q0 .* _halfsine_profile(Val(CA.N_BERES_PROFILE), FT)   # 64 points (production)

        beres_hs = BeresSourceParams{FT}(;
            Q0_threshold = FT(0), beres_scale_factor = FT(1.0),
            σ_x = FT(4000.0), ν_min = FT(2π / (120 * 60)),
            ν_max = FT(2π / (10 * 60)), n_ν = 9,
            beres_shape_general = false,
        )
        beres_g = BeresSourceParams{FT}(;
            Q0_threshold = FT(0), beres_scale_factor = FT(1.0),
            σ_x = FT(4000.0), ν_min = FT(2π / (120 * 60)),
            ν_max = FT(2π / (10 * 60)), n_ν = 9,
            beres_shape_general = true,
        )

        # span = z_top - z_bot = h here (profile on [0, h]); passed to the sine
        # transform (the arg after z_bot).
        B_hs = _beres_launch_spectrum(
            c, u_heat, Q0, h, N_source, FT(0), h, nothing, FT(0), beres_hs,
            Val(nc),
        )
        # One sample per bin: the half-sine path takes the bin-centre value, so the
        # shape factors are compared at the same phase speeds.
        B_g = _beres_launch_spectrum(
            c, u_heat, Q0, h, N_source, FT(0), h, g, FT(0), beres_g, Val(nc);
            n_bin_samples = Val(1),
        )

        peak = maximum(abs, B_hs)
        maxabs = maximum(abs.(B_g .- B_hs))
        @test peak > 0
        @test maxabs / peak < 2e-3          # ng=64 → quadrature-limited match
    end

    @testset "shape_general=false with a profile present stays half-sine" begin
        # The flag, not the presence of a profile, selects the shape factor: a
        # Beres-G struct with shape_general=false and a gathered profile must
        # produce the EXACT half-sine spectrum (so half_sine runs are unaffected).
        dc = FT(4.0)
        cmax = FT(100.0)
        nc = Int(2 * cmax / dc + 1)
        c = ntuple(n -> FT((n - 1) * dc - cmax), Val(nc))
        N_source = FT(0.01)
        Q0 = FT(8.0 / 86400.0)
        u_heat = FT(7.0)
        g = _halfsine_profile(Val(64), FT)
        beres_off = BeresSourceParams{FT}(;
            Q0_threshold = FT(0), beres_scale_factor = FT(1.0),
            σ_x = FT(4000.0), ν_min = FT(2π / (120 * 60)),
            ν_max = FT(2π / (10 * 60)), n_ν = 9, beres_shape_general = false,
        )
        B_with = _beres_launch_spectrum(
            c, u_heat, Q0, FT(8000.0), N_source, FT(2000.0), FT(8000.0), g,
            FT(0), beres_off, Val(nc),
        )
        B_plain = wave_source(c, u_heat, Q0, FT(8000.0), N_source, beres_off, Val(nc))
        for n in 1:nc
            @test B_with[n] == B_plain[n]    # bit-identical: profile is ignored
        end
    end

    @testset "Steady flux with a gathered profile vanishes as U → 0" begin
        # A profile that ends at 0.3 at the window top, a jump the transform sees
        # as 1/m₀, would give a steady flux growing like 1/|U| as U → 0. Beyond
        # the wavenumber the profile resolves, m₀·span/(ng−1) > π, the flux is
        # zero, so it decreases toward zero.
        ng = CA.N_BERES_PROFILE
        g = ntuple(
            i -> FT(0.7 * sin(π * (i - 1) / (ng - 1)) + 0.3 * (i - 1) / (ng - 1)),
            Val(ng),
        )
        z_bot = FT(2000)
        span = FT(10000)
        N = FT(0.012)
        steady(U) = _beres_steady_flux(
            U, N, span, FT(5e-5), FT(2e-6), 1, FT(0.1), FT(4000), FT(1e6),
            FT(1), FT(8.727e-4), true, g, z_bot, span,
        )
        mags = [abs(steady(U)) for U in (FT(1), FT(0.1), FT(0.01))]
        @test all(isfinite, mags)
        @test mags[1] > 0                    # m₀·dz_gather ≈ 1.9: resolved
        @test issorted(mags; rev = true)
        @test mags[3] == 0
        # Beyond the resolved wavenumber, exactly zero; the transform there is not.
        U_unresolved = N * (span / (ng - 1)) / FT(3.5)
        @test steady(U_unresolved) == 0
        @test CA._beres_gs(g, z_bot, span, N / U_unresolved) != 0
    end
end

# ============================================================================
# Beres-G Extension 3 — mechanical / obstacle steady (ν=0) source β_mech
#
# τ_mech ~ ρ0 U w_b²/N (eq:tau_mech): quadratic in w_b, LINEAR in U (vanishes as
# U→0 — we implement this, NOT eq:beta_total's divergent 1/U form). Carried
# through the same change of variables as the thermal steady term, into the c≈0
# bin; relative size set by the mechanical ν=0 weight, not the transient
# calibration; no heating shape factor.
# ============================================================================
@testset "Beres-G Ext 3: mechanical steady source β_mech" begin
    FT = Float64
    N_source = FT(0.012)
    σ_x = FT(4000.0)
    L_system = FT(1.0e6)
    sf = FT(2.0e-6)
    mw = FT(1.0)

    mech(U, w; weight = mw) =
        _beres_mech_flux(U, N_source, w, sf, weight, σ_x, L_system)

    @testset "Gate 5 (scaling): τ_mech ~ ρ0 U w_b²/N functional form" begin
        U = FT(15.0)
        w = FT(2.0)
        f0 = mech(U, w)
        # quadratic in w_b
        @test mech(U, FT(2) * w) ≈ FT(4) * f0 rtol = 1e-12
        # linear in |U| (sign flips, magnitude doubles)
        @test mech(FT(2) * U, w) ≈ FT(2) * f0 rtol = 1e-12
        # ∝ 1/N: a source with 2N has half the magnitude
        f_2N = _beres_mech_flux(U, FT(2) * N_source, w, sf, mw, σ_x, L_system)
        @test abs(f_2N) ≈ abs(f0) / 2 rtol = 1e-12
        # ∝ weight (the ν=0 mechanical weight)
        @test mech(U, w; weight = FT(3) * mw) ≈ FT(3) * f0 rtol = 1e-12
    end

    @testset "Gate 6: U→0 limit (vanishes, not the 1/U divergence)" begin
        w = FT(2.0)
        @test mech(FT(1e-7), w) == FT(0)         # hard guard
        @test mech(FT(0.0), w) == FT(0)
        mags = FT[]
        for U in (FT(20.0), FT(5.0), FT(1.0), FT(1e-2), FT(1e-3))
            f = mech(U, w)
            @test isfinite(f)
            push!(mags, abs(f))
        end
        @test issorted(mags; rev = true)        # monotonically → 0 as U → 0
    end

    @testset "Sign: β_mech opposes U (decelerating)" begin
        w = FT(3.0)
        @test mech(FT(20.0), w) < 0             # U>0 ⇒ westward (negative)
        @test mech(FT(-20.0), w) > 0            # U<0 ⇒ eastward (positive)
        @test mech(FT(15.0), FT(0.0)) == FT(0)  # no obstacle ⇒ no flux
    end

    @testset "Gate 8: β_tot toggles with mechanical_source; lands at c≈0" begin
        dc = FT(4.0)
        cmax = FT(100.0)
        nc = Int(2 * cmax / dc + 1)
        c = ntuple(n -> FT((n - 1) * dc - cmax), Val(nc))
        n_zero = clamp(round(Int, cmax / dc) + 1, 1, nc)
        @test c[n_zero] == FT(0)
        U = FT(15.0)
        Q0 = FT(5e-5)
        h = FT(6000.0)
        w = FT(2.5)
        z_bot = FT(3000.0)
        g = _halfsine_profile(Val(64), FT)

        kw = (; Q0_threshold = FT(0), beres_scale_factor = sf, σ_x = σ_x,
            ν_min = FT(8.727e-4), ν_max = FT(1.047e-2), n_ν = 9,
            beres_steady_source = true, beres_L_system = L_system,
            beres_mech_weight = mw)
        beres_nomech = BeresSourceParams{FT}(; kw..., beres_mechanical_source = false)
        beres_mech = BeresSourceParams{FT}(; kw..., beres_mechanical_source = true)

        # span (arg after z_bot) = z_top - z_bot = h (profile on [z_bot, z_bot+h]).
        B_no = _beres_launch_spectrum(
            c,
            U,
            Q0,
            h,
            N_source,
            z_bot,
            h,
            g,
            w,
            beres_nomech,
            Val(nc),
        )
        B_me = _beres_launch_spectrum(
            c,
            U,
            Q0,
            h,
            N_source,
            z_bot,
            h,
            g,
            w,
            beres_mech,
            Val(nc),
        )

        # Only the c≈0 bin differs; the difference equals β_mech.
        for n in 1:nc
            n == n_zero && continue
            @test B_me[n] == B_no[n]
        end
        @test (B_me[n_zero] - B_no[n_zero]) ≈ mech(U, w) rtol = 1e-12
        # β_mech is concentrated at low-c (deposited only in the c≈0 bin).
        @test _beres_b0_band(B_me .- B_no, c, FT(15.0), true, false) +
              _beres_b0_band(B_me .- B_no, c, FT(15.0), true, true) ≈
              (B_me[n_zero] - B_no[n_zero]) rtol = 1e-12

        # Relative normalization set by the mechanical weight, not transient α:
        # doubling mech_weight doubles only the c≈0 increment.
        beres_mech2 = BeresSourceParams{FT}(; kw..., beres_mechanical_source = true,
            beres_mech_weight = FT(2) * mw)
        B_me2 = _beres_launch_spectrum(
            c,
            U,
            Q0,
            h,
            N_source,
            z_bot,
            h,
            g,
            w,
            beres_mech2,
            Val(nc),
        )
        @test (B_me2[n_zero] - B_no[n_zero]) ≈ FT(2) * (B_me[n_zero] - B_no[n_zero]) rtol =
            1e-12
    end
end

# ============================================================================
# Non-finite values: the forcing clean-up, its report, and the heating envelope
#
# The forcing is clamped and NaN/Inf replaced by zero once per callback; the
# replaced values are counted and reported once. The envelope moments must not
# turn an infinite in-cloud heating into a NaN height: Inf/Inf = NaN, and a NaN
# z_top would leave the Beres launch level search without a level.
# ============================================================================
@testset "NOGW non-finite values: clean-up, report, envelope" begin
    @testset "limited_gravity_wave_forcing (FT = $FT)" for FT in (Float32, Float64)
        lim = CA.limited_gravity_wave_forcing
        @test lim(FT(NaN)) == FT(0)
        @test lim(FT(Inf)) == FT(0)
        @test lim(-FT(Inf)) == FT(0)
        @test lim(FT(1)) == FT(3e-3)
        @test lim(-FT(1)) == -FT(3e-3)
        @test lim(FT(1e-4)) == FT(1e-4)
        @test lim(-FT(1e-4)) == -FT(1e-4)
    end

    @testset "Non-finite forcing is counted and reported once" begin
        report = CA.warn_nonfinite_gravity_wave_forcing
        uf = [0.0, NaN, 1e-4, Inf]
        vf = [-Inf, 0.0, 0.0, 0.0]
        CA.NOGW_NONFINITE_WARNED[] = false
        n_first = @test_logs (:warn, r"3 NaN or infinite") report(uf, vf)
        @test n_first == 3
        # Counted again, not reported again.
        n_second = @test_logs report(uf, vf)
        @test n_second == 3
        n_finite = @test_logs report([0.0, 1.0], [2.0, 3.0])
        @test n_finite == 0
        CA.NOGW_NONFINITE_WARNED[] = false
    end

    for FT in (Float32, Float64)
        @testset "Heating centroid and depth are finite (FT = $FT)" begin
            centroid = CA._beres_heating_centroid
            depth = CA._beres_heating_depth
            # A half-sine on [2, 12] km: centroid 7 km and depth 10 km from its
            # moments (looser for Float32, where σ² = ∫z²Q/∫Q − z_c² cancels).
            rtol = FT == Float32 ? 1e-2 : 1e-4
            z = collect(range(FT(2000), FT(12000); length = 2001))
            dz = z[2] - z[1]
            Q = sin.(FT(π) .* (z .- z[1]) ./ (z[end] - z[1]))
            I0 = sum(Q) * dz
            I1 = sum(z .* Q) * dz
            I2 = sum(z .^ 2 .* Q) * dz
            z_c = centroid(I0, I1)
            @test z_c ≈ FT(7000) rtol = rtol
            @test depth(I0, I2, z_c) ≈ FT(10000) rtol = rtol
            # An infinite heating gives infinite moments: zero, not NaN.
            inf_c = centroid(FT(Inf), FT(Inf))
            @test inf_c == FT(0)
            @test depth(FT(Inf), FT(Inf), inf_c) == FT(0)
            @test centroid(FT(1), FT(Inf)) == FT(0)
            @test depth(FT(1), FT(Inf), FT(0)) == FT(0)
            # No positive heating, or a NaN moment: zero.
            @test centroid(FT(0), FT(0)) == FT(0)
            @test depth(FT(0), FT(0), FT(0)) == FT(0)
            @test centroid(FT(NaN), FT(NaN)) == FT(0)
            @test depth(FT(NaN), FT(NaN), FT(0)) == FT(0)
        end
    end
end

# ============================================================================
# Beres-G gather: sampling the in-cloud heating on the model levels onto the
# 64-point profile the sine transform reads.
#
# The tests run the gather (`_beres_gather_step`, upward through the levels as
# `column_reduce!` runs it) on the production vertical grid of the Beres runs
# (63 stretched levels to 60 km, 30 m at the bottom), whose spacing grows from
# about 200 m at 2 km to 1.3 km at 15 km.
# ============================================================================
import ClimaCore.Fields as Fields

# Cell-centre heights and thicknesses of the production Beres grid.
const BERES_GRID = let
    grid = CA.ColumnGrid(
        Float64;
        z_elem = 63,
        z_max = 60000.0,
        z_stretch = true,
        dz_bottom = 30.0,
    )
    (; center_space, face_space) = CA.get_spaces(grid)
    z_centre = vec(Array(parent(Fields.coordinate_field(center_space).z)))
    z_face = vec(Array(parent(Fields.coordinate_field(face_space).z)))
    (; z = z_centre, dz = diff(z_face))
end

# The gathered 64-point profile of the heating `Q(z)` on [z_bot, z_top], run
# level by level as the model runs it.
function gather_profile(Q, z_bot, z_top; z_floor = 0.0)
    state = CA._beres_gather_init(Float64, Val(CA.N_BERES_PROFILE))
    for z in BERES_GRID.z
        state = CA._beres_gather_step(
            state,
            (z, Q(z), z_bot, z_top),
            z_floor,
            Val(CA.N_BERES_PROFILE),
        )
    end
    return first(state)
end

# Largest and mean level thickness over the cells that overlap [z_bot, z_top]; the
# model's dz_level is the mean.
function coarsest_spacing(z_bot, z_top)
    (; z, dz) = BERES_GRID
    return maximum(dz[(z .+ dz ./ 2 .> z_bot) .& (z .- dz ./ 2 .< z_top)])
end
function mean_spacing(z_bot, z_top)
    (; z, dz) = BERES_GRID
    overlap = (z .+ dz ./ 2 .> z_bot) .& (z .- dz ./ 2 .< z_top)
    return sum(dz[overlap]) / count(overlap)
end

# ∫ f(z) sin(m z) dz over [a, b] by the composite Simpson rule.
function sine_transform_quadrature(f, a, b, m; n = 20000)
    zz = range(a, b; length = n + 1)
    w = [i == 0 || i == n ? 1 : (isodd(i) ? 4 : 2) for i in 0:n]
    return sum(w .* f.(zz) .* sin.(m .* zz)) * step(zz) / 3
end

@testset "Beres-G gather" begin
    @testset "Linear interpolation between levels, not the nearest level" begin
        # A smooth Gaussian heating lobe. The spectrum weighs the transform as
        # m²·|gs(m)|². The linear interpolant of the heating on the levels has
        # kinks at the levels, so that weight falls off beyond the level
        # resolution, m > π/dz_level; a nearest-level copy has steps, and the
        # weight stays near its peak there, which puts false flux into the slow
        # bins. The resolved part, m ≤ π/(2 dz_level), must match the exact
        # transform. The nearest-level copy, computed here only for contrast,
        # shows the tail check has the power to catch it.
        for (z_0, σ, z_bot, z_top) in (
            (9000.0, 2000.0, 3000.0, 15000.0),
            (8000.0, 1500.0, 3000.0, 14000.0),
        )
            Q(z) = exp(-((z - z_0) / σ)^2 / 2)
            span = z_top - z_bot
            ng = CA.N_BERES_PROFILE
            dz_level = coarsest_spacing(z_bot, z_top)
            profile = gather_profile(Q, z_bot, z_top)
            z_samples = z_bot .+ (0:(ng - 1)) .* (span / (ng - 1))
            nearest = ntuple(ng) do i
                BERES_GRID.z[argmin(abs.(BERES_GRID.z .- z_samples[i]))] |> Q
            end
            weight(g, m) = (m * CA._beres_gs(g, z_bot, span, m))^2
            m_resolved = range(1e-5, π / (2 * dz_level); length = 100)
            m_tail = range(π / dz_level, π * (ng - 1) / span; length = 200)
            exact = [sine_transform_quadrature(Q, z_bot, z_top, m)^2 for m in m_resolved]
            gathered = [CA._beres_gs(profile, z_bot, span, m)^2 for m in m_resolved]
            @test maximum(abs.(gathered .- exact)) < 0.05 * maximum(exact)
            tail_share(g) =
                maximum(m -> weight(g, m), m_tail) /
                maximum(m -> weight(g, m), m_resolved)
            @test tail_share(profile) < 0.02
            @test tail_share(nearest) > 0.1      # the check has power
        end
    end

    @testset "The window starts at the floor, so the profile has no step inside it" begin
        # A half-sine heating on [1, 9] km whose envelope starts at 1 km, below the
        # 2 km floor under which the gather masks the heating. A window from z_bot
        # holds zeros up to 2 km and then jumps to the heating there; the window
        # from the floor holds the heating from its first point on.
        z_floor = 2000.0
        z_bot, z_top = 1000.0, 9000.0
        Q(z) = z_bot <= z <= z_top ? sin(π * (z - z_bot) / (z_top - z_bot)) : 0.0
        z_start = CA._beres_gather_bottom(z_bot, z_top, z_floor)
        @test z_start == z_floor
        profile = gather_profile(Q, z_start, z_top; z_floor)
        @test all(>(0), profile[2:(end - 1)])
        from_z_bot = gather_profile(Q, z_bot, z_top; z_floor)
        i_step = findlast(iszero, from_z_bot[1:(end ÷ 2)])
        # The step it avoids: from zero to over a third of the peak within two
        # gather points (one model level at 2 km).
        @test !isnothing(i_step) && from_z_bot[i_step + 2] > 0.3
        # The bottom stays inside the envelope.
        @test CA._beres_gather_bottom(3000.0, 9000.0, z_floor) == 3000.0
        @test CA._beres_gather_bottom(500.0, 1500.0, z_floor) == 1500.0
    end

    @testset "Edge taper and wavenumber cut keep the slow bins smooth" begin
        # Two heatings with jumps the gather keeps: a half-sine on [2, 14] km
        # whose window starts at a quarter of its depth (5 km), where it is 0.71
        # of its peak; and a coarse staircase, a half-sine held constant over
        # blocks of 8 model levels. A jump makes the sine transform fall off like
        # 1/m, which puts flux next to u_heat, growing like 1/|c − u_heat| and
        # jumping about as u_heat moves across the 0.8 m/s bins. With the edges
        # tapered over the mean level thickness dz_level and m > π/dz_level
        # dropped, the flux within 2 m/s of u_heat is under 1% of the total. The
        # cut itself is what empties the bins closer to u_heat than
        # c_cut = N·dz_level/π (1.8-2.6 m/s here): a wave there has a vertical
        # wavelength 2π|c − u_heat|/N under two level thicknesses, which the
        # gathered profile does not resolve. Within 0.8·c_cut, the taper alone
        # leaves about 1% of the flux, the cut none.
        # How the total moves with u_heat is checked below for the transient bins
        # alone: the steady c = 0 bin still swings with u_heat (it uses
        # |gs(N/|u_heat|)|² of the gathered profile, F04, awaiting a decision),
        # by 13% of the total for the cut half-sine when u_heat moves 0.4 m/s.
        # The untapered, uncut profile, for contrast, puts more than 1% within
        # 2 m/s of u_heat.
        FT = Float64
        z_floor = 2000.0
        half_sine(z) = 2000 <= z <= 14000 ? sin(π * (z - 2000) / 12000) : 0.0
        # The staircase is defined on the model levels, which is all the gather
        # reads: level k takes the half-sine at the mean height of its block.
        z_levels = BERES_GRID.z
        block_value = map(eachindex(z_levels)) do k
            block = (8 * ((k - 1) ÷ 8) + 1):min(8 * ((k - 1) ÷ 8) + 8, length(z_levels))
            half_sine(sum(z_levels[block]) / length(block))
        end
        staircase(zz) = block_value[findfirst(==(zz), z_levels)]
        beres_g = BeresSourceParams{FT}(;
            Q0_threshold = FT(0),
            beres_scale_factor = FT(2e-6),
            σ_x = FT(4000),
            ν_min = FT(8.727e-4),
            ν_max = FT(1.047e-2),
            n_ν = 9,
            beres_shape_general = true,
        )
        dc, cmax = FT(0.8), FT(40)
        nc = Int(2 * cmax / dc + 1)
        c = ntuple(n -> (n - 1) * dc - cmax, Val(nc))
        Q0, h, N = FT(5e-5), FT(12000), FT(0.012)
        spectrum(profile, z_lo, span, u_heat, dz_level) = CA._beres_launch_spectrum(
            c, u_heat, Q0, h, N, z_lo, span, profile, FT(0), beres_g, Val(nc);
            dz_level,
        )
        near_share(B, u_heat; within = 2) =
            sum(abs(B[n]) for n in 1:nc if abs(c[n] - u_heat) < within; init = 0.0) /
            sum(abs, B)
        for (Q, z_lo, z_top) in ((half_sine, 5000.0, 14000.0), (staircase, 2000.0, 14000.0))
            span = z_top - z_lo
            dz_level = mean_spacing(z_lo, z_top)
            c_cut = N * dz_level / π
            raw = unit_peak(gather_profile(Q, z_lo, z_top; z_floor))
            tapered = CA._beres_taper_edges(raw, span, dz_level)
            @test tapered[1] == 0
            @test tapered[end] == 0
            @test maximum(tapered) <= 1
            totals = map((FT(5.3), FT(5.7))) do u_heat
                B = spectrum(tapered, z_lo, span, u_heat, dz_level)
                @test all(isfinite, B)
                @test near_share(B, u_heat) < 0.01
                @test near_share(B, u_heat; within = 0.8 * c_cut) < 1e-4   # the cut
                sum(abs, B)
            end
            B_raw = spectrum(raw, z_lo, span, FT(5.3), FT(0))
            @info "Beres-G taper and cut (window $(z_lo)-$(z_top) m, dz_level = $(round(dz_level)) m): near-u_heat share $(near_share(B_raw, FT(5.3))) untreated, total flux treated/untreated = $(totals[1] / sum(abs, B_raw)), change for +0.4 m/s = $(totals[2] / totals[1] - 1)"
            @test near_share(B_raw, FT(5.3)) > 0.01   # the check has power
            # Taper without the cut: the bins within 0.8·c_cut of u_heat keep flux.
            B_taper_only = spectrum(tapered, z_lo, span, FT(5.3), FT(0))
            @info "  within 0.8 c_cut = $(0.8 * c_cut) m/s: share with cut $(near_share(spectrum(tapered, z_lo, span, FT(5.3), dz_level), FT(5.3); within = 0.8 * c_cut)), taper only $(near_share(B_taper_only, FT(5.3); within = 0.8 * c_cut))"
            @test near_share(B_taper_only, FT(5.3); within = 0.8 * c_cut) > 1e-3
        end
    end
end

# ============================================================================
# Bin width: each transient bin carries the spectral density times the bin
# width dc, and the steady c = 0 bin carries the steady flux, a total. Halving
# dc (same cmax) must then leave the total launched flux and the steady share
# unchanged.
# ============================================================================
@testset "Beres-G bins average the spectrum over their width" begin
    # Each transient Beres-G bin holds the spectral density averaged over the bin,
    # [c − dc/2, c + dc/2], not its value at the bin centre. The Beres-G density
    # oscillates in c near u_heat on scales finer than a bin (its squared sine
    # transform, measured from the ground, oscillates in m with a phase of about
    # 2·m·z_c, z_c the heating centroid, and m ≈ N/|c − u_heat|), so centre values
    # swing as u_heat moves across the bins; averages do not. Profiles as in the
    # edge-taper test above, on the production grid, steady term off.
    FT = Float64
    half_sine(z) = 2000 <= z <= 14000 ? sin(π * (z - 2000) / 12000) : 0.0
    z_levels = BERES_GRID.z
    block_value = map(eachindex(z_levels)) do k
        block = (8 * ((k - 1) ÷ 8) + 1):min(8 * ((k - 1) ÷ 8) + 8, length(z_levels))
        half_sine(sum(z_levels[block]) / length(block))
    end
    staircase(zz) = block_value[findfirst(==(zz), z_levels)]
    kw = (;
        Q0_threshold = FT(0),
        beres_scale_factor = FT(2e-6),
        σ_x = FT(4000),
        ν_min = FT(8.727e-4),
        ν_max = FT(1.047e-2),
        n_ν = 9,
        beres_shape_general = true,
        beres_steady_source = false,
    )
    beres_g = BeresSourceParams{FT}(; kw...)
    dc, cmax = FT(0.8), FT(40)
    nc = Int(2 * cmax / dc + 1)
    c = ntuple(n -> (n - 1) * dc - cmax, Val(nc))
    Q0, h, N = FT(5e-5), FT(12000), FT(0.012)
    boole_w = (FT(7), FT(32), FT(12), FT(32), FT(7))
    dν = (beres_g.ν_max - beres_g.ν_min) / FT(beres_g.n_ν - 1)
    for (Q, z_lo, z_top) in ((half_sine, 5000.0, 14000.0), (staircase, 2000.0, 14000.0))
        span = z_top - z_lo
        dz_level = mean_spacing(z_lo, z_top)
        profile = CA._beres_taper_edges(
            unit_peak(gather_profile(Q, z_lo, z_top; z_floor = 2000.0)),
            span,
            dz_level,
        )
        spectrum(u_heat, samples) = CA._beres_launch_spectrum(
            c, u_heat, Q0, h, N, z_lo, span, profile, FT(0), beres_g, Val(nc);
            dz_level, n_bin_samples = samples,
        )
        # Moving u_heat by 0.4 m/s (half a bin) barely changes the total
        # transient flux with the bin average; with centre values it swings.
        swing(samples) =
            abs(
                sum(abs, spectrum(FT(5.7), samples)) /
                sum(abs, spectrum(FT(5.3), samples)) - 1,
            )
        swing_avg = swing(Val(CA.N_BERES_BIN_SAMPLES))
        swing_centre = swing(Val(1))
        # The bins sum to the integral of the density over c, from a grid 64
        # times finer (centre values there are already bin averages).
        fine_total = sum((-cmax - dc / 2 + dc / 128):(dc / 64):(cmax + dc / 2)) do c_f
            abs(
                CA._beres_transient_bin(
                    # Beres-G takes no Q0 factor (it transforms the profile).
                    c_f, dc / 64, FT(5.3), h, N^2, N, FT(1), beres_g.σ_x^2,
                    beres_g.ν_min, dν, (beres_g.n_ν - 1) ÷ 4, boole_w, true,
                    profile, z_lo, span, FT(π) / dz_level, 1, FT(0.1),
                    beres_g.beres_scale_factor, nothing, Val(1),
                ),
            )
        end
        coarse_avg = sum(abs, spectrum(FT(5.3), Val(CA.N_BERES_BIN_SAMPLES)))
        coarse_centre = sum(abs, spectrum(FT(5.3), Val(1)))
        @info "Beres-G bin averaging (window $(z_lo)-$(z_top) m): +0.4 m/s swing $(swing_avg) averaged, $(swing_centre) centre; bins/fine integral $(coarse_avg / fine_total) averaged, $(coarse_centre / fine_total) centre"
        @test swing_avg < 0.03
        @test coarse_avg ≈ fine_total rtol = 0.01
    end

    @testset "The five-way diagnostic uses the same bins" begin
        # _beres_banded_spectrum shares the per-bin function with the model
        # spectrum, so its band sums equal the model spectrum's.
        beres_s = BeresSourceParams{FT}(; kw..., beres_steady_source = true)
        z_lo, z_top = 2000.0, 14000.0
        dz_level = mean_spacing(z_lo, z_top)
        profile = CA._beres_taper_edges(
            unit_peak(gather_profile(half_sine, z_lo, z_top; z_floor = 2000.0)),
            z_top - z_lo,
            dz_level,
        )
        for u_heat in (FT(5.3), FT(-7.9))
            B = CA._beres_launch_spectrum(
                c, u_heat, Q0, h, N, z_lo, z_top - z_lo, profile, FT(0), beres_s,
                Val(nc); dz_level,
            )
            sums = CA._beres_banded_spectrum(
                c, u_heat, Q0, h, N, z_lo, z_top - z_lo, profile, FT(0), beres_s,
                nc, FT(15), dz_level,
            )
            band_abs(low, east) =
                sum(
                    abs(B[n]) for n in 1:nc if ((abs(c[n]) <= 15) == low) &&
                        ((c[n] > 0) == east); init = 0.0)
            @test sums[5] ≈ band_abs(true, true) rtol = 1e-10
            @test sums[6] ≈ band_abs(true, false) rtol = 1e-10
            @test sums[7] ≈ band_abs(false, true) rtol = 1e-10
            @test sums[8] ≈ band_abs(false, false) rtol = 1e-10
        end
    end
end

@testset "Beres launch flux does not depend on the bin width" begin
    FT = Float64
    beres = BeresSourceParams{FT}(;
        Q0_threshold = FT(0),
        beres_scale_factor = FT(2e-6),
        σ_x = FT(4000),
        ν_min = FT(8.727e-4),
        ν_max = FT(1.047e-2),
        n_ν = 9,
        beres_steady_source = true,
    )
    cmax = FT(40)
    Q0, h, N = FT(5e-5), FT(8000), FT(0.012)
    function totals(dc, u_heat)
        nc = Int(2 * cmax / dc + 1)
        c = ntuple(n -> (n - 1) * dc - cmax, Val(nc))
        B = wave_source(c, u_heat, Q0, h, N, beres, Val(nc))
        n_zero = findfirst(iszero, c)
        moving = sum(abs, B) - abs(B[n_zero])
        return (; moving, steady = abs(B[n_zero]))
    end
    for u_heat in (FT(7.3), FT(-11.1))
        coarse = totals(FT(0.8), u_heat)
        fine = totals(FT(0.4), u_heat)
        @test coarse.moving > 0
        @test coarse.steady > 0
        @test fine.moving ≈ coarse.moving rtol = 0.02
        @test fine.steady == coarse.steady
        @test fine.steady / fine.moving ≈ coarse.steady / coarse.moving rtol = 0.02
    end
end

# ============================================================================
# Launch filter: waves that meet a critical level inside the heating layer, on
# their way from the heating to the launch level, are removed (Beres 2004) or,
# height-resolved, see only the heating above their critical level.
#
# The wind rises linearly from -15 m/s at the bottom of the gather window (2 km)
# to +5 m/s at the launch level just above its top (14 km), so u_heat is about
# -5 m/s and every phase speed in [-15, 5] m/s meets a critical level in the
# layer, the steady c = 0 wave included.
# ============================================================================
@testset "Beres launch filter" begin
    FT = Float64
    ng = CA.N_BERES_PROFILE
    z_lo, z_top, z_launch = FT(2000), FT(14000), FT(14500)
    span = z_top - z_lo
    z_points = z_lo .+ (0:(ng - 1)) .* (span / (ng - 1))
    wind(z) = FT(-15) + FT(20) * (z - z_lo) / (z_launch - z_lo)
    wind_profile = ntuple(i -> wind(z_points[i]), Val(ng))
    wind_range = (FT(-15), FT(5), FT(0), FT(0), FT(5), FT(0))
    g = _halfsine_profile(Val(ng), FT)
    u_heat = FT(-5)
    kw = (;
        Q0_threshold = FT(0),
        beres_scale_factor = FT(2e-6),
        σ_x = FT(4000),
        ν_min = FT(8.727e-4),
        ν_max = FT(1.047e-2),
        n_ν = 9,
        beres_shape_general = true,
    )
    dc, cmax = FT(0.8), FT(40)
    nc = Int(2 * cmax / dc + 1)
    c = ntuple(n -> (n - 1) * dc - cmax, Val(nc))
    n_zero = findfirst(iszero, c)
    filter_of(mode; margin = FT(0), profile = wind_profile) = CA._beres_launch_filter(
        BeresSourceParams{FT}(; kw..., launch_filter = mode, launch_filter_margin = margin),
        wind_range,
        profile,
        Val(1),
    )
    beres_g = BeresSourceParams{FT}(; kw...)
    spectrum(filter; profile = g, beres = beres_g) = _beres_launch_spectrum(
        c, u_heat, FT(5e-5), span, FT(0.012), z_lo, span, profile, FT(0), beres,
        Val(nc); dz_level = FT(600), launch_filter = filter,
    )
    B_plain = spectrum(nothing)
    B_none = spectrum(filter_of(CA.BERES_LAUNCH_FILTER_NONE))
    B_2004 = spectrum(filter_of(CA.BERES_LAUNCH_FILTER_BERES2004))
    B_2004_margin = spectrum(filter_of(CA.BERES_LAUNCH_FILTER_BERES2004; margin = FT(2)))
    B_height = spectrum(filter_of(CA.BERES_LAUNCH_FILTER_HEIGHT_RESOLVED))
    # Bins whose whole width lies inside or outside a band [lo, hi].
    inside(n, lo, hi) = (c[n] - dc / 2 >= lo) && (c[n] + dc / 2 <= hi)
    outside(n, lo, hi) = (c[n] + dc / 2 < lo) || (c[n] - dc / 2 > hi)

    @testset "none leaves the spectrum unchanged" begin
        @test B_none == B_plain
        @test abs(B_plain[n_zero]) > 0          # setup: a steady flux to remove
    end

    @testset "beres2004 removes the wind range over the layer" begin
        @test all(n -> B_2004[n] == 0, filter(n -> inside(n, -15, 5), 1:nc))
        @test B_2004[n_zero] == 0               # the steady c = 0 wave too
        @test all(n -> B_2004[n] == B_plain[n], filter(n -> outside(n, -15, 5), 1:nc))
        @test any(n -> B_plain[n] != 0, filter(n -> inside(n, -15, 5), 1:nc))
    end

    @testset "the margin widens the band" begin
        @test all(n -> B_2004_margin[n] == 0, filter(n -> inside(n, -17, 7), 1:nc))
        widened = filter(n -> inside(n, 5, 7) || inside(n, -17, -15), 1:nc)
        @test !isempty(widened)
        @test all(n -> B_2004[n] != 0, widened)
        @test all(
            n -> B_2004_margin[n] == B_plain[n],
            filter(n -> outside(n, -17, 7), 1:nc),
        )
    end

    @testset "height_resolved keeps the heating above each critical level" begin
        f = filter_of(CA.BERES_LAUNCH_FILTER_HEIGHT_RESOLVED)
        # c = -10 m/s: the wind reaches it at 5 km, so the heating below is masked.
        z_crit = z_lo + (FT(-10) + 15) / 20 * (z_launch - z_lo)
        masked = CA._beres_filter_profile(f, g, FT(-10))
        @test all(i -> masked[i] == 0, findall(z -> z <= z_crit, collect(z_points)))
        above = findall(z -> z > z_crit + span / (ng - 1), collect(z_points))
        @test all(i -> masked[i] == g[i], above)
        # c = 4.9 m/s: the wind reaches it between the window top and the launch
        # level, so nothing is left; c = 6 m/s is never reached, so all is left.
        @test all(iszero, CA._beres_filter_profile(f, g, FT(4.9)))
        @test CA._beres_filter_profile(f, g, FT(6)) == g
        # The margin lowers the critical level of c = -10 m/s (the wind is within
        # 2 m/s of it already at -12 m/s) and widens the band.
        f_margin = filter_of(CA.BERES_LAUNCH_FILTER_HEIGHT_RESOLVED; margin = FT(2))
        @test count(iszero, CA._beres_filter_profile(f_margin, g, FT(-10))) >
              count(iszero, masked)
        @test all(iszero, CA._beres_filter_profile(f_margin, g, FT(6)))
        # In the spectrum: unchanged outside the band, partly kept inside it (more
        # where the critical level is low), and the steady c = 0 wave is built from
        # the heating above the zero-wind level.
        @test all(n -> B_height[n] == B_plain[n], filter(n -> outside(n, -15, 5), 1:nc))
        n_low = argmin(abs.(collect(c) .+ 12))
        n_high = argmin(abs.(collect(c) .- 4))
        @test B_height[n_low] != 0
        @test abs(B_height[n_low] / B_plain[n_low]) >
              abs(B_height[n_high] / B_plain[n_high])
        @test B_height[n_zero] != 0
        B_above_zero_wind =
            spectrum(nothing; profile = CA._beres_filter_profile(f, g, FT(0)))
        @test B_height[n_zero] == B_above_zero_wind[n_zero]
        # With a taper, the kept heating rises from zero over that many gather
        # spacings above the cut instead of jumping.
        f_taper = CA._beres_launch_filter(
            BeresSourceParams{FT}(;
                kw..., launch_filter = CA.BERES_LAUNCH_FILTER_HEIGHT_RESOLVED,
            ),
            wind_range,
            wind_profile,
            Val(1),
            4,
        )
        tapered = CA._beres_filter_profile(f_taper, g, FT(-10))
        i_first = findfirst(!iszero, masked)
        @test tapered[i_first] < masked[i_first]
        @test tapered[i_first + 4] == masked[i_first + 4]
    end

    @testset "the half-sine path uses the Beres (2004) band in both filter modes" begin
        beres_hs = BeresSourceParams{FT}(; kw..., beres_shape_general = false)
        spectrum_hs(filter) = _beres_launch_spectrum(
            c, u_heat, FT(5e-5), span, FT(0.012), z_lo, span, nothing, FT(0),
            beres_hs, Val(nc); launch_filter = filter,
        )
        B_hs_2004 =
            spectrum_hs(filter_of(CA.BERES_LAUNCH_FILTER_BERES2004; profile = nothing))
        B_hs_height =
            spectrum_hs(
                filter_of(CA.BERES_LAUNCH_FILTER_HEIGHT_RESOLVED; profile = nothing),
            )
        @test B_hs_height == B_hs_2004
        @test all(n -> B_hs_2004[n] == 0, filter(n -> -15 <= c[n] <= 5, 1:nc))
    end

    @testset "the five-way diagnostic applies the same filter" begin
        f = filter_of(CA.BERES_LAUNCH_FILTER_HEIGHT_RESOLVED)
        sums = CA._beres_banded_spectrum(
            c, u_heat, FT(5e-5), span, FT(0.012), z_lo, span, g, FT(0), beres_g,
            nc, FT(15), FT(600), Val(CA.N_BERES_BIN_SAMPLES), f,
        )
        @test sums[5] + sums[6] + sums[7] + sums[8] ≈ sum(abs, B_height) rtol = 1e-10
        @test sums[10] ≈ abs(B_height[n_zero]) rtol = 1e-10
    end

    @testset "configuration values" begin
        @test CA.beres_launch_filter_mode("beres2004") == CA.BERES_LAUNCH_FILTER_BERES2004
        @test CA.beres_launch_filter_mode("height_resolved") ==
              CA.BERES_LAUNCH_FILTER_HEIGHT_RESOLVED
        @test CA.beres_launch_filter_mode("none") == CA.BERES_LAUNCH_FILTER_NONE
        @test_throws ErrorException CA.beres_launch_filter_mode("cam")
        @test_throws ErrorException BeresSourceParams{FT}(;
            kw...,
            launch_filter_margin = FT(-1),
        )
        @test BeresSourceParams{FT}(; kw...).launch_filter ==
              CA.BERES_LAUNCH_FILTER_BERES2004
    end
end

# ============================================================================
# Beres-G transforms the heating itself (F03, option ii): the gathered profile,
# after the optional smoothing and the edge taper, is rescaled to the column
# heating ∫Q⁺dz above the floor, and the spectrum takes no Q0 factor.
# ============================================================================
@testset "Beres-G transforms the gathered heating itself" begin
    # An exact half-sine of peak Q0 from the floor (2 km) to 12 km on the
    # production grid, gathered and processed as the model does.
    function processed(FT; taper = true, smoothing = 0)
        Q0, z_lo, h = FT(5e-5), FT(2000), FT(10000)
        z_top = z_lo + h
        Q(z) = z_lo <= z <= z_top ? Q0 * sin(FT(π) * (z - z_lo) / h) : FT(0)
        state = CA._beres_gather_init(FT, Val(CA.N_BERES_PROFILE))
        for z in FT.(BERES_GRID.z)
            state = CA._beres_gather_step(
                state, (z, Q(z), z_lo, z_top), FT(2000), Val(CA.N_BERES_PROFILE),
            )
        end
        dz_level = FT(mean_spacing(z_lo, z_top))
        g = CA._beres_smooth_profile(first(state), h, FT(smoothing))
        g = taper ? CA._beres_taper_edges(g, h, dz_level) : g
        column_heating = 2 / FT(π) * Q0 * h   # ∫Q⁺dz of the half-sine
        return (; g = CA._beres_scale_profile(g, h, column_heating), Q0, z_lo, h, dz_level)
    end
    window_integral(g, h) = h / (length(g) - 1) * (sum(g) - (g[1] + g[end]) / 2)
    # Transient spectrum only: the steady c = 0 term of Beres-G uses the transform
    # of the gathered profile at m₀ = N/|u_heat| (F04, awaiting a decision), where the
    # half-sine's response is tiny and the profile's interpolation kinks are not.
    beres_kw(FT, general) = BeresSourceParams{FT}(;
        Q0_threshold = FT(0),
        beres_scale_factor = FT(2e-6),
        σ_x = FT(4000),
        ν_min = FT(8.727e-4),
        ν_max = FT(1.047e-2),
        n_ν = 9,
        beres_shape_general = general,
        beres_steady_source = false,
    )

    @testset "The rescaled profile carries the column heating" begin
        for taper in (true, false), smoothing in (0, 800)
            (; g, Q0, h) = processed(Float64; taper, smoothing)
            @test window_integral(g, h) ≈ 2 / π * Q0 * h rtol = 1e-12
        end
        # A degenerate window gives no profile, not NaN.
        @test all(iszero, CA._beres_scale_profile(ntuple(_ -> 0.0, 64), 1000.0, 1.0))
        @test all(iszero, CA._beres_scale_profile(ntuple(_ -> 1.0, 64), 0.0, 1.0))
    end

    @testset "A half-sine from the floor: Beres-G and Beres-C agree" begin
        FT = Float64
        (; g, Q0, z_lo, h) = processed(FT; taper = false)
        dc, cmax = FT(0.8), FT(40)
        nc = Int(2 * cmax / dc + 1)
        c = ntuple(n -> (n - 1) * dc - cmax, Val(nc))
        N, u_heat = FT(0.012), FT(-3.3)
        B_C = _beres_launch_spectrum(
            c, u_heat, Q0, h, N, z_lo, h, nothing, FT(0), beres_kw(FT, false), Val(nc),
        )
        # Untapered, sampled at the bin centres, with the transform origin at the
        # bottom of the heating: the Beres-C spectrum, up to the interpolation of
        # the half-sine on the model levels. Its kinks add flux at vertical
        # wavelengths shorter than two levels, i.e. in the bins closer to u_heat
        # than c_cut = N·dz_level/π, which the model's wavenumber cut removes; the
        # comparison is on the bins outside that.
        dz_level = FT(mean_spacing(z_lo, z_lo + h))
        B_G_ground = _beres_launch_spectrum(
            c, u_heat, Q0, h, N, FT(0), h, g, FT(0), beres_kw(FT, true), Val(nc);
            n_bin_samples = Val(1), dz_level,
        )
        c_cut = N * dz_level / π
        resolved = filter(n -> abs(c[n] - u_heat) > c_cut, 1:nc)
        @test maximum(n -> abs(B_G_ground[n] - B_C[n]), resolved) <
              0.02 * maximum(abs, B_C)
        # As the model runs it (taper, cut, bin average, origin at the ground
        # 2 km below the heating), the total differs by the elevation phase and
        # those choices only.
        (; g, dz_level) = processed(FT; taper = true)
        B_G = _beres_launch_spectrum(
            c, u_heat, Q0, h, N, z_lo, h, g, FT(0), beres_kw(FT, true), Val(nc);
            dz_level,
        )
        ratio = sum(abs, B_G) / sum(abs, B_C)
        @info "Half-sine from the floor: total |B0| Beres-G / Beres-C = $ratio (model settings), max bin difference at the ground origin outside c_cut = $(maximum(n -> abs(B_G_ground[n] - B_C[n]), resolved) / maximum(abs, B_C)) of the peak"
        # A regression guard pinned to the measured 0.83, not a derived bound: the
        # taper removes heating at the window edges that the rescaling puts back in
        # the middle, and the transform origin at the ground 2 km below the heating
        # and the cut change the slow bins. Recompute it if the grid, the taper or
        # the cut changes.
        @test 0.78 < ratio < 0.88
    end

    @testset "Float32 and Float64 agree" begin
        totals = map((Float32, Float64)) do FT
            (; g, Q0, z_lo, h, dz_level) = processed(FT; taper = true)
            dc, cmax = FT(0.8), FT(40)
            nc = Int(2 * cmax / dc + 1)
            c = ntuple(n -> (n - 1) * dc - cmax, Val(nc))
            B = _beres_launch_spectrum(
                c, FT(-3.3), Q0, h, FT(0.012), z_lo, h, g, FT(0),
                beres_kw(FT, true), Val(nc); dz_level,
            )
            (Float64(window_integral(g, h)), Float64(sum(abs, B)))
        end
        @test totals[1][1] ≈ totals[2][1] rtol = 1e-5
        @test totals[1][2] ≈ totals[2][2] rtol = 1e-3
    end
end
