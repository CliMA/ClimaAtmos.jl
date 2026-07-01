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
import Thermodynamics as TD
import ClimaParams as CP
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.BulkMicrophysicsTendencies as BMT

const BeresSourceParams = CA.BeresSourceParams
const wave_source = CA.wave_source
const V_hs_sq = CA.V_hs_sq
const _beres_steady_flux = CA._beres_steady_flux
const _beres_steady_horizontal_const = CA._beres_steady_horizontal_const
const _beres_gs = CA._beres_gs
const _beres_launch_spectrum = CA._beres_launch_spectrum
const _beres_mech_flux = CA._beres_mech_flux
const _beres_b0_band = CA._beres_b0_band
const _beres_peaknorm = CA._beres_peaknorm

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
# reference grid and H value hard-coded below (V_hs_sq, β, even-folded H, F_T
# continuous/discrete, R table, L-invariance) were computed offline from the
# closed-form expressions in the docstrings of `_beres_steady_flux` and
# `_beres_steady_horizontal_const`.
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

    @testset "Even-folded horizontal constant H" begin
        # Closed-form H matches the offline-computed quadrature reference value.
        H = _beres_steady_horizontal_const(σx, Lsys)
        @test H > 0
        @test H ≈ 2.993366e7 rtol = 1e-5
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

    @testset "Reference grid match (Eqs 31–34, even-folded H + DC weight)" begin
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
            @test sflux(U, h, Q0) ≈ FT(fref) rtol = 1e-10
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
# *aggregated* phase-change tendencies (the tested `_aggregate_tendencies`).
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

# Independent oracle: the explicit canonical sum Q_lat = (1/cp) Σ_p L_p R_p,
# assembled from the individual verbose source terms with hand-assigned signs
# and the COMPLETE latent-heat-releasing selection (audited against
# CloudMicrophysics `_aggregate_tendencies`). L_f is derived as L_s − L_v so the
# two forms share identical constants and must agree to round-off.
function Q_lat_explicit(mp, thp, ρ, T, q_tot, q_lcl, q_icl, q_rai, q_sno)
    src = BMT.bulk_microphysics_tendencies(
        BMT.InstantaneousVerbose(),
        BMT.Microphysics1Moment(),
        mp,
        thp,
        ρ,
        T,
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
            rtol = FT === Float32 ? FT(2e-4) : FT(1e-10)
            atol = FT === Float32 ? FT(1e-9) : FT(1e-14)

            for T in Ts, (q_tot, q_lcl, q_icl, q_rai, q_sno) in qs
                shipped = CA._beres_latent_heating(
                    mp,
                    thp,
                    ρ,
                    T,
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

@testset "Beres canonical latent heating: physical sign sanity" begin
    FT = Float64
    toml_dict = CP.create_toml_dict(FT)
    mp = CMP.Microphysics1MParams(
        toml_dict;
        rain_autoconversion = CMP.PrescribedNd(toml_dict),
    )
    thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
    ρ = FT(0.7)

    # Warm, strongly supersaturated, no ice → net condensation → warming.
    T_warm = FT(295)
    q_sat_warm = TD.q_vap_saturation(thp, T_warm, ρ)
    Q_cond = CA._beres_latent_heating(
        mp,
        thp,
        ρ,
        T_warm,
        FT(2) * q_sat_warm,  # q_tot well above saturation
        FT(1e-3),
        FT(0),
        FT(5e-4),
        FT(0),
    )
    @test Q_cond > 0  # condensation releases latent heat

    # Warm, strongly subsaturated, rain present → net evaporation → cooling.
    Q_evap = CA._beres_latent_heating(
        mp,
        thp,
        ρ,
        T_warm,
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

    # diagnostic_edmfx → reject (no per-draft prognostic condensate state).
    args_diag = merge(base, Dict("turbconv" => "diagnostic_edmfx"))
    @test_throws ErrorException CA.get_non_orographic_gravity_wave_model(
        args_diag,
        nothing,
        Float64,
    )

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
        e33 = abs(_beres_gs(_halfsine_profile(Val(33), FT), FT(0), h, m_test)^2 - Vref) / Vref
        e65 = abs(_beres_gs(_halfsine_profile(Val(65), FT), FT(0), h, m_test)^2 - Vref) / Vref
        e129 = abs(_beres_gs(_halfsine_profile(Val(129), FT), FT(0), h, m_test)^2 - Vref) / Vref
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

        g = _halfsine_profile(Val(CA.N_BERES_PROFILE), FT)   # = 64 (production)

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
        B_g = _beres_launch_spectrum(
            c, u_heat, Q0, h, N_source, FT(0), h, g, FT(0), beres_g, Val(nc),
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

    @testset "Peak-normalization to unit maximum" begin
        t = (FT(0), FT(0.5), FT(2.0), FT(1.0), FT(0))
        tn = _beres_peaknorm(t)
        @test maximum(tn) ≈ FT(1.0)
        @test tn[3] ≈ FT(1.0)               # the peak maps to 1
        @test tn[2] ≈ FT(0.25)
        # All-zero / nonpositive ⇒ zeros (no division blow-up).
        @test all(==(FT(0)), _beres_peaknorm((FT(0), FT(0), FT(0))))
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
