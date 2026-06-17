using Test
import ClimaAtmos as CA

const BeresSourceParams = CA.BeresSourceParams
const wave_source = CA.wave_source
const V_hs_sq = CA.V_hs_sq
const _beres_steady_flux = CA._beres_steady_flux
const _beres_steady_horizontal_const = CA._beres_steady_horizontal_const

# Tests for the STEADY (ν=0) stationary-mechanical component of the Beres source
# (Beres 2004 Eqs. 31–34), normalization per the (30)→(32) rederivation. The
# reference grid and H value hard-coded below (V_hs_sq, β, even-folded H, F_T
# continuous/discrete, R table, L-invariance) were computed offline from the
# closed-form expressions in the docstrings of `_beres_steady_flux` and
# `_beres_steady_horizontal_const`.

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
