using Test

# Standalone copy of beres_wave_source for unit testing.
# This avoids pulling in ClimaCore/ClimaAtmos dependencies.
# Once integrated, this test can import from ClimaAtmos instead.

function beres_wave_source_standalone(
    c::NTuple{nc, FT},
    u_heat::FT,
    Q0::FT,
    h::FT,
    N_source::FT,
    σ_x::FT,
    ν_min::FT,
    ν_max::FT,
    n_ν::Int,
    scale_factor::FT,
    gw_ncval::Val{nc},
) where {nc, FT}
    boole_w = (FT(7), FT(32), FT(12), FT(32), FT(7))
    dν = (ν_max - ν_min) / FT(n_ν - 1)
    n_groups = (n_ν - 1) ÷ 4

    h_half = h / FT(2)
    π_val = FT(π)
    inv_2π = FT(1) / (FT(2) * π_val)
    Q0_h_over_π_sq = (Q0 * h / π_val)^2

    ntuple(
        n -> begin
            c_n = c[n]
            c_hat = c_n - u_heat

            if abs(c_hat) < FT(1e-6)
                FT(0)
            else
                integral = FT(0)
                for g in 1:n_groups
                    for j in 1:5
                        idx = (g - 1) * 4 + j
                        ν_j = ν_min + FT(idx - 1) * dν

                        if abs(c_n) < FT(1e-6)
                            continue
                        end
                        k = ν_j / c_n

                        ω_hat = ν_j - k * u_heat

                        ω_hat_min = FT(1e-4) * N_source
                        if abs(ω_hat) < ω_hat_min
                            continue
                        end

                        m_sq = N_source^2 * k^2 / ω_hat^2 - k^2

                        if m_sq <= FT(0)
                            continue
                        end

                        m = sqrt(m_sq)
                        m_h = m * h

                        denom = m_h^2 - π_val^2
                        if abs(denom) < FT(1e-6)
                            vert_response = Q0_h_over_π_sq * FT(4) / π_val^2
                        else
                            vert_term =
                                FT(2) * m_h * cos(m * h_half) / denom
                            vert_response = Q0_h_over_π_sq * vert_term^2
                        end

                        horiz_response = exp(-k^2 * σ_x^2)
                        transfer = abs(k) / (m * ω_hat^2)
                        f_val =
                            inv_2π * vert_response * horiz_response * transfer
                        w = boole_w[j] * FT(2) * dν / FT(45)
                        integral = integral + w * f_val
                    end
                end

                sign(c_hat) * scale_factor * integral
            end
        end,
        Val(nc),
    )
end

@testset "Beres (2004) source spectrum" begin
    FT = Float64

    # Phase speed bins: -100 to +100 m/s in steps of 2 m/s
    dc = FT(2.0)
    cmax = FT(100.0)
    nc = Int(2 * cmax / dc + 1)  # 101 bins
    c = ntuple(n -> FT((n - 1) * dc - cmax), Val(nc))

    # Physical parameters
    N_source = FT(0.01)        # buoyancy frequency (1/s)
    σ_x = FT(4000.0)          # horizontal half-width (m), 4 km
    ν_min = FT(2π / (120 * 60))  # min frequency: period = 120 min
    ν_max = FT(2π / (10 * 60))   # max frequency: period = 10 min
    n_ν = 9                    # quadrature points (4k+1 = 9)
    scale_factor = FT(1.0)

    @testset "Output structure" begin
        Q0 = FT(10.0 / 86400.0)  # 10 K/day → K/s
        h = FT(10000.0)           # 10 km
        u_heat = FT(0.0)

        B = beres_wave_source_standalone(
            c, u_heat, Q0, h, N_source, σ_x, ν_min, ν_max, n_ν,
            scale_factor, Val(nc),
        )

        @test B isa NTuple{nc, FT}
        @test length(B) == nc
        @test all(isfinite, B)
        @test !all(iszero, B)  # should produce non-zero spectrum
    end

    @testset "Symmetry with U=0" begin
        Q0 = FT(10.0 / 86400.0)
        h = FT(10000.0)
        u_heat = FT(0.0)

        B = beres_wave_source_standalone(
            c, u_heat, Q0, h, N_source, σ_x, ν_min, ν_max, n_ν,
            scale_factor, Val(nc),
        )

        # With u_heat=0, spectrum should be antisymmetric: B(-c) = -B(c)
        mid = (nc + 1) ÷ 2  # index of c=0
        for i in 1:(mid - 1)
            # c[mid - i] = -c[mid + i]
            @test B[mid - i] ≈ -B[mid + i] atol = 1e-20 rtol = 1e-10
        end
        # c=0 bin should be zero
        @test B[mid] == FT(0)
    end

    @testset "Q0² amplitude scaling" begin
        h = FT(10000.0)
        u_heat = FT(0.0)

        Q0_1 = FT(5.0 / 86400.0)   # 5 K/day
        Q0_2 = FT(10.0 / 86400.0)  # 10 K/day (2x)

        B1 = beres_wave_source_standalone(
            c, u_heat, Q0_1, h, N_source, σ_x, ν_min, ν_max, n_ν,
            scale_factor, Val(nc),
        )
        B2 = beres_wave_source_standalone(
            c, u_heat, Q0_2, h, N_source, σ_x, ν_min, ν_max, n_ν,
            scale_factor, Val(nc),
        )

        # Doubling Q0 should quadruple flux (Q0² dependence)
        # Check at a few non-zero phase speed bins
        for i in [10, 30, 50, 70, 90]
            if abs(B1[i]) > 1e-30
                ratio = B2[i] / B1[i]
                @test ratio ≈ 4.0 rtol = 1e-10
            end
        end
    end

    @testset "Scale factor linearity" begin
        Q0 = FT(10.0 / 86400.0)
        h = FT(10000.0)
        u_heat = FT(0.0)

        B1 = beres_wave_source_standalone(
            c, u_heat, Q0, h, N_source, σ_x, ν_min, ν_max, n_ν,
            FT(1.0), Val(nc),
        )
        B2 = beres_wave_source_standalone(
            c, u_heat, Q0, h, N_source, σ_x, ν_min, ν_max, n_ν,
            FT(3.0), Val(nc),
        )

        for i in 1:nc
            @test B2[i] ≈ 3.0 * B1[i] atol = 1e-30
        end
    end

    @testset "Doppler shift with nonzero U" begin
        Q0 = FT(10.0 / 86400.0)
        h = FT(10000.0)
        u_heat = FT(10.0)  # 10 m/s mean wind

        B = beres_wave_source_standalone(
            c, u_heat, Q0, h, N_source, σ_x, ν_min, ν_max, n_ν,
            scale_factor, Val(nc),
        )

        @test all(isfinite, B)

        # With positive u_heat, the spectrum should be shifted:
        # positive phase speeds (c > u_heat) should have positive flux
        # negative phase speeds (c < u_heat) should have negative flux
        # The peak should be shifted toward c = u_heat

        # Find indices where |B| is largest for positive and negative sides
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
        # Both sides should have non-zero flux
        @test positive_max > 0
        @test negative_min < 0
    end

    @testset "Zero Q0 gives zero spectrum" begin
        Q0 = FT(0.0)
        h = FT(10000.0)
        u_heat = FT(0.0)

        B = beres_wave_source_standalone(
            c, u_heat, Q0, h, N_source, σ_x, ν_min, ν_max, n_ν,
            scale_factor, Val(nc),
        )

        @test all(iszero, B)
    end

    @testset "Magnitude comparison with AD spectrum" begin
        # The Beres spectrum should produce momentum flux densities
        # in a similar range to the AD Gaussian spectrum when Q0 and h
        # correspond to typical tropical deep convection.
        Q0 = FT(10.0 / 86400.0)  # 10 K/day
        h = FT(12000.0)           # 12 km deep convection
        u_heat = FT(0.0)

        B = beres_wave_source_standalone(
            c, u_heat, Q0, h, N_source, σ_x, ν_min, ν_max, n_ν,
            scale_factor, Val(nc),
        )

        # Typical AD spectrum amplitudes are O(0.001-1) Pa/(m/s)
        # Beres should be in a similar ballpark (within a few orders of magnitude)
        max_B = maximum(abs, B)
        @test max_B > 0  # non-zero
        @test max_B < 1e10  # not unphysically large

        # Print for manual inspection
        println("Beres spectrum max amplitude: ", max_B, " Pa/(m/s)")
        println("Spectrum at c = 20 m/s: ", B[Int((20 + cmax) / dc + 1)])
        println("Spectrum at c = -20 m/s: ", B[Int((-20 + cmax) / dc + 1)])
    end

    @testset "Different n_ν values (convergence)" begin
        Q0 = FT(10.0 / 86400.0)
        h = FT(10000.0)
        u_heat = FT(0.0)

        # Compare n_ν = 5 vs n_ν = 9 vs n_ν = 13
        B5 = beres_wave_source_standalone(
            c, u_heat, Q0, h, N_source, σ_x, ν_min, ν_max, 5,
            scale_factor, Val(nc),
        )
        B9 = beres_wave_source_standalone(
            c, u_heat, Q0, h, N_source, σ_x, ν_min, ν_max, 9,
            scale_factor, Val(nc),
        )
        B13 = beres_wave_source_standalone(
            c, u_heat, Q0, h, N_source, σ_x, ν_min, ν_max, 13,
            scale_factor, Val(nc),
        )

        # With Boole's rule, 9 and 13 points should be closer than 5 and 9
        # (convergence test)
        diff_5_9 = sum(abs, B5 .- B9)
        diff_9_13 = sum(abs, B9 .- B13)

        println("Convergence: |B5-B9| = $diff_5_9, |B9-B13| = $diff_9_13")
        # 9→13 difference should be smaller than 5→9 (convergence)
        if diff_5_9 > 1e-30
            @test diff_9_13 < diff_5_9
        end
    end
end
