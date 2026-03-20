using Test
import ClimaAtmos as CA

const BeresSourceParams = CA.BeresSourceParams
const wave_source = CA.wave_source

@testset "Beres squall-line forcing" begin
    FT = Float64

    # Phase speed bins: -100 to +100 m/s in steps of 4 m/s
    dc = FT(4.0)
    cmax = FT(100.0)
    nc = Int(2 * cmax / dc + 1)  # 51 bins
    c = ntuple(n -> FT((n - 1) * dc - cmax), Val(nc))

    # Hardcoded squall-line 4-tuple (Beres 2004, mature tropical squall line)
    Q0 = FT(0.004)         # K/s — intense deep convective heating
    h = FT(6000.0)         # m — deep heating depth
    u_heat = FT(-5.0)      # m/s — easterly trade-wind mean flow
    N_source = FT(0.01)    # rad/s — tropical buoyancy frequency

    beres = BeresSourceParams{FT}(;
        Q0_threshold = FT(1.157e-4),
        beres_scale_factor = FT(1.0),
        σ_x = FT(4000.0),
        ν_min = FT(2π / (120 * 60)),
        ν_max = FT(2π / (10 * 60)),
        n_ν = 9,
    )

    @testset "Squall-line spectrum is nonzero and finite" begin
        B = wave_source(c, u_heat, Q0, h, N_source, beres, Val(nc))

        @test B isa NTuple{nc, FT}
        @test all(isfinite, B)
        @test !all(iszero, B)
        # Squall-line should produce substantial momentum flux
        @test maximum(abs, B) > 1e-10
    end

    @testset "Easterly wind breaks east-west symmetry" begin
        B_squall = wave_source(c, u_heat, Q0, h, N_source, beres, Val(nc))
        B_nowind = wave_source(c, FT(0.0), Q0, h, N_source, beres, Val(nc))

        # With u_heat=0, spectrum is antisymmetric
        mid = (nc + 1) ÷ 2
        for i in 1:(mid - 1)
            @test B_nowind[mid - i] ≈ -B_nowind[mid + i] atol = 1e-30 rtol = 1e-10
        end

        # With u_heat=-5 (easterly), symmetry is broken
        east_flux = sum(B_squall[i] for i in 1:nc if c[i] < 0)
        west_flux = sum(B_squall[i] for i in 1:nc if c[i] > 0)
        @test east_flux != west_flux  # asymmetric
        # Easterly wind Doppler-shifts the spectrum: westward-propagating waves
        # (positive B, positive c_hat) should have different magnitudes than eastward
        @test abs(east_flux - west_flux) / max(abs(east_flux), abs(west_flux)) > 0.01
    end

    @testset "Squall amplitude exceeds weak-convection amplitude" begin
        Q0_weak = FT(2e-4)  # just above activation threshold
        B_squall = wave_source(c, u_heat, Q0, h, N_source, beres, Val(nc))
        B_weak = wave_source(c, u_heat, Q0_weak, h, N_source, beres, Val(nc))

        max_squall = maximum(abs, B_squall)
        max_weak = maximum(abs, B_weak)

        # Squall must dominate weak convection
        @test max_squall > 10 * max_weak

        # Q0² scaling: ratio should be (0.004/0.0002)² = 400
        # Test at individual bins where both are nonzero
        test_indices = filter(i -> abs(B_weak[i]) > 1e-40, 1:nc)
        for i in test_indices
            ratio = B_squall[i] / B_weak[i]
            @test ratio ≈ (Q0 / Q0_weak)^2 rtol = 1e-10
        end
    end

    @testset "Breaking potential: B0/c_hat^3 exceeds saturation" begin
        B = wave_source(c, u_heat, Q0, h, N_source, beres, Val(nc))

        # Representative stratospheric values for breaking condition check
        ρ_source = FT(0.4)     # kg/m³ at source (~450 hPa)
        ρ_kp1 = FT(0.01)      # kg/m³ in upper stratosphere (~1 hPa)
        bf = FT(0.02)          # rad/s (stratospheric buoyancy frequency)
        kwv = FT(2π / (30e3))  # wave number for first wavelength bin (ink=1)
        u_strat = FT(0.0)     # m/s — assume zero stratospheric wind for simplicity

        fac = FT(0.5) * (ρ_kp1 / ρ_source) * kwv / bf

        # Count how many phase speeds satisfy the breaking condition Foc >= 0
        n_breaking = 0
        for i in 1:nc
            c_hat = c[i] - u_strat
            if abs(c_hat) < 1e-6
                continue
            end
            Foc = B[i] / c_hat^3 - fac
            if Foc >= 0
                n_breaking += 1
            end
        end

        # At least some waves should break in the stratosphere
        @test n_breaking > 0
        println("  Squall-line test: $n_breaking / $nc phase speeds satisfy breaking condition")
    end

    @testset "Sensitivity to heating depth" begin
        # Deep convection (h=6000m) should produce more total integrated flux
        # than shallow (h=2000m). Note: peak amplitude can be higher for shallow h
        # due to sharper sinc resonance, but total flux across all phase speeds
        # should be greater for deeper heating.
        h_shallow = FT(2000.0)
        B_deep = wave_source(c, u_heat, Q0, h, N_source, beres, Val(nc))
        B_shallow = wave_source(c, u_heat, Q0, h_shallow, N_source, beres, Val(nc))

        @test sum(abs, B_deep) > sum(abs, B_shallow)
    end
end
