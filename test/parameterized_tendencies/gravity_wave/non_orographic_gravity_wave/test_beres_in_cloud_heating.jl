# ============================================================================
# Beres in-cloud heating extraction test
#
# Builds a single-column simulation with prognostic EDMF, injects a synthetic
# updraft state (uniform area fraction a0, half-sine temperature anomaly,
# constant updraft velocity) into the EDMF cache, and calls
# compute_beres_convective_heating! directly. Verifies:
#
#   1. Q_conv (grid-mean) = a0 · Q_conv_ic (in-cloud) where a is uniform —
#      the area-fraction dilution is exactly the factor a0.
#   2. a_cover equals the (uniform) updraft area fraction over the envelope.
#   3. gw_Q0 (the spectrum amplitude) is built from the IN-CLOUD heating,
#      i.e. it is ~1/a0 larger than the grid-mean-derived amplitude.
#   4. Activation still gates on the GRID-MEAN amplitude (thresholds are
#      calibrated to grid-mean magnitudes).
#   5. The envelope (z_bot, z_top) is detected from the grid-mean field and
#      the updraft area fraction, exactly as before the in-cloud fix.
#
# Does NOT test propagation/deposition (see test_beres_column_drag.jl for the
# ā-linearity of the deposited drag).
# ============================================================================

using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore
import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry

@testset "Beres in-cloud heating extraction (synthetic updraft)" begin
    comms_ctx = ClimaComms.SingletonCommsContext()
    config_file = joinpath(
        @__DIR__,
        "../../../../config/model_configs/single_column_beres_nogw_test.yml",
    )
    config = CA.AtmosConfig(
        config_file;
        job_id = "beres_in_cloud_heating",
        comms_ctx,
    )

    simulation = CA.get_simulation(config)
    p = simulation.integrator.p
    Y = simulation.integrator.u

    FT = eltype(Y.c.ρ)

    (;
        gw_Q_conv,
        gw_Q_conv_ic,
        gw_a_cover,
        gw_Q0,
        gw_h_heat,
        gw_zbot,
        gw_ztop,
        gw_beres_active,
        ᶜbuoyancy_frequency,
    ) = p.non_orographic_gravity_wave
    beres = p.non_orographic_gravity_wave.gw_beres_source
    @test beres isa CA.BeresSourceParams

    ᶜz = Fields.coordinate_field(Y.c).z
    center_z = Array(Fields.field2array(ᶜz))[:, 1]

    # ------------------------------------------------------------------
    # Synthetic updraft state in the EDMF cache + prognostic ρa
    # ------------------------------------------------------------------
    a0 = FT(0.05)                       # uniform updraft area fraction in band
    z_lo = FT(3000)                     # updraft band bottom (above z_bot_floor)
    z_hi = FT(10000)                    # updraft band top

    (; ᶜρʲs, ᶜTʲs, ᶜT, ᶠu³ʲs, ᶠu³) = p.precomputed
    ᶜρ = Y.c.ρ

    # Draft density = grid-mean density ⇒ Q_conv = a0·Q_conv_ic exactly
    # (the two constructions then differ only by the area factor).
    parent(ᶜρʲs.:(1)) .= parent(ᶜρ)

    # Half-sine in-band temperature anomaly, 2 K peak
    @. ᶜTʲs.:(1) =
        ᶜT + ifelse(
            (ᶜz >= z_lo) & (ᶜz <= z_hi),
            FT(2) * sin(FT(π) * (ᶜz - z_lo) / (z_hi - z_lo)),
            FT(0),
        )

    # Uniform updraft area fraction inside the band, zero outside
    @. Y.c.sgsʲs.:(1).ρa =
        ifelse((ᶜz >= z_lo) & (ᶜz <= z_hi), a0 * ᶜρ, FT(0))

    # Updraft vertical velocity 1 m/s (physical), zero grid-mean velocity.
    # The DSE anomaly is band-limited, so the constant out-of-band velocity
    # carries no flux.
    ᶠlg = Fields.local_geometry_field(Y.f)
    @. ᶠu³ʲs.:(1) =
        Geometry.Contravariant3Vector(Geometry.WVector(FT(1)), ᶠlg)
    @. ᶠu³ = Geometry.Contravariant3Vector(Geometry.WVector(FT(0)), ᶠlg)

    fill!(ᶜbuoyancy_frequency, FT(0.012))

    # ------------------------------------------------------------------
    # Run the extraction
    # ------------------------------------------------------------------
    CA.compute_beres_convective_heating!(Y, p, ᶜbuoyancy_frequency)

    Qgm = Array(Fields.field2array(gw_Q_conv))[:, 1]
    Qic = Array(Fields.field2array(gw_Q_conv_ic))[:, 1]
    dz = Array(Fields.field2array(Fields.Δz_field(axes(Y.c))))[:, 1]

    zbot = parent(gw_zbot)[1]
    ztop = parent(gw_ztop)[1]
    h_heat = parent(gw_h_heat)[1]
    a_cover = parent(gw_a_cover)[1]
    Q0 = parent(gw_Q0)[1]
    active = parent(gw_beres_active)[1]

    in_band = findall(z -> z_lo <= z <= z_hi, center_z)
    interior = in_band[2:(end - 1)]

    @testset "Q_conv = a0 · Q_conv_ic where a is uniform" begin
        @test maximum(abs, Qic[interior]) > 0
        @test all(
            isapprox.(
                Qgm[interior],
                a0 .* Qic[interior];
                rtol = sqrt(eps(FT)),
                atol = 100 * eps(FT) * maximum(abs, Qic[interior]),
            ),
        )
    end

    @testset "a_cover equals the uniform area fraction" begin
        # The envelope may include at most a boundary level outside the
        # updraft band (where a = 0), so allow a small low-side tolerance.
        @test FT(0.8) * a0 <= a_cover <= a0 * (1 + sqrt(eps(FT)))
    end

    # Reference integrals over the diagnosed envelope
    env = findall(z -> zbot <= z <= ztop, center_z)
    @test !isempty(env)
    @test h_heat > 0
    Q0_ic_ref = max(FT(π) / 2 * sum(Qic[env] .* dz[env]) / h_heat, FT(0))
    Q0_gm_ref = max(FT(π) / 2 * sum(Qgm[env] .* dz[env]) / h_heat, FT(0))

    @testset "Q0 amplitude is built from the in-cloud heating" begin
        @test Q0 ≈ Q0_ic_ref rtol = FT(1e-5)
        # In-cloud amplitude must exceed the grid-mean amplitude by ~1/a0;
        # require at least a 5× separation to make the convention unambiguous.
        @test Q0 > 5 * Q0_gm_ref
    end

    @testset "Activation gates on the grid-mean amplitude" begin
        expected_active =
            (Q0_gm_ref > beres.Q0_threshold) && (h_heat > beres.h_heat_min)
        @test active == FT(expected_active)
        # The synthetic case is constructed to activate; guard against the
        # test silently passing with a degenerate (inactive) envelope.
        @test active == FT(1)
    end

    @testset "In-cloud profile invariant under a(z) — construction (a), not Q₁/a" begin
        # Building the area factor in (the correct way) makes the in-cloud
        # heating independent of the a(z) profile. Dividing it out afterwards
        # (Q₁/a(z)) does not: it leaves a spurious term wherever a varies with
        # height, e.g., near the plume top. So re-running with a strongly
        # height-varying a(z) (8× drop across the band) and the SAME Tʲ, wʲ, ρʲ
        # must leave Q_conv_ic unchanged.
        a_bot, a_top = FT(0.08), FT(0.01)
        @. Y.c.sgsʲs.:(1).ρa = ifelse(
            (ᶜz >= z_lo) & (ᶜz <= z_hi),
            (a_bot + (a_top - a_bot) * (ᶜz - z_lo) / (z_hi - z_lo)) * ᶜρ,
            FT(0),
        )
        CA.compute_beres_convective_heating!(Y, p, ᶜbuoyancy_frequency)
        Qgm_var = Array(Fields.field2array(gw_Q_conv))[:, 1]
        Qic_var = Array(Fields.field2array(gw_Q_conv_ic))[:, 1]

        # (a)-construction ⇒ in-cloud heating identical to the uniform-a case
        @test all(
            isapprox.(
                Qic_var[interior],
                Qic[interior];
                rtol = sqrt(eps(FT)),
                atol = 100 * eps(FT) * maximum(abs, Qic[interior]),
            ),
        )
        # sanity: the grid-mean DID change (a(z) really differs from a0)
        @test !all(
            isapprox.(Qgm_var[interior], Qgm[interior]; rtol = FT(1e-3)),
        )
        # post-hoc division is materially wrong where ∂z a ≠ 0
        a_of_z = [
            a_bot + (a_top - a_bot) * (z - z_lo) / (z_hi - z_lo) for
            z in center_z
        ]
        posthoc = Qgm_var[interior] ./ a_of_z[interior]
        rel_dev =
            maximum(abs, posthoc .- Qic_var[interior]) /
            maximum(abs, Qic_var[interior])
        @test rel_dev > 0.1
    end

    @testset "Native source-shape diagnostics (halfsine, launch flux, centroid)" begin
        # Re-run on the uniform-a0 state so the diagnosed envelope matches the
        # `zbot`/`ztop`/`h_heat`/`Q0` captured above.
        @. Y.c.sgsʲs.:(1).ρa =
            ifelse((ᶜz >= z_lo) & (ᶜz <= z_hi), a0 * ᶜρ, FT(0))
        CA.compute_beres_convective_heating!(Y, p, ᶜbuoyancy_frequency)

        (; gw_halfsine, gw_launch_flux, gw_c_centroid) =
            p.non_orographic_gravity_wave
        hs = Array(Fields.field2array(gw_halfsine))[:, 1]
        lf = parent(gw_launch_flux)[1]
        cc = parent(gw_c_centroid)[1]

        # The native half-sine equals Q0·sin(π(z−z_bot)/h) over [z_bot, z_bot+h]
        # and is zero elsewhere. Compare against an offline reconstruction from
        # the SAME column's (Q0, z_bot, h) — must agree to round-off (this is the
        # whole point: in-column, the three are mutually consistent).
        hs_ref = [
            (zbot <= z <= zbot + h_heat && h_heat > 0) ?
            Q0 * sin(FT(π) * (z - zbot) / h_heat) : FT(0) for z in center_z
        ]
        @test all(
            isapprox.(
                hs,
                hs_ref;
                rtol = sqrt(eps(FT)),
                atol = 10 * eps(FT) * Q0,
            ),
        )
        @test maximum(hs) ≈ Q0 rtol = FT(1e-3)   # half-sine peaks at Q0
        # zero below the envelope bottom and above the envelope top
        @test all(hs[center_z .< zbot] .== 0)
        @test all(hs[center_z .> zbot + h_heat] .== 0)

        # Launched-spectrum summaries: finite, and the active column launches a
        # positive flux. The centroid is a phase speed within the c-grid range.
        @test isfinite(lf) && lf > 0
        @test isfinite(cc)
        cmax = -p.non_orographic_gravity_wave.gw_c[1]
        @test -cmax <= cc <= cmax
    end

    @testset "Envelope detection unchanged (grid-mean field + area)" begin
        # z_bot: lowest level above the altitude floor with Qgm > threshold
        cand = findall(
            i ->
                center_z[i] >= beres.z_bot_floor &&
                    Qgm[i] > beres.z_bot_Q_threshold,
            eachindex(center_z),
        )
        @test !isempty(cand)
        @test zbot ≈ center_z[minimum(cand)]
        # z_top: highest level with updraft area fraction > 1e-3 (band top)
        @test ztop ≈ center_z[maximum(in_band)]
        @test h_heat ≈ ztop - zbot
    end
end
