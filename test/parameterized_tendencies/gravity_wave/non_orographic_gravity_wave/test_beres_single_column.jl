# ============================================================================
# Beres (2004) NOGW — single-column tests (ONE get_simulation build)
#
# Both phases below use the SAME single-column config
# (single_column_beres_nogw_test.yml), so the simulation is built once and
# reused — the build is the dominant cost. Two phases:
#
#   Phase A — in-cloud heating extraction: inject a synthetic EDMF updraft and
#             call compute_beres_convective_heating!; verify the in-cloud vs
#             grid-mean amplitude convention, coverage, envelope, and native
#             source-shape diagnostics.
#   Phase B — column drag: fill the gw_* source fields directly and call
#             non_orographic_gravity_wave_forcing; verify vertical drag
#             structure, ā-linearity, and the steady (ν=0) component.
#
# Order-safe: Phase B fill!s every gw_Q0/gw_h_heat/… so it cleanly overwrites
# Phase A's heating output; Phase B reads only fields it sets plus ᶜu/ᶜv/ᶜρ.
#
# Does NOT test: spectral shape (see test_beres_unit.jl) or sphere-grid wiring
# (see test_beres_sphere_integration.jl).
# ============================================================================

using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore
import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry

# ----------------------------------------------------------------------------
# Build the single-column simulation once and share it across both phases.
# ----------------------------------------------------------------------------
comms_ctx = ClimaComms.SingletonCommsContext()
config_file = joinpath(
    @__DIR__,
    "../../../../config/model_configs/single_column_beres_nogw_test.yml",
)
config = CA.AtmosConfig(config_file; job_id = "beres_single_column", comms_ctx)
simulation = CA.get_simulation(config)
p = simulation.integrator.p
Y = simulation.integrator.u

FT = eltype(Y.c.ρ)

ᶜz = Fields.coordinate_field(Y.c).z
center_z = Array(Fields.field2array(ᶜz))[:, 1]

# ============================================================================
# Phase A — Beres in-cloud heating extraction (synthetic updraft)
#
# Builds a synthetic updraft state (uniform area fraction a0, half-sine
# temperature anomaly, constant updraft velocity) into the EDMF cache, and
# calls compute_beres_convective_heating! directly. Verifies:
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
# ============================================================================
# Outer testset so Phase A failures do not abort Phase B: a *top-level*
# @testset throws at its `end` when it contains failures, which would stop the
# script before Phase B. Nesting both phases inside one outer testset keeps
# Phase A failures non-fatal (only the outermost testset throws, at the very
# end) so Phase B always runs.
@testset "Beres NOGW single column (one shared build)" begin
    @testset "Beres in-cloud heating extraction (synthetic updraft)" begin
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
            # The discrete half-sine samples Q0·sin(π(z−z_bot)/h) ≤ Q0 at each
            # level; its grid maximum sits just below Q0 because the continuous
            # peak (z_bot+h/2) rarely lands exactly on a level.
            @test maximum(hs) <= Q0 * (1 + sqrt(eps(FT)))
            @test maximum(hs) > FT(0.95) * Q0
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

        @testset "Moment-matched envelope is a valid heating layer" begin
            # The envelope (z_bot, z_top, h) is set by moment-matching a half-sine
            # to the in-cloud heating Q_conv_ic (the legacy area-threshold
            # detection is deprecated). Assert the invariants a valid
            # moment-matched envelope must satisfy, rather than the old area-mode
            # reference values (which no longer match the implementation).
            @test beres.z_bot_floor <= zbot       # respects the PBL altitude floor
            @test zbot < ztop                     # ordered envelope
            @test h_heat ≈ ztop - zbot            # depth = envelope span
            @test h_heat > beres.h_heat_min       # deep enough to activate Beres
            # The envelope intersects the injected heating band [z_lo, z_hi]
            # (the moment-matched top may slightly overshoot the literal band top).
            @test zbot < z_hi
            @test ztop > z_lo
        end
    end

    # ============================================================================
    # Phase B — Beres (2004) column drag (single-column vertical propagation)
    #
    # Validates the full Beres forcing pipeline (source spectrum → propagation →
    # breaking → momentum deposition) on a single column with analytic wind shear.
    # Uses the Beres (2004) §4 squall-line parameters.
    #
    # Tests: nonzero drag, max drag location (stratospheric), bounded magnitude,
    # v-forcing near zero (no meridional wind), ā-linearity, steady (ν=0) component.
    # σ_x = 2500 m (Beres 2004 §4 squall-line case) comes from the config's
    # toml/beres_squall_sigma_x.toml override.
    # ============================================================================
    @testset "Beres column drag -- single-column with wind shear" begin
        # Verify Beres is enabled in column mode (requires the column-cache fix)
        @test p.non_orographic_gravity_wave.gw_beres_source isa CA.BeresSourceParams

        # Extract NOGW cache
        (;
            gw_source_height,
            gw_ncval,
            ᶜbuoyancy_frequency,
            ᶜlevel,
            u_waveforcing,
            v_waveforcing,
            uforcing,
            vforcing,
            gw_Q0,
            gw_h_heat,
            gw_u_heat,
            gw_v_heat,
            gw_N_source,
            gw_beres_active,
            gw_a_cover,
            gw_flag,
        ) = p.non_orographic_gravity_wave

        center_space = axes(Y.c)
        z_max = FT(50000.0)  # matches config

        # Compute source_level and damp_level from height (column mode)
        source_level = argmin(abs.(center_z .- gw_source_height))
        damp_level = Spaces.nlevels(center_space)

        # ------------------------------------------------------------------
        # Fill atmospheric state with analytic profiles
        # ------------------------------------------------------------------
        ᶜρ = Y.c.ρ
        ᶜu = similar(ᶜρ, FT)
        ᶜv = similar(ᶜρ, FT)

        # Linear wind shear: u(z) = -10 + z × 20/z_max
        # Ranges from -10 m/s at surface to +10 m/s at model top.
        # Creates critical levels for waves with different phase speeds,
        # producing non-trivial vertical drag structure.
        # Use ClimaCore field broadcast (GPU-compatible) instead of parent().
        @. ᶜu = FT(-10.0) + ᶜz * FT(20.0 / z_max)
        fill!(ᶜv, FT(0.0))

        # Uniform buoyancy frequency matching Beres §4
        fill!(ᶜbuoyancy_frequency, FT(0.012))

        # Source and damp level fields
        ᶜρ_source = Fields.level(ᶜρ, source_level)
        ᶜu_source = Fields.level(ᶜu, source_level)
        ᶜv_source = Fields.level(ᶜv, source_level)

        # ------------------------------------------------------------------
        # Fill Beres convective-source fields (squall-line values)
        # ------------------------------------------------------------------
        fill!(gw_Q0, FT(0.004))         # K/s — peak heating rate (Beres §4)
        fill!(gw_h_heat, FT(5000.0))    # m — heating depth (Beres §4)
        fill!(gw_u_heat, FT(-5.0))      # m/s — mean wind in heating region (Beres §4)
        fill!(gw_v_heat, FT(0.0))       # m/s — no meridional wind
        fill!(gw_N_source, FT(0.012))   # s⁻¹ — buoyancy frequency (Beres §4)
        fill!(gw_beres_active, FT(1.0)) # enable Beres branch
        # ā = 1: no coverage dilution, so the deposited drag equals the local
        # (in-cloud) Beres flux — preserves this test's expected magnitudes.
        # The coverage scaling itself is tested below.
        fill!(gw_a_cover, FT(1.0))
        fill!(gw_flag, FT(0.0))         # tropics flag (AD99 Doppler frame; no effect on Beres)

        # ------------------------------------------------------------------
        # Call forcing
        # ------------------------------------------------------------------
        function call_forcing!()
            uforcing .= 0
            vforcing .= 0
            CA.non_orographic_gravity_wave_forcing(
                ᶜu,
                ᶜv,
                ᶜbuoyancy_frequency,
                ᶜρ,
                ᶜz,
                ᶜlevel,
                source_level,
                damp_level,
                ᶜρ_source,
                ᶜu_source,
                ᶜv_source,
                uforcing,
                vforcing,
                gw_ncval,
                u_waveforcing,
                v_waveforcing,
                p,
            )
            return (
                Array(Fields.field2array(uforcing))[:, 1],
                Array(Fields.field2array(vforcing))[:, 1],
            )
        end
        uf_data, vf_data = call_forcing!()

        println("Column drag test:")
        println("  max |uforcing| = $(maximum(abs, uf_data)) m/s²")
        println("  max |vforcing| = $(maximum(abs, vf_data)) m/s²")
        println("  source_level = $source_level (z = $(center_z[source_level]) m)")

        # ------------------------------------------------------------------
        # Assertions
        # ------------------------------------------------------------------
        @testset "All finite" begin
            @test all(isfinite, uf_data)
            @test all(isfinite, vf_data)
        end

        @testset "Nonzero drag" begin
            max_uf = maximum(abs, uf_data)
            @test max_uf > 0
            # Beres forcing with these parameters should produce measurable drag
            @test max_uf > 1e-8
        end

        @testset "Max drag above source level (stratospheric)" begin
            # Wave breaking and momentum deposition should occur above the
            # source level, typically in the stratosphere (15–30 km for
            # squall-line inputs with this wind profile).
            peak_idx = argmax(abs.(uf_data))
            peak_z = center_z[peak_idx]
            source_z = center_z[source_level]
            println("  Peak drag at z = $(peak_z) m (level $peak_idx)")
            @test peak_z > source_z
        end

        @testset "Drag magnitude bounded" begin
            max_uf = maximum(abs, uf_data)
            # 3e-3 m/s² ≈ 260 m/s/day — the forcing clamp in the tendency
            # function. Physical drag should be well below this.
            @test max_uf < 3e-3
        end

        @testset "V-forcing near zero" begin
            # With v_heat = 0 and v = 0 everywhere, the meridional spectrum
            # is symmetric and the net v-forcing should be negligible compared
            # to the u-forcing.
            max_vf = maximum(abs, vf_data)
            max_uf = maximum(abs, uf_data)
            if max_uf > 1e-10
                @test max_vf < 0.01 * max_uf
            end
        end

        @testset "Deposition scales linearly with coverage ā" begin
            # The Beres deposition factor is ε = ā/(ρ_source·nk): the launched
            # spectrum (and hence the breaking levels) is independent of ā,
            # while the deposited drag is exactly linear in it. Difference out
            # the (ā-independent) AD99 background and check linearity.
            a_frac = FT(0.05)
            fill!(gw_a_cover, FT(0.0))
            uf_ad99, _ = call_forcing!()
            fill!(gw_a_cover, a_frac)
            uf_frac, _ = call_forcing!()
            beres_full = uf_data .- uf_ad99   # ā = 1 Beres contribution
            beres_frac = uf_frac .- uf_ad99   # ā = 0.05 Beres contribution
            @test maximum(abs, beres_full) > 0
            @test isapprox(
                beres_frac,
                a_frac .* beres_full;
                rtol = sqrt(eps(FT)),
                atol = 100 * eps(FT) * maximum(abs, beres_full),
            )
            fill!(gw_a_cover, FT(1.0))  # restore
        end

        @testset "Steady (ν=0) component: finite, decelerates U, deposits at c≈0" begin
            # Smoke test of the full pipeline (source → propagation → breaking →
            # deposition) for the steady (ν=0) component. The steady source has no config
            # switch: it is on by default and gated by the grid (this config's coarse grid,
            # nogw_beres_test.toml cmax=100/dc=4 → c=0 bin, enables it). So `uf_data` from
            # above is the steady-ON result. To isolate the steady contribution we rebuild
            # the source struct with beres_steady_source = false (the field is retained on
            # BeresSourceParams precisely for this fixed-grid toggle), swap it into the
            # NOGW cache, and difference: steady = (ON) − (OFF) on identical inputs.
            bs_on = p.non_orographic_gravity_wave.gw_beres_source
            @test bs_on.beres_steady_source  # default-on
            bs_off = CA.BeresSourceParams{FT}(;
                (
                    n => getfield(bs_on, n) for n in fieldnames(typeof(bs_on))
                )...,
                beres_steady_source = false,
            )
            nogw_off = merge(
                p.non_orographic_gravity_wave,
                (; gw_beres_source = bs_off),
            )
            # Current src no longer supports merge(::AtmosCache); build a lightweight
            # cache NamedTuple with just the fields the GW forcing path reads.
            p_off = (;
                p.precomputed,
                p.params,
                p.atmos,
                p.scratch,
                non_orographic_gravity_wave = nogw_off,
            )

            uforcing .= 0
            vforcing .= 0
            CA.non_orographic_gravity_wave_forcing(
                ᶜu,
                ᶜv,
                ᶜbuoyancy_frequency,
                ᶜρ,
                ᶜz,
                ᶜlevel,
                source_level,
                damp_level,
                ᶜρ_source,
                ᶜu_source,
                ᶜv_source,
                uforcing,
                vforcing,
                gw_ncval,
                u_waveforcing,
                v_waveforcing,
                p_off,
            )
            uf_off = Array(Fields.field2array(uforcing))[:, 1]
            @test all(isfinite, uf_off)

            # Steady contribution = (steady ON) − (steady OFF), identical inputs.
            steady_contrib = uf_data .- uf_off
            max_steady = maximum(abs, steady_contrib)
            max_trans = maximum(abs, uf_off)
            println(
                "  steady: max |Δuforcing| = $max_steady m/s² " *
                "($(round(100 * max_steady / max_trans; digits = 4))% of transient)",
            )
            @test max_steady > 0

            # The stationary (c≈0) wave launched in a heating-region wind u_heat=-5
            # m/s deposits drag that OPPOSES that flow: the mass-weighted net steady
            # momentum forcing should be positive (decelerating the westward U<0).
            ρ_col = Array(Fields.field2array(ᶜρ))[:, 1]
            net_steady = sum(ρ_col .* steady_contrib)
            println("  steady: Σρ·Δuforcing = $net_steady (U_heat = -5 ⇒ expect > 0)")
            @test net_steady > 0
        end
    end
end
