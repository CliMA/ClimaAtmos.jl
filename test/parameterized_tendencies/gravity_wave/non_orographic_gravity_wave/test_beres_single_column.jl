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

# A deep updraft (area fraction 0.05 from 3 to 12 km, warmest at 6.5 km) and, if
# `thin_layer`, a nearly empty one above it (area 1e-5, below a_min, from 13 to 20 km)
# with a large cold anomaly (-3 K), whose in-cloud heating is large but meaningless.
# The 12-13 km gap keeps the deep updraft's levels from seeing the thin layer through
# the flux at their faces. Returns the envelope and amplitude the Beres extraction
# gives.
function beres_thin_layer_envelope(Y, p; thin_layer)
    FT = eltype(Y.c.ρ)
    ᶜz = Fields.coordinate_field(Y.c).z
    (; ᶜρʲs, ᶜTʲs, ᶜT, ᶠu³ʲs, ᶠu³) = p.precomputed
    ᶜρ = Y.c.ρ
    @. ᶜρʲs.:(1) = ᶜρ
    @. ᶜTʲs.:(1) =
        ᶜT + ifelse(
            (ᶜz >= 3000) & (ᶜz <= 10000),
            FT(2) * sin(FT(π) * (ᶜz - 3000) / 7000),
            ifelse(thin_layer & (ᶜz > 13000) & (ᶜz <= 20000), FT(-3), FT(0)),
        )
    a_thin = thin_layer ? FT(1e-5) : FT(0)
    @. Y.c.sgsʲs.:(1).ρa = ifelse(
        (ᶜz >= 3000) & (ᶜz <= 12000),
        FT(0.05) * ᶜρ,
        ifelse((ᶜz > 13000) & (ᶜz <= 20000), a_thin * ᶜρ, FT(0)),
    )
    ᶠlg = Fields.local_geometry_field(Y.f)
    @. ᶠu³ʲs.:(1) = Geometry.Contravariant3Vector(Geometry.WVector(FT(1)), ᶠlg)
    @. ᶠu³ = Geometry.Contravariant3Vector(Geometry.WVector(FT(0)), ᶠlg)
    nogw = p.non_orographic_gravity_wave
    fill!(nogw.ᶜbuoyancy_frequency, FT(0.012))
    CA.compute_beres_convective_heating!(Y, p, nogw.ᶜbuoyancy_frequency)
    value(f) = parent(f)[1]
    return (;
        z_bot = value(nogw.gw_zbot),
        z_top = value(nogw.gw_ztop),
        h = value(nogw.gw_h_heat),
        Q0 = value(nogw.gw_Q0),
        a_cover = value(nogw.gw_a_cover),
        u_heat = value(nogw.gw_u_heat),
        active = value(nogw.gw_beres_active),
    )
end
const BERES_THIN_LAYER_FLOAT64 = Ref{Any}(nothing)

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
#   5. The envelope (z_bot, z_top) is moment-matched to the in-cloud heating
#      Q_conv_ic (centroid z_c + spread σ → half-sine of depth h); the legacy
#      grid-mean area-threshold detection is deprecated and gated off.
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
            @test ztop <= center_z[end]           # at or below the top cell centre
            @test h_heat ≈ ztop - zbot            # depth = envelope span
            @test h_heat > beres.h_heat_min       # deep enough to activate Beres
            # The envelope intersects the injected heating band [z_lo, z_hi]
            # (the moment-matched top may slightly overshoot the literal band top).
            @test zbot < z_hi
            @test ztop > z_lo
        end

        @testset "Activation counts only the heating above the floor" begin
            # The draft is warmest at 2 km and its anomaly falls linearly to zero
            # at 14 km, so it heats from 2 to 14 km and cools from 1 to 2 km. The
            # fitted envelope reaches below 1 km, so it includes the cooling, which
            # the moment fit and the gather leave out (it is below the floor). The
            # activation amplitude must leave it out too and divide by the width of
            # the envelope above the floor. The area fraction is set so that this
            # amplitude is just above the threshold, while the amplitude over the
            # whole envelope divided by h, which includes the cooling, is below it.
            z_floor = beres.z_bot_floor
            ΔT(z) =
                z < 1000 || z > 14000 ? 0.0 :
                z < 2000 ? 2 * (z - 1000) / 1000 : 2 * (14000 - z) / 12000
            @. ᶜTʲs.:(1) = ᶜT + ΔT(ᶜz)
            dz_col = Array(Fields.field2array(Fields.Δz_field(axes(Y.c))))[:, 1]
            function run_with_area(a)
                @. Y.c.sgsʲs.:(1).ρa =
                    ifelse((ᶜz >= 500) & (ᶜz <= 15000), FT(a) * ᶜρ, FT(0))
                CA.compute_beres_convective_heating!(Y, p, ᶜbuoyancy_frequency)
                Q = Array(Fields.field2array(gw_Q_conv))[:, 1]
                z_b, z_t, h = parent(gw_zbot)[1], parent(gw_ztop)[1], parent(gw_h_heat)[1]
                z_start = max(z_b, z_floor)
                in_env = (center_z .>= z_b) .& (center_z .<= z_t)
                above = in_env .& (center_z .>= z_floor)
                return (;
                    floor_amplitude = (π / 2) * sum(Q[above] .* dz_col[above]) /
                                      (z_t - z_start),
                    whole_amplitude = (π / 2) * sum(Q[in_env] .* dz_col[in_env]) / h,
                    z_b,
                    cooling = minimum(Q[in_env .& (center_z .< z_floor)]; init = 0.0),
                    active = parent(gw_beres_active)[1],
                )
            end
            unit = run_with_area(1)
            a_test = 1.05 * beres.Q0_threshold / unit.floor_amplitude
            @test 0 < a_test < 1
            res = run_with_area(a_test)
            @test res.z_b < 1000                  # the envelope includes the cooling
            @test res.cooling < 0
            @test res.floor_amplitude ≈ 1.05 * beres.Q0_threshold rtol = 1e-6
            @test res.whole_amplitude < beres.Q0_threshold
            @test res.active == 1
        end

        @testset "Levels with near-empty updrafts are left out of the fit (a_min)" begin
            # With a nearly empty updraft (area below a_min) above the deep one, the
            # envelope, amplitude and envelope means equal the ones without that
            # layer, and the envelope top stays below it. The moment-matched top
            # can lie above the highest heated level (a half-sine fit reaches h/2
            # above the centroid), so it is compared with the run without the
            # layer, not with that level. With no per-level floor, the layer's
            # heating moves the envelope up.
            with_layer = beres_thin_layer_envelope(Y, p; thin_layer = true)
            without_layer = beres_thin_layer_envelope(Y, p; thin_layer = false)
            @test with_layer.active == 1
            @test with_layer.z_top < 13000
            @test with_layer.z_top == without_layer.z_top
            @test with_layer.z_bot == without_layer.z_bot
            @test with_layer.h ≈ without_layer.h rtol = 1e-10
            @test with_layer.Q0 ≈ without_layer.Q0 rtol = 1e-10
            @test with_layer.a_cover ≈ without_layer.a_cover rtol = 1e-10
            BERES_THIN_LAYER_FLOAT64[] = with_layer
            nogw = p.non_orographic_gravity_wave
            bs_no_floor = CA.BeresSourceParams{FT}(;
                (n => getfield(beres, n) for n in fieldnames(typeof(beres)))...,
                a_min = FT(1e-9),
            )
            p_no_floor = (;
                p.precomputed,
                p.params,
                p.atmos,
                p.scratch,
                non_orographic_gravity_wave = merge(
                    nogw,
                    (; gw_beres_source = bs_no_floor),
                ),
            )
            no_floor = beres_thin_layer_envelope(Y, p_no_floor; thin_layer = true)
            @test no_floor.z_top > with_layer.z_top + 1000   # the check has power
        end

        @testset "Coverage averages over the whole envelope (a_min)" begin
            # The updraft has area 1.1e-4 (just above a_min) only on two 1 km layers,
            # 4-5 and 10-11 km, and none elsewhere. Only those levels enter the heating
            # fit, whose envelope then spans about 14 km; the coverage averages over
            # the whole envelope, about 2e-5, below a_min, so the column stays
            # inactive. (Averaging over the qualifying levels only would give 1.1e-4.)
            @. ᶜTʲs.:(1) =
                ᶜT + ifelse(
                    (ᶜz >= 3000) & (ᶜz <= 14000),
                    FT(2) * (14000 - ᶜz) / 11000,
                    FT(0),
                )
            in_layers = @. ((ᶜz >= 4000) & (ᶜz <= 5000)) | ((ᶜz >= 10000) & (ᶜz <= 11000))
            @. Y.c.sgsʲs.:(1).ρa = ifelse(in_layers, FT(1.1e-4) * ᶜρ, FT(0))
            nogw = p.non_orographic_gravity_wave
            beres_no_threshold = CA.BeresSourceParams{FT}(;
                (n => getfield(beres, n) for n in fieldnames(typeof(beres)))...,
                Q0_threshold = FT(0),
            )
            p_no_threshold = (;
                p.precomputed,
                p.params,
                p.atmos,
                p.scratch,
                non_orographic_gravity_wave = merge(
                    nogw,
                    (; gw_beres_source = beres_no_threshold),
                ),
            )
            CA.compute_beres_convective_heating!(Y, p_no_threshold, ᶜbuoyancy_frequency)
            @test parent(gw_ztop)[1] - parent(gw_zbot)[1] > 10000   # a deep envelope
            @test parent(gw_a_cover)[1] < 3e-5
            @test parent(gw_a_cover)[1] > 1e-5
            @test parent(gw_beres_active)[1] == 0
        end

        @testset "Columns with near-empty updrafts stay inactive (a_min)" begin
            # A column is active only when its coverage a_cover (the mass-weighted
            # mean updraft area over the envelope) is at least a_min = 1e-4, and
            # levels with less updraft area than a_min are left out of the fit. With
            # the area fraction uniform in the whole column, a_cover equals it.
            # The heating threshold is set to zero so that only the coverage gate
            # decides; the in-cloud heating, and so the envelope, does not depend
            # on a uniform area fraction.
            @test beres.a_min == FT(1e-4)
            @. ᶜTʲs.:(1) =
                ᶜT + ifelse(
                    (ᶜz >= z_lo) & (ᶜz <= z_hi),
                    FT(2) * sin(FT(π) * (ᶜz - z_lo) / (z_hi - z_lo)),
                    FT(0),
                )
            nogw = p.non_orographic_gravity_wave
            with_source(bs) = (;
                p.precomputed,
                p.params,
                p.atmos,
                p.scratch,
                non_orographic_gravity_wave = merge(nogw, (; gw_beres_source = bs)),
            )
            source_with(; kw...) = CA.BeresSourceParams{FT}(;
                (n => getfield(beres, n) for n in fieldnames(typeof(beres)))...,
                kw...,
            )
            p_no_threshold = with_source(source_with(; Q0_threshold = FT(0)))
            for (a, expected_active) in ((0.99 * beres.a_min, 0), (1.01 * beres.a_min, 1))
                @. Y.c.sgsʲs.:(1).ρa = FT(a) * ᶜρ
                CA.compute_beres_convective_heating!(Y, p_no_threshold, ᶜbuoyancy_frequency)
                # Below a_min every level is left out, so there is no envelope and
                # no coverage; above it the coverage is the area fraction.
                if expected_active == 1
                    @test parent(gw_a_cover)[1] ≈ a rtol = 1e-10
                    @test parent(gw_h_heat)[1] > beres.h_heat_min
                else
                    @test parent(gw_a_cover)[1] == 0
                end
                @test parent(gw_beres_active)[1] == expected_active
            end

            # A regular deep column (area fraction a0 in the band, none outside)
            # is active, and its launch spectra are the same with a 10 times lower
            # a_min: every level is either well above both values or empty.
            @. Y.c.sgsʲs.:(1).ρa =
                ifelse((ᶜz >= z_lo) & (ᶜz <= z_hi), a0 * ᶜρ, FT(0))
            (; gw_beres_B0_u) = nogw
            CA.compute_beres_convective_heating!(Y, p_no_threshold, ᶜbuoyancy_frequency)
            @test parent(gw_beres_active)[1] == 1
            # The coverage is the mass-weighted mean of the draft area ρa/ρʲ over
            # the whole envelope: a0·ρ/ρʲ (here a0, with ρʲ = ρ) on the band levels
            # inside it and zero on the rest.
            ρ_col = Array(Fields.field2array(ᶜρ))[:, 1]
            ρʲ_col = Array(Fields.field2array(ᶜρʲs.:(1)))[:, 1]
            dz_col = Array(Fields.field2array(Fields.Δz_field(axes(Y.c))))[:, 1]
            in_env =
                (center_z .>= parent(gw_zbot)[1]) .& (center_z .<= parent(gw_ztop)[1])
            in_band = in_env .& (center_z .>= z_lo) .& (center_z .<= z_hi)
            expected_a_cover =
                sum(
                    a0 .* ρ_col[in_band] ./ ρʲ_col[in_band] .* ρ_col[in_band] .*
                    dz_col[in_band],
                ) /
                sum(ρ_col[in_env] .* dz_col[in_env])
            @test parent(gw_a_cover)[1] ≈ expected_a_cover rtol = 0.05
            @test parent(gw_a_cover)[1] > beres.a_min
            B0_with_gate = copy(Array(parent(gw_beres_B0_u)))
            CA.compute_beres_convective_heating!(
                Y,
                with_source(source_with(; Q0_threshold = FT(0), a_min = FT(1e-5))),
                ᶜbuoyancy_frequency,
            )
            @test parent(gw_beres_active)[1] == 1
            @test Array(parent(gw_beres_B0_u)) == B0_with_gate
        end

        @testset "A draft with negative area counts as empty in the coverage" begin
            # The EDMF solver can leave ρa slightly negative. Such a level holds no
            # updraft, so it adds zero to the coverage, as it adds nothing to the
            # in-cloud heating; counting the negative value would lower a_cover.
            # Here the band from z_lo to z_hi has area a0, except the levels from 8
            # to 9 km, which have area -a0. They lie in the warming upper half of the
            # band, which sets the envelope (the lower half cools).
            in_negative = @. (ᶜz >= 8000) & (ᶜz <= 9000)
            @. Y.c.sgsʲs.:(1).ρa = ifelse(
                (ᶜz >= z_lo) & (ᶜz <= z_hi),
                ifelse(in_negative, -a0, a0) * ᶜρ,
                FT(0),
            )
            nogw = p.non_orographic_gravity_wave
            beres_no_threshold = CA.BeresSourceParams{FT}(;
                (n => getfield(beres, n) for n in fieldnames(typeof(beres)))...,
                Q0_threshold = FT(0),
            )
            p_no_threshold = (;
                p.precomputed,
                p.params,
                p.atmos,
                p.scratch,
                non_orographic_gravity_wave = merge(
                    nogw,
                    (; gw_beres_source = beres_no_threshold),
                ),
            )
            CA.compute_beres_convective_heating!(Y, p_no_threshold, ᶜbuoyancy_frequency)
            ρ_col = Array(Fields.field2array(ᶜρ))[:, 1]
            ρa_col = Array(Fields.field2array(Y.c.sgsʲs.:(1).ρa))[:, 1]
            ρʲ_col = Array(Fields.field2array(ᶜρʲs.:(1)))[:, 1]
            dz_col = Array(Fields.field2array(Fields.Δz_field(axes(Y.c))))[:, 1]
            in_env =
                (center_z .>= parent(gw_zbot)[1]) .& (center_z .<= parent(gw_ztop)[1])
            mass = ρ_col[in_env] .* dz_col[in_env]
            area = ρa_col[in_env] ./ ρʲ_col[in_env]
            expected_a_cover = sum(max.(area, 0) .* mass) / sum(mass)
            signed_a_cover = sum(area .* mass) / sum(mass)
            # Setup: the negative levels lie inside the envelope and matter.
            @test any(in_env .& (center_z .>= 8000) .& (center_z .<= 9000))
            @test signed_a_cover < 0.9 * expected_a_cover
            @test parent(gw_a_cover)[1] ≈ expected_a_cover rtol = 1e-6
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
    # toml/nogw_beres_test.toml override.
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
            gw_zbot,
            gw_ztop,
            gw_w_obstacle,
            beres_source_ρ_z_u_v_level,
        ) = p.non_orographic_gravity_wave

        center_space = axes(Y.c)
        z_max = FT(50000.0)  # matches config

        # Compute source_level and damp_level from height (column mode)
        # The level closest to the source height, the upper of two equally close levels,
        # as `non_orographic_gravity_wave_compute_tendency!` picks it.
        source_distance = abs.(center_z .- gw_source_height)
        source_level = findlast(==(minimum(source_distance)), source_distance)
        damp_level = CA.highest_damp_level(Spaces.nlevels(center_space))

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
        # Envelope consistent with h_heat, and no obstacle updraft. With the
        # default half-sine shape and the mechanical source off these do not
        # change the spectrum, but they must be finite.
        fill!(gw_zbot, FT(2000.0))
        fill!(gw_ztop, FT(7000.0))
        fill!(gw_w_obstacle, FT(0.0))
        # Launch-filter winds of this envelope: u over [2, 7.5] km (up to the launch
        # level, the first level at or above z_top = 7 km) runs from -9.2 to -7 m/s,
        # and v is zero. The default (Beres 2004) filter removes those phase speeds.
        u_of(z) = FT(-10.0) + z * FT(20.0 / z_max)
        z_launch = center_z[findfirst(>=(FT(7000)), center_z)]
        z_first = center_z[findfirst(>=(FT(2000)), center_z)]
        fill!(
            p.non_orographic_gravity_wave.gw_beres_wind_range,
            (u_of(z_first), u_of(z_launch), FT(0), FT(0), u_of(z_launch), FT(0)),
        )
        # The forcing kernel reads the precomputed launch spectra
        # gw_beres_B0_u/v, not the scalar source fields above, so rebuild them
        # from the values just set.
        CA._beres_precompute_launch_spectra!(p.non_orographic_gravity_wave)
        # The waves launch at the first level at or above gw_ztop. The launch
        # state is set as production sets it, from these analytic profiles, not
        # left from the startup callback.
        CA.set_beres_launch_state!(
            beres_source_ρ_z_u_v_level, ᶜρ, ᶜz, ᶜu, ᶜv, ᶜlevel, gw_ztop,
        )

        # ------------------------------------------------------------------
        # Call forcing
        # ------------------------------------------------------------------
        # `pp` is the cache the forcing reads; a NamedTuple with swapped NOGW fields
        # runs the same column with other source settings.
        function call_forcing!(pp = p)
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
                pp,
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
            # The steady term is folded into the precomputed spectra, so rebuild
            # them with the steady-OFF parameters (nogw_off shares the B0 arrays
            # with the original cache; they are restored below).
            CA._beres_precompute_launch_spectra!(nogw_off)
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
            CA._beres_precompute_launch_spectra!(p.non_orographic_gravity_wave)
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

        @testset "Launch level falls back to the top level when gw_ztop is above it" begin
            # On a terrain-following grid the top cell centre differs between
            # columns, so a heating top can lie above every level of its own
            # column. The launch state must then be the top level, never the
            # zero density of the search's starting value, and the Beres waves
            # launched there all go to the damping layer, so below it the drag is
            # the AD99 drag alone, bit for bit.
            nlev = Spaces.nlevels(axes(ᶜρ))
            damp_first = CA.highest_damp_level(nlev)
            fill!(gw_a_cover, FT(0))
            uf_ad99, _ = call_forcing!()
            fill!(gw_a_cover, FT(1))
            launch_value(i) = parent(getproperty(beres_source_ρ_z_u_v_level, i))[1]
            for ztop_beyond in (FT(1e6), FT(NaN))
                fill!(gw_ztop, ztop_beyond)
                CA.set_beres_launch_state!(
                    beres_source_ρ_z_u_v_level, ᶜρ, ᶜz, ᶜu, ᶜv, ᶜlevel, gw_ztop,
                )
                @test launch_value(1) > 0                 # launch density
                @test launch_value(2) == center_z[end]    # launch height
                @test launch_value(5) == nlev             # launch level
                uf_top, vf_top = call_forcing!()
                @test all(isfinite, uf_top)
                @test all(isfinite, vf_top)
                @test uf_top[1:(damp_first - 1)] == uf_ad99[1:(damp_first - 1)]
            end

            # A zero launch density (no level found, as the search's starting value
            # leaves it) gives no Beres deposit rather than an infinite one.
            fill!(gw_ztop, FT(7000.0))
            CA.set_beres_launch_state!(
                beres_source_ρ_z_u_v_level, ᶜρ, ᶜz, ᶜu, ᶜv, ᶜlevel, gw_ztop,
            )
            ρ_launch = launch_value(1)
            parent(beres_source_ρ_z_u_v_level.:1) .= FT(0)
            uf_zero_ρ, vf_zero_ρ = call_forcing!()
            @test all(isfinite, uf_zero_ρ)
            @test all(isfinite, vf_zero_ρ)
            @test uf_zero_ρ == uf_ad99
            parent(beres_source_ρ_z_u_v_level.:1) .= ρ_launch
            @test call_forcing!()[1] == uf_data   # the launch state is restored
        end

        @testset "An empty AD99 spectrum adds no drag and keeps the Beres drag" begin
            # With nogw_Bw = nogw_Bn = 0 every AD99 bin is zero, so Σ|B0| = 0. The
            # AD99 intermittency must then be zero, not infinite: an infinite one
            # gives NaN above the AD99 launch level, which the clean-up turns into
            # zero drag for both sources. Zeroing the source amplitude instead
            # (Bt_0 = Bt_eq = 0) is the other way to switch the AD99 drag off and
            # must give the same drag. Read before the clean-up, so a NaN shows.
            nogw = p.non_orographic_gravity_wave
            zero_level = zero(nogw.gw_Bw)
            cache_with(nt) = (;
                p.precomputed,
                p.params,
                p.atmos,
                p.scratch,
                non_orographic_gravity_wave = merge(nogw, nt),
            )
            uf_bw0, vf_bw0 =
                call_forcing!(cache_with((; gw_Bw = zero_level, gw_Bn = zero_level)))
            uf_amp0, vf_amp0 = call_forcing!(cache_with((; gw_source_ampl = zero_level)))
            @test all(isfinite, uf_bw0)
            @test all(isfinite, vf_bw0)
            @test maximum(abs, uf_bw0) > 0
            @test uf_bw0 == uf_amp0
            @test vf_bw0 == vf_amp0
            # Both equal the Beres drag of the full run, the full drag minus the
            # AD99 drag (a_cover = 0 switches the Beres deposit off).
            fill!(gw_a_cover, FT(0))
            uf_ad99, _ = call_forcing!()
            fill!(gw_a_cover, FT(1))
            @test isapprox(
                uf_bw0,
                uf_data .- uf_ad99;
                rtol = sqrt(eps(FT)),
                atol = 100 * eps(FT) * maximum(abs, uf_bw0),
            )
        end

        @testset "Beres breaking test treats B0 as a stress" begin
            # The Beres B0 is a stress [Pa], so a wave breaks where
            # B0/ĉ³ ≥ ½·ρ·k/N, with no division by the launch density (that
            # division turns the AD99 flux per unit mass into a stress). One bin
            # at c = 20 m/s with 1.5 times that limit at the first level above
            # launch must break in the first layer, for a launch density of 0.2
            # and of 1 (for 1 the two forms agree); with 0.5 times the limit it
            # must not. The AD99 drag is switched off by its source amplitude.
            nogw = p.non_orographic_gravity_wave
            (; gw_c, gw_beres_B0_u, gw_beres_B0_v) = nogw
            p_beres_only = (;
                p.precomputed,
                p.params,
                p.atmos,
                p.scratch,
                non_orographic_gravity_wave = merge(
                    nogw,
                    (; gw_source_ampl = zero(nogw.gw_source_ampl)),
                ),
            )
            launch_level = Int(parent(beres_source_ρ_z_u_v_level.:5)[1])
            ρ_launch = parent(beres_source_ρ_z_u_v_level.:1)[1]
            ρ_col = Array(Fields.field2array(ᶜρ))[:, 1]
            u_col = Array(Fields.field2array(ᶜu))[:, 1]
            N = FT(0.012)                      # uniform buoyancy frequency above
            k_band1 = 2 * FT(π) / (30 * FT(10) * 1000)   # largest k of the bands
            n_bin = findfirst(==(FT(20)), gw_c)
            c_hat = gw_c[n_bin] - u_col[launch_level + 1]
            B0_limit = ρ_col[launch_level + 1] * k_band1 * c_hat^3 / (2 * N)
            @test c_hat > 0                    # setup: no sign change in the first layer
            fill!(gw_beres_B0_v, CA.BeresB0(ntuple(_ -> FT(0), length(gw_c))))
            # Drag outside the two cells the first layer spans (it puts half its
            # flux in the launch cell and half in the cell above).
            function drag_beyond_first_layer(B0_factor, ρ_source)
                B0 = ntuple(n -> n == n_bin ? B0_factor * B0_limit : FT(0), length(gw_c))
                fill!(gw_beres_B0_u, CA.BeresB0(B0))
                parent(beres_source_ρ_z_u_v_level.:1) .= ρ_source
                uf, _ = call_forcing!(p_beres_only)
                @test maximum(abs, uf) > 0
                beyond = vcat(uf[1:(launch_level - 1)], uf[(launch_level + 2):end])
                return maximum(abs, beyond)
            end
            for ρ_source in (FT(0.2), FT(1))
                @test drag_beyond_first_layer(FT(1.5), ρ_source) == 0
                @test drag_beyond_first_layer(FT(0.5), ρ_source) > 0
            end
            parent(beres_source_ρ_z_u_v_level.:1) .= ρ_launch
            CA._beres_precompute_launch_spectra!(nogw)
            @test call_forcing!()[1] == uf_data   # the launch spectra are restored
        end
    end

    # ============================================================================
    # Phase C — Beres-G (sine_transform) end to end
    #
    # The same column with the shape mode switched to the sine transform of the
    # gathered in-cloud heating, run through the production entry point
    # `non_orographic_gravity_wave_compute_tendency!` (buoyancy frequency, heating
    # extraction, gather, launch state, propagation, clean-up). The synthetic
    # draft heats from 2 to 14 km and cools from 1 to 2 km, so its envelope
    # reaches below the 2 km floor. Checks: the gathered profile is the
    # floor-started, linearly interpolated, peak-normalized and edge-tapered
    # heating; the launch spectra are the ones built from it with the window and
    # level spacing; the slow bins next to u_heat carry almost nothing; and the
    # drag is finite.
    # ============================================================================
    @testset "Beres-G (sine_transform) single column, end to end" begin
        nogw = p.non_orographic_gravity_wave
        bs_c = nogw.gw_beres_source
        bs_g = CA.BeresSourceParams{FT}(;
            (n => getfield(bs_c, n) for n in fieldnames(typeof(bs_c)))...,
            beres_shape_general = true,
        )
        p_g = (;
            p.precomputed,
            p.params,
            p.atmos,
            p.scratch,
            non_orographic_gravity_wave = merge(nogw, (; gw_beres_source = bs_g)),
        )
        (; ᶜρʲs, ᶜTʲs, ᶜT) = p.precomputed
        parent(ᶜρʲs.:(1)) .= parent(Y.c.ρ)
        ΔT(z) =
            z < 1000 || z > 14000 ? 0.0 :
            z < 2000 ? 2 * (z - 1000) / 1000 : 2 * (14000 - z) / 12000
        @. ᶜTʲs.:(1) = ᶜT + ΔT(ᶜz)
        @. Y.c.sgsʲs.:(1).ρa =
            ifelse((ᶜz >= 500) & (ᶜz <= 15000), FT(1) * Y.c.ρ, FT(0))

        CA.non_orographic_gravity_wave_compute_tendency!(
            Y,
            p_g,
            p.atmos.non_orographic_gravity_wave,
        )

        (;
            gw_zbot,
            gw_ztop,
            gw_Q0,
            gw_h_heat,
            gw_u_heat,
            gw_v_heat,
            gw_N_source,
            gw_beres_active,
            gw_Q_conv_ic,
            gw_beres_gprofile,
            gw_beres_dz_level,
            gw_w_obstacle,
            gw_beres_B0_u,
            gw_beres_B0_v,
            gw_c,
            gw_ncval,
            uforcing,
            vforcing,
        ) = nogw
        value(f) = parent(f)[1]
        # The value of a level field of tuples, read component by component.
        tuple_value(f, n) = ntuple(i -> parent(getproperty(f, i))[1], n)
        z_floor = bs_g.z_bot_floor
        @test value(gw_beres_active) == 1
        @test value(gw_zbot) < z_floor            # the envelope reaches below the floor
        z_start = CA._beres_gather_bottom(value(gw_zbot), value(gw_ztop), z_floor)
        @test z_start == z_floor
        span = value(gw_ztop) - z_start

        # The level spacing is the mean thickness of the cells in the window.
        dz_col = Array(Fields.field2array(Fields.Δz_field(axes(Y.c))))[:, 1]
        overlap =
            (center_z .+ dz_col ./ 2 .> z_start) .&
            (center_z .- dz_col ./ 2 .< value(gw_ztop))
        dz_level = sum(dz_col[overlap]) / count(overlap)
        @test value(gw_beres_dz_level) ≈ dz_level rtol = 1e-12

        # The gathered profile, rebuilt from the model's in-cloud heating.
        Q_ic = Array(Fields.field2array(gw_Q_conv_ic))[:, 1]
        ng_val = Val(CA.N_BERES_PROFILE)
        state = CA._beres_gather_init(FT, ng_val)
        for k in eachindex(center_z)
            state = CA._beres_gather_step(
                state,
                (center_z[k], Q_ic[k], z_start, value(gw_ztop)),
                z_floor,
                ng_val,
            )
        end
        expected_profile =
            CA._beres_scale_profile(
                CA._beres_taper_edges(first(state), span, dz_level),
                span,
                2 / FT(π) * value(gw_Q0) * value(gw_h_heat),
            )
        profile = tuple_value(gw_beres_gprofile, CA.N_BERES_PROFILE)
        @test all(isapprox.(profile, expected_profile; rtol = 1e-12, atol = 1e-14))
        @test profile[1] == 0
        @test profile[end] == 0
        # No gaps between heated samples. The first and last samples can be zero
        # besides the tapered end points: on this 1 km grid the level just above
        # the floor cools (masked to zero), and the window top may lie above the
        # highest heated level.
        i_first = findfirst(>(0), profile)
        i_last = findlast(>(0), profile)
        @test i_last - i_first > CA.N_BERES_PROFILE ÷ 2
        @test all(>(0), profile[i_first:i_last])

        # The launch spectra are the ones built from that profile, on that window,
        # with that level spacing.
        # The launch-filter winds: the range of u and v over the heating layer up to
        # the launch level (the first level at or above z_top), and those winds.
        wind_range = tuple_value(nogw.gw_beres_wind_range, 6)
        u_col = Array(Fields.field2array(nogw.ᶜuv_temp.components.data.:1))[:, 1]
        v_col = Array(Fields.field2array(nogw.ᶜuv_temp.components.data.:2))[:, 1]
        # The layer starts at the bottom of the Beres-G heating, the gather window
        # bottom max(z_bot, floor), here the floor (z_bot is below it).
        k_launch = findfirst(>=(value(gw_ztop)), center_z)
        layer = findall(z -> z >= z_start, center_z[1:k_launch])
        @test z_start > value(gw_zbot)
        @test wind_range ==
              (
            minimum(u_col[layer]),
            maximum(u_col[layer]),
            minimum(v_col[layer]),
            maximum(v_col[layer]),
            u_col[k_launch],
            v_col[k_launch],
        )
        for (azimuth, B0, wind) in
            ((1, gw_beres_B0_u, gw_u_heat), (2, gw_beres_B0_v, gw_v_heat))
            launch_filter =
                CA._beres_launch_filter(bs_g, wind_range, nothing, Val(azimuth))
            expected_B0 = CA._beres_launch_spectrum(
                gw_c, value(wind), value(gw_Q0), value(gw_h_heat),
                value(gw_N_source), z_start, span, profile, value(gw_w_obstacle),
                bs_g, gw_ncval, Val(true); dz_level, launch_filter,
            )
            @test all(
                isapprox.(
                    tuple_value(B0.B0, length(gw_c)),
                    expected_B0;
                    rtol = 1e-12,
                    atol = 0,
                ),
            )
        end

        # Beres-C (half-sine) starts the launch-filter layer at z_bot itself.
        let
            CA.compute_beres_convective_heating!(Y, p, nogw.ᶜbuoyancy_frequency)
            k_launch_c = findfirst(>=(value(gw_ztop)), center_z)
            layer_c = findall(z -> z >= value(gw_zbot), center_z[1:k_launch_c])
            @test minimum(center_z[layer_c]) < z_floor   # setup: the layers differ
            @test tuple_value(nogw.gw_beres_wind_range, 6)[1:2] ==
                  (minimum(u_col[layer_c]), maximum(u_col[layer_c]))
            CA.compute_beres_convective_heating!(Y, p_g, nogw.ᶜbuoyancy_frequency)
        end

        # The slow bins next to u_heat carry almost nothing of the launched flux.
        B0_u = tuple_value(gw_beres_B0_u.B0, length(gw_c))
        u_heat = value(gw_u_heat)
        near = sum(
            abs(B0_u[n]) for n in eachindex(gw_c) if abs(gw_c[n] - u_heat) < 2;
            init = zero(FT),
        )
        @test sum(abs, B0_u) > 0
        @test near < 0.01 * sum(abs, B0_u)

        # The drag (after the clean-up) is finite and not zero.
        uf = Array(Fields.field2array(uforcing))[:, 1]
        vf = Array(Fields.field2array(vforcing))[:, 1]
        @test all(isfinite, uf)
        @test all(isfinite, vf)
        @test maximum(abs, uf) > 0

        # Beres-G transforms the in-cloud heating itself: the profile it uses
        # integrates to the column heating ∫Q⁺dz above the floor, the integral
        # behind Q0, whatever the window, smoothing and taper (F03, option ii).
        column_heating(prof) =
            span / (CA.N_BERES_PROFILE - 1) *
            (sum(prof) - (prof[1] + prof[end]) / 2)
        above_floor = center_z .>= z_floor
        gathered = sum(max.(Q_ic[above_floor], 0) .* dz_col[above_floor])
        @test column_heating(profile) ≈ gathered rtol = 1e-10
        @test 2 / FT(π) * value(gw_Q0) * value(gw_h_heat) ≈ gathered rtol = 1e-10

        @testset "The same column heating with smoothing" begin
            # A sharper heating than the half-sine (a tanh step in the draft
            # temperature gives a sech² heating, about 1 km wide, at 8 km), with
            # and without a 500 m smoothing: the profile integrates to ∫Q⁺dz.
            @. ᶜTʲs.:(1) =
                ᶜT + ifelse(
                    (ᶜz >= 1000) & (ᶜz <= 15000),
                    1 - tanh((ᶜz - FT(8000)) / FT(1000)),
                    FT(0),
                )
            for width in (FT(0), FT(500))
                bs_w = CA.BeresSourceParams{FT}(;
                    (n => getfield(bs_g, n) for n in fieldnames(typeof(bs_g)))...,
                    beres_smoothing_width = width,
                )
                p_w = (;
                    p.precomputed,
                    p.params,
                    p.atmos,
                    p.scratch,
                    non_orographic_gravity_wave = merge(
                        nogw,
                        (; gw_beres_source = bs_w),
                    ),
                )
                CA.compute_beres_convective_heating!(Y, p_w, nogw.ᶜbuoyancy_frequency)
                Q_w = Array(Fields.field2array(gw_Q_conv_ic))[:, 1]
                z_lo_w = CA._beres_gather_bottom(
                    value(gw_zbot),
                    value(gw_ztop),
                    z_floor,
                )
                span_w = value(gw_ztop) - z_lo_w
                prof_w = tuple_value(gw_beres_gprofile, CA.N_BERES_PROFILE)
                heating_w = sum(max.(Q_w[above_floor], 0) .* dz_col[above_floor])
                @test value(gw_beres_active) == 1
                @test span_w / (CA.N_BERES_PROFILE - 1) *
                      (sum(prof_w) - (prof_w[1] + prof_w[end]) / 2) ≈ heating_w rtol =
                    1e-10
                # The taper comes after the smoothing, so the profile is zero at both
                # window edges even when smoothed; the other order would smooth
                # heating back onto the edge points.
                @test prof_w[1] == 0
                @test prof_w[end] == 0
                @test prof_w[2] > 0
            end
        end
    end
end

# ---------------------------------------------------------------------------
# Latent in-cloud heating source (1M + PrognosticEDMFX + heating_latent).
# Second, separate build: the latent source needs 1-moment microphysics and is
# selected at construction. The synthetic draft is saturated and cloudy in a
# band, so the canonical latent heating Q_lat = (1/cp) Σ L_p R_p is nonzero
# there. This exercises the per-draft evaluation of the 1M tendencies with
# the draft vertical velocity (CloudMicrophysics ≥ 0.41).
# ---------------------------------------------------------------------------
@testset "Beres latent in-cloud heating (1M + heating_latent, second build)" begin
    config_latent = CA.AtmosConfig(
        joinpath(
            @__DIR__,
            "../../../../config/model_configs/single_column_beres_nogw_latent_test.yml",
        );
        job_id = "beres_single_column_latent",
        comms_ctx,
    )
    sim_latent = CA.get_simulation(config_latent)
    p_l = sim_latent.integrator.p
    Y_l = sim_latent.integrator.u
    FT_l = eltype(Y_l.c.ρ)
    @test p_l.non_orographic_gravity_wave.gw_beres_source.heating_latent

    thp_l = CA.CAP.thermodynamics_params(p_l.params)
    (; ᶜρʲs, ᶜTʲs, ᶜT, ᶜq_tot_nonnegʲs) = p_l.precomputed
    ᶜz_l = Fields.coordinate_field(Y_l.c).z
    ᶜρ_l = Y_l.c.ρ
    a0 = FT_l(0.05)
    z_lo = FT_l(3000)
    z_hi = FT_l(10000)
    in_band = @. (ᶜz_l >= z_lo) & (ᶜz_l <= z_hi)

    # Saturated, cloudy, slightly supersaturated draft in the band; the draft
    # density and temperature follow the grid mean.
    parent(ᶜρʲs.:(1)) .= parent(ᶜρ_l)
    @. ᶜTʲs.:(1) = ᶜT
    @. ᶜq_tot_nonnegʲs.:(1) =
        CA.TD.q_vap_saturation(thp_l, ᶜT, ᶜρ_l) + FT_l(1e-3) + FT_l(2e-4)
    @. Y_l.c.sgsʲs.:(1).ρa = ifelse(in_band, a0 * ᶜρ_l, FT_l(0))
    @. Y_l.c.sgsʲs.:(1).q_lcl = ifelse(in_band, FT_l(1e-3), FT_l(0))
    @. Y_l.c.sgsʲs.:(1).q_icl = FT_l(0)
    @. Y_l.c.sgsʲs.:(1).q_rai = FT_l(0)
    @. Y_l.c.sgsʲs.:(1).q_sno = FT_l(0)

    ᶜN_l = p_l.non_orographic_gravity_wave.ᶜbuoyancy_frequency
    fill!(ᶜN_l, FT_l(0.012))

    CA.compute_beres_convective_heating!(Y_l, p_l, ᶜN_l)

    Q_ic = Array(Fields.field2array(p_l.non_orographic_gravity_wave.gw_Q_conv_ic))[:, 1]
    z_l = Array(Fields.field2array(ᶜz_l))[:, 1]
    band = (z_l .>= z_lo) .& (z_l .<= z_hi)
    @testset "finite, nonzero in the band, zero outside" begin
        @test all(isfinite, Q_ic)
        @test any(!=(0), Q_ic[band])
        @test all(==(0), Q_ic[.!band])
    end
end

# ---------------------------------------------------------------------------
# The minimum updraft area is a physical threshold, so a Float32 build of the
# same column gives the envelope of the Float64 build (third, separate build).
# ---------------------------------------------------------------------------
@testset "Beres envelope with a near-empty layer: Float32 agrees with Float64" begin
    config32 = CA.AtmosConfig(
        (CA.load_yaml_file(config_file), Dict("FLOAT_TYPE" => "Float32"));
        job_id = "beres_single_column_float32",
        comms_ctx,
        config_files = [config_file],
    )
    sim32 = CA.get_simulation(config32)
    @test eltype(sim32.integrator.u.c.ρ) == Float32
    res32 =
        beres_thin_layer_envelope(sim32.integrator.u, sim32.integrator.p; thin_layer = true)
    res64 = BERES_THIN_LAYER_FLOAT64[]
    @test !isnothing(res64)
    @test res32.active == res64.active == 1
    for key in (:z_bot, :z_top, :h, :Q0, :a_cover)
        @test getproperty(res32, key) ≈ getproperty(res64, key) rtol = 1e-3 atol = 1e-6
    end
end
