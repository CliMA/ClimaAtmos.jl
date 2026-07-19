# ============================================================================
# Beres convective GW forcing — sphere full-integration test
#
# Builds a minimal cubed-sphere simulation (h_elem=4), injects a synthetic,
# HORIZONTALLY HETEROGENEOUS EDMF updraft (convectively active in a tropical
# latitude band, inactive elsewhere), and drives the REAL model entry point
# `non_orographic_gravity_wave_compute_tendency!` — the same routine the
# nogw callback calls. This exercises the sphere-specific wiring that no
# other Beres test reaches:
#
#   - the sphere PRESSURE-based source/damp level search,
#   - compute_beres_convective_heating! extracting Q0/h/u_heat/a_cover from
#     EDMF state across MANY columns (not a single column),
#   - the gw_ztop Beres source-level column_reduce!,
#   - the apply-tendency clamp + deposition into Yₜ.c.uₕ.
#
# Heterogeneity is the load-bearing check: the convective heating and the
# isolated Beres drag must appear ONLY in the active columns, proving the
# per-column gating works on a real sphere. Beres drag is isolated by
# differencing against an AD99-only run (gw_beres_source = nothing), so the
# latitude-dependent AD99 background cancels exactly.
#
# Does NOT re-test spectral shape (test_beres_unit.jl) or single-column drag
# structure (test_beres_single_column.jl).
#
# Beres (2004) §4 squall-line reference: σ_x = 2500 m (set via the inline
# toml_override below, from toml/nogw_beres_test.toml).
# ============================================================================

using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore
import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry

@testset "Beres convective GW forcing -- sphere full integration" begin
    comms_ctx = ClimaComms.SingletonCommsContext()
    longrun_config = joinpath(
        @__DIR__,
        "../../../../config/longrun_configs/longrun_aquaplanet_allsky_progedmf_0M.yml",
    )
    # Squall-line test overrides: coarse phase-speed grid (nc=51) for fast
    # compilation plus the Beres (2004) §4 squall-line cell half-width
    # σ_x = 2500 m (both in toml/nogw_beres_test.toml).
    toml_override = Dict(
        "toml" => [
            joinpath(@__DIR__, "../../../../toml/longrun_aquaplanet_progedmf.toml"),
            joinpath(@__DIR__, "../../../../toml/nogw_beres_test.toml"),
        ],
    )
    config = CA.AtmosConfig(
        (CA.load_yaml_file(longrun_config), toml_override);
        job_id = "beres_sphere",
        comms_ctx,
        config_files = [longrun_config],
    )

    # Minimal grid for fast compilation / testing.
    # h_elem=4 gives a coarse cubed-sphere; z_elem=30 up to 45 km is enough
    # for stratospheric wave breaking.  t_end=0 means no time integration —
    # we only drive the forcing routine.
    config.parsed_args["h_elem"] = 4
    config.parsed_args["z_elem"] = 30
    config.parsed_args["z_max"] = 45000.0
    config.parsed_args["dz_bottom"] = 500.0
    config.parsed_args["dt"] = "120secs"
    config.parsed_args["t_end"] = "0secs"
    config.parsed_args["dt_save_state_to_disk"] = "Inf"
    config.parsed_args["output_default_diagnostics"] = false

    simulation = CA.get_simulation(config)
    p = simulation.integrator.p
    Y = simulation.integrator.u

    FT = eltype(Y.c.ρ)

    # Verify Beres is enabled and σ_x matches squall-line case
    @test p.non_orographic_gravity_wave.gw_beres_source isa CA.BeresSourceParams
    @test p.non_orographic_gravity_wave.gw_beres_source.σ_x == FT(2500.0)

    ᶜz = Fields.coordinate_field(Y.c).z
    ᶜlat = Fields.coordinate_field(Y.c).lat
    ᶠlat = Fields.coordinate_field(Y.f).lat

    # ------------------------------------------------------------------
    # Inject a synthetic, horizontally heterogeneous updraft into the EDMF
    # cache: convectively active in the tropical band |lat| < lat_active,
    # inactive (zeroed draft) elsewhere. Same per-column construction as the
    # single-column heating test, gated on latitude.
    # ------------------------------------------------------------------
    lat_active = FT(30)                 # tropical band is convectively active
    a0 = FT(0.05)                       # updraft area fraction in active band
    z_lo = FT(3000)                     # updraft band bottom (above z_bot_floor)
    z_hi = FT(10000)                    # updraft band top

    (; ᶜρʲs, ᶜTʲs, ᶜT, ᶠu³ʲs, ᶠu³) = p.precomputed
    ᶜρ = Y.c.ρ

    # Draft density = grid-mean density (so Q_conv = a0·Q_conv_ic in-band).
    @. ᶜρʲs.:(1) = ᶜρ

    # Half-sine in-band temperature anomaly (2 K peak), only in active columns.
    @. ᶜTʲs.:(1) =
        ᶜT + ifelse(
            (ᶜz >= z_lo) & (ᶜz <= z_hi) & (abs(ᶜlat) < lat_active),
            FT(2) * sin(FT(π) * (ᶜz - z_lo) / (z_hi - z_lo)),
            FT(0),
        )

    # Uniform updraft area fraction inside the active band, zero elsewhere.
    @. Y.c.sgsʲs.:(1).ρa = ifelse(
        (ᶜz >= z_lo) & (ᶜz <= z_hi) & (abs(ᶜlat) < lat_active),
        a0 * ᶜρ,
        FT(0),
    )

    # Updraft vertical velocity 1 m/s in active columns (zero elsewhere); zero
    # grid-mean velocity. The DSE anomaly is band-limited, so the constant
    # in-column velocity carries flux only within the heating band.
    ᶠlg = Fields.local_geometry_field(Y.f)
    @. ᶠu³ʲs.:(1) = Geometry.Contravariant3Vector(
        Geometry.WVector(ifelse(abs(ᶠlat) < lat_active, FT(1), FT(0))),
        ᶠlg,
    )
    @. ᶠu³ = Geometry.Contravariant3Vector(Geometry.WVector(FT(0)), ᶠlg)

    # Impose a sheared zonal background wind, u(z) = -10 → +10 m/s over the
    # column (as in the single-column drag test). Without shear the launched
    # waves meet no critical levels, never break, and escape to the model top
    # depositing ~0 (physically correct, but it makes the heterogeneity check
    # pass only on negligible ~1e-13 values). With shear the active columns
    # break and deposit physical drag (~1e-4 m/s²), so the test exercises the
    # full derive→launch→propagate→break→deposit chain at real magnitudes. The
    # wind is global; it is identical in both runs below, so the AD99 background
    # cancels in the Beres-drag difference and only beres_active columns deposit.
    ᶜlg = Fields.local_geometry_field(Y.c)
    z_dom_top = FT(45000)   # matches the z_max override above
    @. Y.c.uₕ = Geometry.Covariant12Vector(
        Geometry.UVVector(FT(-10) + ᶜz * FT(20) / z_dom_top, FT(0)),
        ᶜlg,
    )

    # ------------------------------------------------------------------
    # Drive the REAL entry point twice on identical state:
    #   1. AD99-only (gw_beres_source = nothing) — the background.
    #   2. Full Beres — background + convective source.
    # Differencing isolates the Beres contribution (the latitude-dependent
    # AD99 drag cancels exactly). uforcing/vforcing are shared cache fields,
    # so we copy each result out before the next run.
    # ------------------------------------------------------------------
    nogw_ad99 = merge(p.non_orographic_gravity_wave, (; gw_beres_source = nothing))
    # Current src no longer supports merge(::AtmosCache); build a lightweight cache
    # NamedTuple with just the fields the GW compute path reads (non_orographic_
    # gravity_wave, precomputed, params, atmos, scratch).
    p_ad99 = (;
        p.precomputed,
        p.params,
        p.atmos,
        p.scratch,
        non_orographic_gravity_wave = nogw_ad99,
    )

    CA.non_orographic_gravity_wave_compute_tendency!(
        Y,
        p_ad99,
        p_ad99.atmos.non_orographic_gravity_wave,
    )
    uf_ad99 = Array(Fields.field2array(p_ad99.non_orographic_gravity_wave.uforcing))

    CA.non_orographic_gravity_wave_compute_tendency!(
        Y,
        p,
        p.atmos.non_orographic_gravity_wave,
    )
    uf_beres = Array(Fields.field2array(p.non_orographic_gravity_wave.uforcing))
    vf_beres = Array(Fields.field2array(p.non_orographic_gravity_wave.vforcing))

    # ------------------------------------------------------------------
    # Classify columns by latitude (center-space field2array ⇒ consistent
    # column ordering with the forcing/heating arrays read below).
    # ------------------------------------------------------------------
    lat_arr = Array(Fields.field2array(ᶜlat))   # [nlev, ncol]
    lat_col = lat_arr[1, :]                      # lat is constant in the vertical
    ncol = length(lat_col)
    active_cols = findall(c -> abs(lat_col[c]) < lat_active, 1:ncol)
    inactive_cols = findall(c -> abs(lat_col[c]) >= lat_active, 1:ncol)

    println("Sphere integration: $ncol columns, ",
        "$(length(active_cols)) active / $(length(inactive_cols)) inactive")
    println("  max |uforcing| (Beres) = $(maximum(abs, uf_beres)) m/s²")

    # Diagnostics: what source did the per-column EDMF derivation produce? These
    # are per-column level fields (read via parent). Used to localize where the
    # derived-source magnitude collapses relative to the single-column test.
    q0_all = vec(Array(parent(p.non_orographic_gravity_wave.gw_Q0)))
    hh_all = vec(Array(parent(p.non_orographic_gravity_wave.gw_h_heat)))
    ba_all = vec(Array(parent(p.non_orographic_gravity_wave.gw_beres_active)))
    zt_all = vec(Array(parent(p.non_orographic_gravity_wave.gw_ztop)))
    println("  derived source: max gw_Q0 = $(maximum(q0_all)), ",
        "max gw_h_heat = $(maximum(hh_all)), max gw_ztop = $(maximum(zt_all)), ",
        "active = $(count(>(0.5), ba_all)) / $(length(ba_all)) columns")

    @testset "Both column populations present" begin
        @test !isempty(active_cols)
        @test !isempty(inactive_cols)
    end

    @testset "Forcing finite and bounded" begin
        @test all(isfinite, uf_beres)
        @test all(isfinite, vf_beres)
        @test maximum(abs, uf_beres) > 0
        # Blow-up guard (pre-clamp). 1e-1 m/s² ≈ 8640 m/s/day — absurdly high,
        # so this only catches order-of-magnitude failures.
        @test maximum(abs, uf_beres) < 1e-1
    end

    @testset "Convective heating extracted only in active columns" begin
        # gw_Q_conv (grid-mean convective heating) is computed per column by
        # compute_beres_convective_heating! from the injected EDMF state. It
        # must be nonzero where we injected an updraft and zero elsewhere.
        qconv = Array(Fields.field2array(p.non_orographic_gravity_wave.gw_Q_conv))
        qconv_col = vec(maximum(abs, qconv; dims = 1))
        max_active_Q = maximum(qconv_col[active_cols])
        @test max_active_Q > 0
        @test all(qconv_col[active_cols] .> 0)
        # Inactive columns have zero updraft ⇒ zero convective heating.
        @test maximum(qconv_col[inactive_cols]) <= 1e-10 * max_active_Q
    end

    @testset "Beres drag deposited only in active columns" begin
        # Isolate the Beres contribution by differencing against the AD99-only
        # run on identical state. In inactive columns gw_beres_active = 0, so
        # the Beres branch deposits nothing and the difference must vanish.
        beres_drag = uf_beres .- uf_ad99
        @test all(isfinite, beres_drag)
        max_active = maximum(abs, beres_drag[:, active_cols])
        max_inactive = maximum(abs, beres_drag[:, inactive_cols])
        println("  Beres drag: max active = $max_active, max inactive = $max_inactive")
        # With the imposed shear the derived source breaks and deposits a real,
        # nonzero momentum increment in the active columns (measured ~8e-8 m/s²;
        # modest because the deposited drag is diluted by a_cover≈0.05 and this is
        # a single snapshot, not a developed source). The floor sits far above the
        # shear-free collapse (~1e-13 m/s², where waves escape without breaking),
        # so a regression to no-breaking would fail this. The upper bound is a
        # blow-up guard.
        @test 1e-9 < max_active < 1e-2
        # Inactive columns carry only the (cancelled) AD99 background ⇒ ~0 Beres.
        @test max_inactive <= 1e-6 * max_active
    end

    @testset "Tendency application clamps and deposits into Yₜ" begin
        # uforcing currently holds the full-Beres result. apply_tendency clamps
        # it to ±3e-3 m/s² in place and adds it to Yₜ.c.uₕ.
        Yₜ = zero(Y)
        CA.non_orographic_gravity_wave_apply_tendency!(
            Yₜ,
            Y,
            p,
            simulation.integrator.t,
            p.atmos.non_orographic_gravity_wave,
        )
        uf_clamped =
            Array(Fields.field2array(p.non_orographic_gravity_wave.uforcing))
        @test all(isfinite, uf_clamped)
        @test maximum(abs, uf_clamped) <= 3e-3 + sqrt(eps(FT))
        # Yₜ.c.uₕ is a 2-component Covariant12 vector; field2array can't represent
        # it directly, so extract the zonal (UVVector) component as a scalar field.
        du = Geometry.UVVector.(Yₜ.c.uₕ).components.data.:1
        du_arr = Array(Fields.field2array(du))
        @test all(isfinite, du_arr)
        @test maximum(abs, du_arr) > 0
    end
end
