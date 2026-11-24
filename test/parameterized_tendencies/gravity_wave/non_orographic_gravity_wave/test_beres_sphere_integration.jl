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
#   - the compute-tendency clamp and the apply-tendency deposition into Yₜ.c.uₕ.
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

# Draft moist static energy at which the draft temperature, evaluated from the draft
# state at the grid-mean pressure `p_air` (`CA.draft_thermo_state`), is `T`, found by
# secant iterations from the current draft mse (the 0-moment scheme diagnoses the
# temperature with a saturation adjustment).
function draft_mse_at(mm, thp, T_min, T, sgsʲ, Φ, p_air)
    sgs = ClimaCore.Utilities.drop_auto_broadcasters(sgsʲ)
    T_at(mse) = CA.thermo_T(
        CA.draft_thermo_state(mm, merge(sgs, (; mse)), Φ, p_air, thp, T_min),
    )
    m₀ = sgs.mse
    T₀ = T_at(m₀)
    m₁ = m₀ + 1004 * (T - T₀)
    for _ in 1:30
        T₁ = T_at(m₁)
        (T₁ == T || T₁ == T₀) && break
        (m₀, T₀, m₁) = (m₁, T₁, m₁ + (T - T₁) * (m₁ - m₀) / (T₁ - T₀))
    end
    return m₁
end

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
            joinpath(@__DIR__, "../../../../toml/default_parameters.toml"),
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
    # This grid lies outside the range the two-scale stretching admits.
    config.parsed_args["z_stretch"] = false
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

    (; ᶜρʲs, ᶜT, ᶠu³) = p.precomputed
    ᶜρ = Y.c.ρ

    # Draft density = grid-mean density (so Q_conv = a0·Q_conv_ic in-band).
    @. ᶜρʲs.:(1) = ᶜρ

    # Half-sine in-band temperature anomaly (2 K peak), only in active columns,
    # set through the draft mse, from which the draft temperature is evaluated.
    thp = CA.CAP.thermodynamics_params(p.params)
    mse_at(T, sgsʲ, Φ, p_air) = draft_mse_at(
        p.atmos.microphysics_model, thp, CA.CAP.T_min_sgs(p.params), T, sgsʲ, Φ, p_air,
    )
    @. Y.c.sgsʲs.:(1).mse = mse_at(
        ᶜT + ifelse(
            (ᶜz >= z_lo) & (ᶜz <= z_hi) & (abs(ᶜlat) < lat_active),
            FT(2) * sin(FT(π) * (ᶜz - z_lo) / (z_hi - z_lo)),
            FT(0),
        ),
        Y.c.sgsʲs.:(1),
        p.core.ᶜΦ,
        p.precomputed.ᶜp,
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
    @. Y.f.sgsʲs.:(1).u₃ = Geometry.Covariant3Vector(
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

    # The 45 km top lies below `nogw_damp_pressure`, so the first call warns that the
    # damping layer is widened. The warning fires once per run, and the startup
    # callback has already used it; reset its flag.
    CA.NOGW_DAMP_LAYER_CHECKED[] = false
    @test_logs (:warn, r"Fewer than") match_mode = :any CA.non_orographic_gravity_wave_compute_tendency!(
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

    # The AD99 launch and damping levels that `compute_tendency!` picked, checked in
    # every column against the selection rules written out from the column's
    # pressure profile. The 45 km top lies below `nogw_damp_pressure` (85 Pa, about
    # 50 km), so the pressure rule alone would leave the top cell as the whole
    # damping layer; the guard must widen it to `NOGW_MIN_DAMP_CELLS` cells.
    @testset "Launch and damp levels follow the pressure rules" begin
        nogw = p.non_orographic_gravity_wave
        p_arr = Array(Fields.field2array(p.precomputed.ᶜp))     # [nlev, ncol]
        nlev = size(p_arr, 1)
        level_values(f) = round.(Int, vec(Array(Fields.field2array(f))))
        source_level = level_values(nogw.source_p_ρ_z_u_v_level.:6)
        damp_level = level_values(nogw.damp_level)
        @test length(source_level) == length(damp_level) == size(p_arr, 2)
        # Launch: the highest level whose pressure exceeds the source pressure.
        expected_source = [
            findlast(>(nogw.gw_source_pressure), p_arr[:, col]) for
            col in axes(p_arr, 2)
        ]
        @test source_level == expected_source
        # Damping layer: from the lowest level whose pressure is below the damp
        # pressure, or from the top cell when no level is, and always at least the
        # top `NOGW_MIN_DAMP_CELLS` cells.
        expected_damp = map(axes(p_arr, 2)) do col
            last_above = findlast(>=(nogw.gw_damp_pressure), p_arr[:, col])
            pressure_rule = min(something(last_above, 0) + 1, nlev)
            min(pressure_rule, CA.highest_damp_level(nlev))
        end
        @test damp_level == expected_damp
        @test minimum(p_arr[end, :]) > nogw.gw_damp_pressure    # setup guard
        @test all(==(nlev - CA.NOGW_MIN_DAMP_CELLS + 1), damp_level)
    end

    @testset "Forcing finite and bounded" begin
        @test all(isfinite, uf_beres)
        @test all(isfinite, vf_beres)
        @test maximum(abs, uf_beres) > 0
        # The compute step limits the drag magnitude.
        @test maximum(hypot.(uf_beres, vf_beres)) <= 3e-3 * sqrt(2) + sqrt(eps(FT))
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

    @testset "Tendency application deposits the clamped forcing into Yₜ" begin
        # uforcing holds the full-Beres result, magnitude-limited by the compute step.
        # apply_tendency adds it to Yₜ.c.uₕ.
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
        vf_clamped =
            Array(Fields.field2array(p.non_orographic_gravity_wave.vforcing))
        @test all(isfinite, uf_clamped)
        @test maximum(hypot.(uf_clamped, vf_clamped)) <= 3e-3 * sqrt(2) + sqrt(eps(FT))
        # Yₜ.c.uₕ is a 2-component Covariant12 vector; field2array can't represent
        # it directly, so extract the zonal (UVVector) component as a scalar field.
        du = Geometry.UVVector.(Yₜ.c.uₕ).components.data.:1
        du_arr = Array(Fields.field2array(du))
        @test all(isfinite, du_arr)
        @test maximum(abs, du_arr) > 0
    end

    # N in the bottom and top cells uses the one-sided dT/dz across the face between
    # each end cell and its only neighbour, recomputed here from the column's T, z and
    # moist heat capacity. The initial state is isothermal near the top, where half of a
    # zero gradient equals the full one, so T gets an added 2 K/km first. Runs last
    # because it changes ᶜT and the NOGW cache.
    @testset "Buoyancy frequency in the end cells uses a one-sided dT/dz" begin
        ᶜz = Fields.coordinate_field(Y.c).z
        @. ᶜT += FT(2e-3) * ᶜz
        CA.non_orographic_gravity_wave_compute_tendency!(
            Y,
            p,
            p.atmos.non_orographic_gravity_wave,
        )
        (; ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
        thermo_params = CA.Parameters.thermodynamics_params(p.params)
        grav = CA.Parameters.grav(p.params)
        ᶜcp_m = @. CA.TD.cp_m(thermo_params, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice)
        T_arr = Array(Fields.field2array(ᶜT))
        z_arr = Array(Fields.field2array(ᶜz))
        cp_arr = Array(Fields.field2array(ᶜcp_m))
        N_arr = Array(
            Fields.field2array(p.non_orographic_gravity_wave.ᶜbuoyancy_frequency),
        )
        n = size(T_arr, 1)
        N_from(dTdz, k) = map(axes(T_arr, 2)) do col
            N² = grav / T_arr[k, col] * (dTdz[col] + grav / cp_arr[k, col])
            N² < 2.5e-5 ? sqrt(2.5e-5) : sqrt(abs(N²))
        end
        dTdz_bottom = (T_arr[2, :] .- T_arr[1, :]) ./ (z_arr[2, :] .- z_arr[1, :])
        dTdz_top = (T_arr[n, :] .- T_arr[n - 1, :]) ./ (z_arr[n, :] .- z_arr[n - 1, :])
        @test N_arr[1, :] ≈ N_from(dTdz_bottom, 1) rtol = 1e-5
        @test N_arr[n, :] ≈ N_from(dTdz_top, n) rtol = 1e-5
        # setup guard: half the gradient would give a different N in these cells
        @test !isapprox(N_from(dTdz_top ./ 2, n), N_from(dTdz_top, n); rtol = 1e-3)
        @test !isapprox(N_from(dTdz_bottom ./ 2, 1), N_from(dTdz_bottom, 1); rtol = 1e-3)
    end
end
