using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore
import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields

# ============================================================================
# Beres convective GW forcing — sphere integration smoke test
#
# Builds a minimal sphere simulation (h_elem=4), injects Beres (2004) §4
# squall-line parameters, calls non_orographic_gravity_wave_forcing, and
# verifies the output is finite and within the physically expected magnitude
# range (~1e-5 m/s², i.e. ~1 m/s/day).
#
# Does NOT validate spectral shape (see test_beres_source.jl) or vertical
# drag structure (see test_beres_column_drag.jl).  This test catches wiring
# bugs and gross magnitude regressions only.
#
# Beres (2004) §4 squall-line reference values:
#   Q₀ = 0.004 K/s, h = 5000 m, ū_heat = −5 m/s, N = 0.012 s⁻¹,
#   σ_x = 2500 m  (set via beres_squall_integration_test.yml)
# ============================================================================
@testset "Beres convective GW forcing -- sphere integration smoke test" begin
    comms_ctx = ClimaComms.SingletonCommsContext()
    config_files = [
        joinpath(
            @__DIR__,
            "../../../../config/longrun_configs/longrun_aquaplanet_allsky_diagedmf_0M.yml",
        ),
        joinpath(
            @__DIR__,
            "../../../../config/model_configs/beres_squall_integration_test.yml",
        ),
    ]
    config = CA.AtmosConfig(config_files; job_id = "beres_squall_sph", comms_ctx)

    # Minimal grid for fast compilation / testing.
    # h_elem=4 gives a coarse cubed-sphere; z_elem=30 up to 45 km is enough
    # for stratospheric wave breaking.  t_end=0 means no time integration —
    # we only call the forcing routine once.
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

    # Extract NOGW cache
    (; gw_ncval, ᶜbuoyancy_frequency, ᶜlevel, u_waveforcing, v_waveforcing,
        uforcing, vforcing,
        gw_Q0, gw_h_heat, gw_u_heat, gw_v_heat, gw_N_source,
        gw_beres_active, gw_flag,
    ) = p.non_orographic_gravity_wave

    ᶜz = Fields.coordinate_field(Y.c).z

    # ------------------------------------------------------------------
    # Fill atmospheric state with a uniform tropical-like profile.
    # The specific values don't matter much for a smoke test — we just
    # need a physically plausible atmosphere so the forcing routine
    # doesn't hit edge cases.
    # ------------------------------------------------------------------
    ᶜρ = Y.c.ρ
    ᶜu = similar(ᶜρ, FT)
    ᶜv = similar(ᶜρ, FT)

    # Uniform wind matching u_heat — no Doppler shift in the column,
    # so the forcing is purely from the Beres spectrum shape.
    parent(ᶜu) .= FT(-5.0)     # m/s, matches Beres §4 ū_heat
    parent(ᶜv) .= FT(0.0)
    # Buoyancy frequency: 0.012 s⁻¹ matches Beres §4 N
    parent(ᶜbuoyancy_frequency) .= FT(0.012)

    # Compute source/damp levels (sphere mode uses pressure-based search)
    (; source_level, damp_level, source_p_ρ_z_u_v_level) =
        p.non_orographic_gravity_wave
    (; ᶜp) = p.precomputed
    (; gw_source_pressure, gw_damp_pressure) = p.non_orographic_gravity_wave

    input_src = Base.Broadcast.broadcasted(tuple, ᶜp, ᶜρ, ᶜz, ᶜu, ᶜv, ᶜlevel)
    ClimaCore.Operators.column_reduce!(
        source_p_ρ_z_u_v_level,
        input_src,
    ) do (p_prev, ρ_prev, z_prev, u_prev, v_prev, level_prev),
    (p_val, ρ, z, u, v, level)
        if (p_val - gw_source_pressure) <= 0
            return (p_prev, ρ_prev, z_prev, u_prev, v_prev, level_prev)
        else
            return (p_val, ρ, z, u, v, level)
        end
    end

    ᶜρ_source = source_p_ρ_z_u_v_level.:2
    ᶜu_source = source_p_ρ_z_u_v_level.:4
    ᶜv_source = source_p_ρ_z_u_v_level.:5

    input_dmp = Base.Broadcast.broadcasted(tuple, ᶜlevel, ᶜp)
    ClimaCore.Operators.column_reduce!(
        damp_level,
        input_dmp;
        transform = first,
    ) do (level_prev, p_prev), (level, p_val)
        if (p_prev - gw_damp_pressure) >= 0
            return (level, p_val)
        else
            return (level_prev, p_prev)
        end
    end

    # ------------------------------------------------------------------
    # Fill Beres convective-source fields with squall-line values.
    # These are normally computed by compute_beres_convective_heating!
    # from EDMF output; here we inject them directly.
    # ------------------------------------------------------------------
    fill!(gw_Q0, FT(0.004))         # K/s — peak heating rate (Beres §4)
    fill!(gw_h_heat, FT(5000.0))    # m — heating depth (Beres §4)
    fill!(gw_u_heat, FT(-5.0))      # m/s — mean wind in heating region (Beres §4)
    fill!(gw_v_heat, FT(0.0))       # m/s — no meridional wind
    fill!(gw_N_source, FT(0.012))   # s⁻¹ — buoyancy frequency (Beres §4)
    fill!(gw_beres_active, FT(1.0)) # 1.0 = Beres branch active for all columns
    # gw_flag = 0 means "tropics" for the AD99 background spectrum: use
    # Doppler-shifted phase speeds (c - u_source) rather than ground-relative c.
    # This does NOT affect the Beres dispatch.
    fill!(gw_flag, FT(0.0))

    # ------------------------------------------------------------------
    # Call forcing
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
    uf_data = Array(parent(uforcing))

    @testset "Nonzero forcing" begin
        @test all(isfinite, uf_data)
        max_uf = maximum(abs, uf_data)
        @test max_uf > 0
        println("  Sphere: max |uforcing| = $max_uf m/s² ($(max_uf * 86400) m/s/day)")
    end

    @testset "Forcing magnitude is physical" begin
        max_uf = maximum(abs, uf_data)
        # Expected magnitude for squall-line inputs: ~1e-5 m/s² (~1 m/s/day).
        # Lower bound: must produce measurable drag.
        @test max_uf > 1e-8
        # Upper bound: 1e-3 m/s² ≈ 86 m/s/day — well above physical but
        # catches order-of-magnitude blowups (previous bound was 1.0).
        @test max_uf < 1e-3
    end

    @testset "N_source values" begin
        N_data = Array(parent(gw_N_source))
        @test all(isfinite, N_data)
    end
end
