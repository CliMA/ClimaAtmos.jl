using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore
import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields

# ============================================================================
# Sphere integration test for Beres squall-line forcing.
# Uses the same physics config as the working CI run
# (longrun_aquaplanet_allsky_diagedmf_0M_conv_gw.yml) but with a minimal grid
# and t_end=0. Builds the simulation, injects hardcoded squall-line values,
# calls non_orographic_gravity_wave_forcing, and verifies nonzero drag.
# ============================================================================
@testset "Beres squall-line sphere integration" begin
    comms_ctx = ClimaComms.SingletonCommsContext()
    config_files = [
        joinpath(@__DIR__, "../../../../config/longrun_configs/longrun_aquaplanet_allsky_diagedmf_0M_conv_gw.yml"),
        joinpath(@__DIR__, "../../../../config/model_configs/beres_squall_integration_test.yml"),
    ]
    config = CA.AtmosConfig(config_files; job_id = "beres_squall_sph", comms_ctx)

    # Override for a minimal test grid
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

    # Get FT from the simulation (Float32 for longrun configs)
    FT = eltype(Y.c.ρ)

    # Verify Beres is enabled
    @test p.non_orographic_gravity_wave.gw_beres_source isa CA.BeresSourceParams

    # Extract NOGW cache
    (; gw_ncval, ᶜbuoyancy_frequency, ᶜlevel, u_waveforcing, v_waveforcing,
       uforcing, vforcing,
       gw_Q0, gw_h_heat, gw_u_heat, gw_v_heat, gw_N_source,
       gw_beres_active, gw_flag,
    ) = p.non_orographic_gravity_wave

    # Get coordinates
    ᶜz = Fields.coordinate_field(Y.c).z

    # ------------------------------------------------------------------
    # Fill atmospheric state with tropical-like profiles
    # ------------------------------------------------------------------
    ᶜρ = Y.c.ρ
    ᶜu = similar(ᶜρ, FT)
    ᶜv = similar(ᶜρ, FT)

    parent(ᶜu) .= FT(-5.0)
    parent(ᶜv) .= FT(0.0)
    parent(ᶜbuoyancy_frequency) .= FT(0.01)

    # Compute source/damp levels (sphere uses pressure-based)
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
    # Fill squall-line values
    # ------------------------------------------------------------------
    fill!(gw_Q0, FT(0.004))
    fill!(gw_h_heat, FT(6000.0))
    fill!(gw_u_heat, FT(-5.0))
    fill!(gw_v_heat, FT(0.0))
    fill!(gw_N_source, FT(0.01))
    fill!(gw_beres_active, FT(1.0))
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
        @test max_uf > 1e-8       # nonzero drag
        @test max_uf < 1.0        # not unphysically large (< 1 m/s²)
    end

    @testset "N_source values" begin
        N_data = Array(parent(gw_N_source))
        @test all(isfinite, N_data)
    end
end
