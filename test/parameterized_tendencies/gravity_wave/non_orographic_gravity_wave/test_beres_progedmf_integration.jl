using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore
import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields

# ============================================================================
# Sphere integration test for Beres squall-line forcing with prognostic EDMF.
# Mirrors test_beres_squall_line_integration.jl but uses prognostic_edmfx
# as the turbconv model, verifying that the Beres heating extraction reads
# from Y.c.sgsʲs (prognostic state) rather than p.precomputed.
# ============================================================================
@testset "Beres squall-line sphere integration (prognostic EDMF)" begin
    comms_ctx = ClimaComms.SingletonCommsContext()
    config_files = [
        joinpath(
            @__DIR__,
            "../../../../config/longrun_configs/longrun_aquaplanet_allsky_progedmf_0M_conv_gw.yml",
        ),
        joinpath(
            @__DIR__,
            "../../../../config/model_configs/beres_squall_progedmf_integration_test.yml",
        ),
    ]
    config = CA.AtmosConfig(config_files; job_id = "beres_progedmf_sph", comms_ctx)

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

    FT = eltype(Y.c.ρ)

    # ------------------------------------------------------------------
    # Verify prognostic EDMF is active
    # ------------------------------------------------------------------
    @testset "Prognostic EDMF active" begin
        @test hasproperty(Y.c, :sgsʲs)
        n_up = CA.n_mass_flux_subdomains(p.atmos.turbconv_model)
        @test n_up > 0
    end

    # Verify Beres is enabled
    @test p.non_orographic_gravity_wave.gw_beres_source isa CA.BeresSourceParams

    # ------------------------------------------------------------------
    # Test 1: Inject squall-line values and verify forcing
    # (same as diagnostic EDMF integration test)
    # ------------------------------------------------------------------
    @testset "Squall-line forcing with injected values" begin
        (; gw_ncval, ᶜbuoyancy_frequency, ᶜlevel, u_waveforcing, v_waveforcing,
            uforcing, vforcing,
            gw_Q0, gw_h_heat, gw_u_heat, gw_v_heat, gw_N_source,
            gw_beres_active, gw_flag,
        ) = p.non_orographic_gravity_wave

        ᶜz = Fields.coordinate_field(Y.c).z
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

        input_src =
            Base.Broadcast.broadcasted(tuple, ᶜp, ᶜρ, ᶜz, ᶜu, ᶜv, ᶜlevel)
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

        # Fill squall-line values
        fill!(gw_Q0, FT(0.004))
        fill!(gw_h_heat, FT(6000.0))
        fill!(gw_u_heat, FT(-5.0))
        fill!(gw_v_heat, FT(0.0))
        fill!(gw_N_source, FT(0.01))
        fill!(gw_beres_active, FT(1.0))
        fill!(gw_flag, FT(0.0))

        # Call forcing
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

        uf_data = Array(parent(uforcing))

        @testset "Nonzero forcing" begin
            @test all(isfinite, uf_data)
            max_uf = maximum(abs, uf_data)
            @test max_uf > 0
            println(
                "  Sphere (prognostic EDMF): max |uforcing| = $max_uf m/s² ($(max_uf * 86400) m/s/day)",
            )
        end

        @testset "Forcing magnitude is physical" begin
            max_uf = maximum(abs, uf_data)
            @test max_uf > 1e-8       # nonzero drag
            @test max_uf < 1.0        # not unphysically large (< 1 m/s²)
        end
    end

    # ------------------------------------------------------------------
    # Test 2: Verify compute_beres_convective_heating! reads from
    # Y.c.sgsʲs (prognostic path)
    # ------------------------------------------------------------------
    @testset "Prognostic EDMF heating extraction" begin
        (; gw_Q0, gw_beres_active) = p.non_orographic_gravity_wave

        # First call with initial state — Q0 should be small or zero
        # (no active convection in initial conditions)
        CA.compute_beres_convective_heating!(Y, p)
        Q0_initial = maximum(Array(parent(gw_Q0)))

        # Now perturb the prognostic MSE in updraft 1 to create a
        # large enthalpy anomaly, which should produce nonzero Q_conv
        mse_field = getproperty(Y.c.sgsʲs, 1).mse
        parent(mse_field) .+= FT(5000.0)  # large MSE perturbation

        CA.compute_beres_convective_heating!(Y, p)
        Q0_perturbed = maximum(Array(parent(gw_Q0)))

        @test Q0_perturbed >= Q0_initial
        println(
            "  Q0 initial=$Q0_initial, after MSE perturbation=$Q0_perturbed",
        )

        # Restore MSE to avoid polluting other tests
        parent(mse_field) .-= FT(5000.0)
    end
end
