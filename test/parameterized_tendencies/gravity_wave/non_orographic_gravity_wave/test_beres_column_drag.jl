# ============================================================================
# Beres (2004) column drag test — single-column vertical propagation
#
# Validates the full Beres forcing pipeline (source spectrum → propagation →
# breaking → momentum deposition) on a single column with analytic wind shear.
# Uses the Beres (2004) §4 squall-line parameters.
#
# Tests: nonzero drag, max drag location (stratospheric), bounded magnitude,
# v-forcing near zero (no meridional wind).
#
# Does NOT test: spectral shape (see test_beres_source.jl) or sphere-grid
# wiring (see test_beres_squall_line_integration.jl).
# ============================================================================

using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore
import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields

@testset "Beres column drag -- single-column with wind shear" begin
    comms_ctx = ClimaComms.SingletonCommsContext()
    config_file = joinpath(
        @__DIR__,
        "../../../../config/model_configs/single_column_beres_nogw_test.yml",
    )
    config = CA.AtmosConfig(
        config_file;
        job_id = "beres_column_drag",
        comms_ctx,
    )
    # Override σ_x to match Beres (2004) §4 squall-line case
    config.parsed_args["beres_sigma_x"] = 2500.0

    simulation = CA.get_simulation(config)
    p = simulation.integrator.p
    Y = simulation.integrator.u

    FT = eltype(Y.c.ρ)

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
        gw_flag,
    ) = p.non_orographic_gravity_wave

    # Get spaces and coordinate fields
    center_space = axes(Y.c)
    ᶜz = Fields.coordinate_field(Y.c).z
    center_z = Array(Fields.field2array(ᶜz))[:, 1]
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
    parent(ᶜu) .= FT(-10.0) .+ center_z .* FT(20.0 / z_max)
    parent(ᶜv) .= FT(0.0)

    # Uniform buoyancy frequency matching Beres §4
    parent(ᶜbuoyancy_frequency) .= FT(0.012)

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
    fill!(gw_flag, FT(0.0))         # tropics flag (AD99 Doppler frame; no effect on Beres)

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
    # Extract results
    # ------------------------------------------------------------------
    uf_data = Array(Fields.field2array(uforcing))[:, 1]
    vf_data = Array(Fields.field2array(vforcing))[:, 1]

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
end
