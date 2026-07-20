using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA

const FT = Float32

const TEST_GRID = CA.ColumnGrid(FT; z_elem = 10, z_max = 3e3, z_stretch = false)
const TEST_PARAMS = CA.ClimaAtmosParameters(FT)
preset_model(preset_kwargs) =
    CA.AtmosModel(TEST_GRID; params = TEST_PARAMS, preset_kwargs...)

# ============================================================================
# Model presets — NamedTuples of AtmosModel kwargs; cheap structural tests
# ============================================================================

@testset "Model preset physics defaults" begin
    dry = preset_model(CA.Presets.dry())
    @test dry isa CA.AtmosModel
    @test dry.microphysics_model isa CA.DryModel

    equil = preset_model(CA.Presets.equil_moist_0m())
    @test equil.microphysics_model isa CA.EquilibriumMicrophysics0M
    @test equil.cloud_model isa CA.GridScaleCloud
    # SST/insolation are the AtmosModel defaults (not restated by the preset,
    # so a case setup can override them)
    @test equil.surface.temperature isa CA.SurfaceConditions.AnalyticTemperature
    @test equil.insolation isa CA.IdealizedInsolation

    nonequil = preset_model(CA.Presets.nonequil_moist_1m())
    @test nonequil.microphysics_model isa CA.NonEquilibriumMicrophysics1M
    @test nonequil.cloud_model isa CA.GridScaleCloud
    @test nonequil.surface.temperature isa CA.SurfaceConditions.AnalyticTemperature
    @test nonequil.insolation isa CA.IdealizedInsolation
end

@testset "Model preset kwargs pass through to AtmosModel" begin
    # A kwarg the preset doesn't set should come through unchanged.
    m = preset_model(CA.Presets.dry(; disable_surface_flux_tendency = true))
    @test m.disable_surface_flux_tendency == true

    # A kwarg the preset *does* set should be overridable by the caller.
    m = preset_model(
        CA.Presets.equil_moist_0m(; microphysics_model = CA.DryModel()),
    )
    @test m.microphysics_model isa CA.DryModel
    # Other equil defaults should still be in place:
    @test m.cloud_model isa CA.GridScaleCloud
end

@testset "Prognostic EDMF preset" begin
    prog = preset_model(CA.Presets.prognostic_edmf(FT))
    @test prog.turbconv_model isa CA.PrognosticEDMFX
    @test prog.edmfx_model.entr_model isa CA.InvZEntrainment
    @test prog.edmfx_model.detr_model isa CA.BuoyancyVelocityDetrainment
    @test prog.edmfx_model.sgs_mass_flux === Val(true)
    @test prog.edmfx_model.sgs_diffusive_flux === Val(true)
    @test prog.edmfx_model.nh_pressure === Val(true)
    @test prog.edmfx_model.vertical_diffusion === Val(true)
    @test prog.edmfx_model.filter === Val(true)

    # area_fraction kwarg flows through to the turbconv model
    custom = preset_model(CA.Presets.prognostic_edmf(FT; area_fraction = FT(5e-5)))
    @test custom.turbconv_model.a_half == FT(5e-5)

    # Composing with a different microphysics scheme still gives an EDMF model
    hybrid = preset_model(
        CA.Presets.prognostic_edmf(
            FT; microphysics_model = CA.NonEquilibriumMicrophysics1M(),
        ),
    )
    @test hybrid.microphysics_model isa CA.NonEquilibriumMicrophysics1M
    @test hybrid.turbconv_model isa CA.PrognosticEDMFX
end

# ============================================================================
# Simulation presets — smoke tests (build a real AtmosSimulation)
# ============================================================================

@testset "aquaplanet simulation preset" begin
    sim = CA.Presets.aquaplanet(FT; t_end = 3600)
    @test sim isa CA.AtmosSimulation
    @test sim.integrator.p.atmos.microphysics_model isa CA.EquilibriumMicrophysics0M
end
