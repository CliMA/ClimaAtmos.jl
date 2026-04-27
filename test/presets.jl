using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA

const FT = Float32

# ============================================================================
# Model presets — cheap structural tests
# ============================================================================

@testset "Model preset physics defaults" begin
    dry = CA.Presets.dry()
    @test dry isa CA.AtmosModel
    @test dry.microphysics_model isa CA.DryModel

    equil = CA.Presets.equil_moist_0m()
    @test equil.microphysics_model isa CA.EquilibriumMicrophysics0M
    @test equil.cloud_model isa CA.GridScaleCloud
    @test equil.surface_model isa CA.PrescribedSST
    @test equil.sfc_temperature isa CA.ZonallySymmetricSST
    @test equil.insolation isa CA.IdealizedInsolation

    nonequil = CA.Presets.nonequil_moist_1m()
    @test nonequil.microphysics_model isa CA.NonEquilibriumMicrophysics1M
    @test nonequil.cloud_model isa CA.GridScaleCloud
    @test nonequil.surface_model isa CA.PrescribedSST
    @test nonequil.sfc_temperature isa CA.ZonallySymmetricSST
    @test nonequil.insolation isa CA.IdealizedInsolation
end

@testset "Model preset kwargs pass through to AtmosModel" begin
    # A kwarg the preset doesn't set should come through unchanged.
    m = CA.Presets.dry(; disable_surface_flux_tendency = true)
    @test m.disable_surface_flux_tendency == true

    # A kwarg the preset *does* set should be overridable by the caller.
    m = CA.Presets.equil_moist_0m(; microphysics_model = CA.DryModel())
    @test m.microphysics_model isa CA.DryModel
    # Other equil defaults should still be in place:
    @test m.sfc_temperature isa CA.ZonallySymmetricSST
end

@testset "Diagnostic/prognostic EDMF presets" begin
    diag = CA.Presets.diagnostic_edmf(FT)
    @test diag.turbconv_model isa CA.DiagnosticEDMFX
    @test diag.edmfx_model.entr_model isa CA.InvZEntrainment
    @test diag.edmfx_model.detr_model isa CA.BuoyancyVelocityDetrainment
    @test diag.edmfx_model.sgs_mass_flux === Val(true)
    @test diag.edmfx_model.sgs_diffusive_flux === Val(true)
    @test diag.edmfx_model.nh_pressure === Val(true)
    # diagnostic does not enable updraft vertical diffusion or the filter
    @test diag.edmfx_model.vertical_diffusion === Val(false)
    @test diag.edmfx_model.filter === Val(false)

    prog = CA.Presets.prognostic_edmf(FT)
    @test prog.turbconv_model isa CA.PrognosticEDMFX
    # prognostic adds these two on top of diagnostic
    @test prog.edmfx_model.vertical_diffusion === Val(true)
    @test prog.edmfx_model.filter === Val(true)

    # area_fraction kwarg flows through to the turbconv model
    custom = CA.Presets.diagnostic_edmf(FT; area_fraction = FT(5e-5))
    @test custom.turbconv_model.a_half == FT(5e-5)

    # Composing with a different microphysics scheme still gives an EDMF model
    hybrid = CA.Presets.diagnostic_edmf(
        FT; microphysics_model = CA.NonEquilibriumMicrophysics1M(),
    )
    @test hybrid.microphysics_model isa CA.NonEquilibriumMicrophysics1M
    @test hybrid.turbconv_model isa CA.DiagnosticEDMFX
end

# ============================================================================
# Simulation presets — smoke tests (build a real AtmosSimulation)
# ============================================================================

@testset "aquaplanet simulation preset" begin
    sim = CA.Presets.aquaplanet(FT; t_end = 3600)
    @test sim isa CA.AtmosSimulation
    @test sim.integrator.p.atmos.microphysics_model isa CA.EquilibriumMicrophysics0M
end

@testset "baroclinic_wave simulation preset" begin
    sim = CA.Presets.baroclinic_wave(FT)
    @test sim isa CA.AtmosSimulation
    @test sim.integrator.p.atmos.microphysics_model isa CA.DryModel
    @test sim.integrator.p.atmos.disable_surface_flux_tendency == true
end

@testset "bomex simulation preset" begin
    sim = CA.Presets.bomex(FT)
    @test sim isa CA.AtmosSimulation
    @test sim.integrator.p.atmos.microphysics_model isa CA.EquilibriumMicrophysics0M
end

@testset "Composing bomex with diagnostic_edmf" begin
    sim = CA.Presets.bomex(
        FT;
        t_end = 600,
        model = CA.Presets.diagnostic_edmf(FT),
    )
    @test sim.integrator.p.atmos.turbconv_model isa CA.DiagnosticEDMFX
end
