using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaAtmos.RRTMGPInterface as RRTMGPI
using Dates

const FT = Float32

@testset "AtmosModel Constructor Tests" begin

    @testset "Intelligent Defaults" begin
        @testset "Basic AtmosModel() with defaults" begin
            # Test that AtmosModel() with no arguments creates a working model with sensible defaults
            model = CA.AtmosModel()

            # Test hydrology defaults
            @test model.moisture_model isa CA.DryModel
            @test model.precip_model isa CA.NoPrecipitation
            @test model.cloud_model isa CA.GridScaleCloud

            # Test surface defaults
            @test model.surface_model isa CA.PrescribedSurfaceTemperature
            @test model.sfc_temperature isa CA.ZonallySymmetricSST

            # Test radiation defaults
            @test model.insolation isa CA.IdealizedInsolation
            @test model.radiation_mode === nothing  # No radiation by default
            @test model.ozone === nothing
            @test model.co2 === nothing

            # Test numerics defaults
            @test model.numerics isa CA.AtmosNumerics
            @test model.numerics.energy_upwinding == Val(:first_order)
            @test model.numerics.tracer_upwinding == Val(:first_order)
            @test model.numerics.edmfx_upwinding == Val(:first_order)
            @test model.numerics.edmfx_sgsflux_upwinding == Val(:none)
            @test model.numerics.test_dycore_consistency === nothing
            @test model.numerics.limiter === nothing
            @test model.numerics.diff_mode isa CA.Explicit

            # Test top-level defaults
            @test model.disable_surface_flux_tendency == false
            @test model.hyperdiff === nothing
            @test model.vert_diff === nothing

            # Test advanced physics defaults (should be nothing/disabled)
            @test model.forcing_type === nothing
            @test model.turbconv_model === nothing
            @test model.non_orographic_gravity_wave === nothing
            @test model.orographic_gravity_wave === nothing
            @test model.viscous_sponge === nothing
            @test model.rayleigh_sponge === nothing
        end

        @testset "User overrides defaults" begin
            # Test that user-provided arguments override the defaults
            model = CA.AtmosModel(;
                moisture_model = CA.EquilMoistModel(),
                precip_model = CA.Microphysics1Moment(),
                radiation_mode = RRTMGPI.ClearSkyRadiation(
                    false,
                    false,
                    false,
                    false,
                ),
                disable_surface_flux_tendency = true,
            )

            # Test overridden values
            @test model.moisture_model isa CA.EquilMoistModel
            @test model.precip_model isa CA.Microphysics1Moment
            @test model.radiation_mode isa RRTMGPI.ClearSkyRadiation
            @test model.disable_surface_flux_tendency == true

            # Test that non-overridden defaults are preserved
            @test model.cloud_model isa CA.GridScaleCloud
            @test model.surface_model isa CA.PrescribedSurfaceTemperature
            @test model.insolation isa CA.IdealizedInsolation
            @test model.numerics.diff_mode isa CA.Explicit
        end

        @testset "Convenience constructors vs main constructor" begin
            # Test that convenience constructors work and override defaults appropriately
            dry_model = CA.DryAtmosModel()
            @test dry_model.moisture_model isa CA.DryModel
            @test dry_model.precip_model isa CA.NoPrecipitation
            @test dry_model.surface_model isa CA.PrescribedSurfaceTemperature
            @test dry_model.insolation isa CA.IdealizedInsolation

            equil_model = CA.EquilMoistAtmosModel()
            @test equil_model.moisture_model isa CA.EquilMoistModel  # overridden by convenience constructor
            @test equil_model.precip_model isa CA.Microphysics0Moment  # overridden by convenience constructor
            @test equil_model.ozone isa CA.IdealizedOzone  # added by convenience constructor
            @test equil_model.co2 isa CA.FixedCO2  # added by convenience constructor
            @test equil_model.numerics.diff_mode isa CA.Explicit  # main constructor default preserved

            nonequil_model = CA.NonEquilMoistAtmosModel()
            @test nonequil_model.moisture_model isa CA.NonEquilMoistModel  # overridden
            @test nonequil_model.precip_model isa CA.Microphysics1Moment  # overridden
            @test nonequil_model.noneq_cloud_formation_mode isa CA.Explicit  # added
            @test nonequil_model.numerics.diff_mode isa CA.Explicit  # main constructor default preserved
        end

        @testset "Partial customization" begin
            # Test that partial customization works well with defaults
            model = CA.AtmosModel(;
                forcing_type = CA.HeldSuarezForcing(),
                hyperdiff = CA.ClimaHyperdiffusion(;
                    ν₄_vorticity_coeff = 1e15,
                    ν₄_scalar_coeff = 1e15,
                    divergence_damping_factor = 1.0,
                ),
            )

            # Test that only specified parameters are customized
            @test model.forcing_type isa CA.HeldSuarezForcing  # customized
            @test model.hyperdiff isa CA.ClimaHyperdiffusion  # customized

            # Test that all other defaults are preserved
            @test model.moisture_model isa CA.DryModel  # default
            @test model.precip_model isa CA.NoPrecipitation  # default
            @test model.surface_model isa CA.PrescribedSurfaceTemperature  # default
            @test model.numerics.diff_mode isa CA.Explicit  # default
            @test model.disable_surface_flux_tendency == false  # default
        end
    end

    @testset "Documentation Examples" begin

        @testset "Basic dry model" begin
            model = CA.AtmosModel(;
                moisture_model = CA.DryModel(),
                surface_model = CA.PrescribedSurfaceTemperature(),
                precip_model = CA.NoPrecipitation(),
            )

            @test model.moisture_model isa CA.DryModel
            @test model.surface_model isa CA.PrescribedSurfaceTemperature
            @test model.precip_model isa CA.NoPrecipitation
            @test model.disable_surface_flux_tendency == false
        end

        @testset "Moist model with radiation" begin
            model = CA.AtmosModel(;
                moisture_model = CA.EquilMoistModel(),
                precip_model = CA.Microphysics0Moment(),
                radiation_mode = RRTMGPI.AllSkyRadiation(
                    false,
                    false,
                    CA.InteractiveCloudInRadiation(),
                    false,
                    false,
                    false,
                    false,
                ),
                ozone = CA.IdealizedOzone(),
                co2 = CA.FixedCO2(),
            )

            @test model.moisture_model isa CA.EquilMoistModel
            @test model.precip_model isa CA.Microphysics0Moment
            @test model.radiation_mode isa RRTMGPI.AllSkyRadiation
            @test model.ozone isa CA.IdealizedOzone
            @test model.co2 isa CA.FixedCO2
        end

        @testset "Model with hyperdiffusion and sponge" begin
            model = CA.AtmosModel(;
                moisture_model = CA.NonEquilMoistModel(),
                precip_model = CA.Microphysics1Moment(),
                hyperdiff = CA.ClimaHyperdiffusion(;
                    ν₄_vorticity_coeff = 1e15,
                    ν₄_scalar_coeff = 1e15,
                    divergence_damping_factor = 1.0,
                ),
                rayleigh_sponge = CA.RayleighSponge(;
                    zd = 12000.0,
                    α_uₕ = 4.0,
                    α_w = 2.0,
                ),
                disable_surface_flux_tendency = false,
            )

            @test model.moisture_model isa CA.NonEquilMoistModel
            @test model.precip_model isa CA.Microphysics1Moment
            @test model.hyperdiff isa CA.ClimaHyperdiffusion
            @test model.rayleigh_sponge isa CA.RayleighSponge
            @test model.disable_surface_flux_tendency == false
        end

        @testset "Additional parameter types" begin
            # Test parameter types not covered in main examples
            model = CA.AtmosModel(;
                cloud_model = CA.QuadratureCloud(CA.SGSQuadrature(FT)),
                forcing_type = CA.HeldSuarezForcing(),
                radiation_mode = RRTMGPI.ClearSkyRadiation(
                    false,
                    false,
                    false,
                    false,
                ),
            )

            @test model.cloud_model isa CA.QuadratureCloud
            @test model.forcing_type isa CA.HeldSuarezForcing
            @test model.radiation_mode isa RRTMGPI.ClearSkyRadiation
        end
    end

    @testset "Error Handling" begin
        try
            CA.AtmosModel(; invalid_param = "test")
        catch e
            @test e isa ErrorException
            @test occursin("Unknown AtmosModel argument: invalid_param", e.msg)
            @test occursin("Available arguments:", e.msg)
            @test occursin("moisture_model", e.msg)
            @test occursin("surface_model", e.msg)
        end
    end

    @testset "Interface Compatibility" begin
        # Test that both flat parameters and grouped struct access work
        model = CA.AtmosModel(;
            moisture_model = CA.NonEquilMoistModel(),
            forcing_type = CA.HeldSuarezForcing(),
        )

        # Flat parameter access
        @test model.moisture_model isa CA.NonEquilMoistModel
        @test model.forcing_type isa CA.HeldSuarezForcing

        # Grouped struct access  
        @test model.hydrology isa CA.AtmosHydrology
        @test model.forcing isa CA.AtmosForcing
        @test model.hydrology.moisture_model isa CA.NonEquilMoistModel
        @test model.forcing.forcing_type isa CA.HeldSuarezForcing
    end

    @testset "Utility Behavior" begin
        model = CA.AtmosModel(; moisture_model = CA.DryModel())

        # Test defaults for unspecified parameters
        @test model.moisture_model isa CA.DryModel
        @test model.precip_model isa CA.NoPrecipitation
        @test model.hyperdiff === nothing
        @test model.disable_surface_flux_tendency == false

        # Test broadcasting compatibility
        @test Base.broadcastable(model) == tuple(model)
        @test Base.broadcastable(model.hydrology) == tuple(model.hydrology)
        @test Base.broadcastable(model.moisture_model) ==
              tuple(model.moisture_model)
    end

    @testset "AtmosModel Argument Uniqueness" begin
        # Ensure no conflicts between grouped arguments and direct AtmosModel fields
        grouped_args = Set(keys(CA.GROUPED_PROPERTY_MAP))
        grouped_struct_fields = Set([
            :hydrology,
            :forcing,
            :radiation,
            :advection,
            :turbconv,
            :gravity_wave,
            :vert_diff,
            :sponge,
            :surface,
        ])
        direct_args = Set(
            filter(fn -> fn ∉ grouped_struct_fields, fieldnames(CA.AtmosModel)),
        )

        # Check for keyword argument conflicts
        overlap = intersect(grouped_args, direct_args)
        @test isempty(overlap)

        full_overlap = intersect(grouped_args, Set(fieldnames(CA.AtmosModel)))
        if !isempty(full_overlap)
            @error "Found conflicts between grouped arguments and grouped struct fieldnames:"
            @info """Conflicting names: $(collect(full_overlap))
            AtmosModel fieldnames: $(collect(fieldnames(CA.AtmosModel)))
            Grouped args (first 20): $(collect(grouped_args)[1:min(20, length(grouped_args))])
            Grouped struct fields: $(collect(grouped_struct_fields))"""
        end
        @test isempty(full_overlap)
    end
end
