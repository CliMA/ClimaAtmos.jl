using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaAtmos.RRTMGPInterface as RRTMGPI
using Dates

const FT = Float32

# Helper function to test defaults against expected types
function test_defaults(model, expected_defaults)
    for (field, expected_type) in expected_defaults
        actual_value = getproperty(model, field)
        if expected_type === nothing
            @test actual_value === nothing || actual_value == nothing
        elseif expected_type isa Type
            @test actual_value isa expected_type
        else
            @test actual_value == expected_type
        end
    end
end

@testset "AtmosModel Constructor Tests" begin

    @testset "Sensible Defaults" begin
        @testset "Basic AtmosModel() creates working model with expected defaults" begin
            model = CA.AtmosModel()

            # Define expected defaults in a compact dictionary
            expected_defaults = Dict(
                # Core physics defaults
                :moisture_model => CA.DryModel,
                :precip_model => CA.NoPrecipitation,
                :cloud_model => CA.GridScaleCloud,
                :surface_model => CA.PrescribedSurfaceTemperature,
                :sfc_temperature => CA.ZonallySymmetricSST,
                :insolation => CA.IdealizedInsolation,
                :disable_surface_flux_tendency => false,

                # Advanced physics defaults (should be nothing/disabled)
                :radiation_mode => nothing,
                :ozone => nothing,
                :co2 => nothing,
                :forcing_type => nothing,
                :turbconv_model => nothing,
                :non_orographic_gravity_wave => nothing,
                :orographic_gravity_wave => nothing,
                :viscous_sponge => nothing,
                :rayleigh_sponge => nothing,
                :hyperdiff => nothing,
                :vert_diff => nothing,
            )

            test_defaults(model, expected_defaults)

            # Test numerics structure separately due to nested fields
            @test model.numerics isa CA.AtmosNumerics
            @test model.numerics.diff_mode isa CA.Explicit
        end

        @testset "User overrides work correctly" begin
            # Test various override scenarios including complex parameter types
            model = CA.AtmosModel(;
                moisture_model = CA.EquilMoistModel(),
                precip_model = CA.Microphysics1Moment(),
                cloud_model = CA.QuadratureCloud(CA.SGSQuadrature(FT)),
                radiation_mode = RRTMGPI.ClearSkyRadiation(
                    false,
                    false,
                    false,
                    false,
                ),
                ozone = CA.IdealizedOzone(),
                co2 = CA.FixedCO2(),
                forcing_type = CA.HeldSuarezForcing(),
                hyperdiff = CA.ClimaHyperdiffusion(;
                    ν₄_vorticity_coeff = 1e15,
                    ν₄_scalar_coeff = 1e15,
                    divergence_damping_factor = 1.0,
                ),
                disable_surface_flux_tendency = true,
            )

            # Test customized values
            @test model.moisture_model isa CA.EquilMoistModel
            @test model.precip_model isa CA.Microphysics1Moment
            @test model.cloud_model isa CA.QuadratureCloud
            @test model.radiation_mode isa RRTMGPI.ClearSkyRadiation
            @test model.ozone isa CA.IdealizedOzone
            @test model.co2 isa CA.FixedCO2
            @test model.forcing_type isa CA.HeldSuarezForcing
            @test model.hyperdiff isa CA.ClimaHyperdiffusion
            @test model.disable_surface_flux_tendency == true

            # Test that non-overridden defaults are preserved
            @test model.surface_model isa CA.PrescribedSurfaceTemperature
            @test model.insolation isa CA.IdealizedInsolation
            @test model.numerics.diff_mode isa CA.Explicit
        end

        @testset "Convenience constructors work with defaults" begin
            # Test that convenience constructors properly override defaults
            models = [
                ("dry", CA.DryAtmosModel(), CA.DryModel),
                ("equil", CA.EquilMoistAtmosModel(), CA.EquilMoistModel),
                (
                    "nonequil",
                    CA.NonEquilMoistAtmosModel(),
                    CA.NonEquilMoistModel,
                ),
            ]

            for (name, model, expected_moisture_type) in models
                @test model.moisture_model isa expected_moisture_type
                @test model.surface_model isa CA.PrescribedSurfaceTemperature  # default preserved
                @test model.numerics.diff_mode isa CA.Explicit  # default preserved
            end
        end
    end

    @testset "Documentation Examples" begin
        @testset "Basic configurations work as documented" begin
            # Test basic dry model
            dry_model = CA.AtmosModel(;
                moisture_model = CA.DryModel(),
                surface_model = CA.PrescribedSurfaceTemperature(),
            )
            @test dry_model.moisture_model isa CA.DryModel
            @test dry_model.surface_model isa CA.PrescribedSurfaceTemperature

            # Test moist model with radiation
            moist_model = CA.AtmosModel(;
                moisture_model = CA.EquilMoistModel(),
                precip_model = CA.Microphysics0Moment(),
                radiation_mode = RRTMGPI.ClearSkyRadiation(
                    false,
                    false,
                    false,
                    false,
                ),
                ozone = CA.IdealizedOzone(),
                co2 = CA.FixedCO2(),
            )
            @test moist_model.moisture_model isa CA.EquilMoistModel
            @test moist_model.radiation_mode isa RRTMGPI.ClearSkyRadiation
            @test moist_model.ozone isa CA.IdealizedOzone
        end
    end

    @testset "Interface Compatibility" begin
        # Test that both flat parameters and grouped struct access work
        model = CA.AtmosModel(moisture_model = CA.NonEquilMoistModel())

        # Flat parameter access
        @test model.moisture_model isa CA.NonEquilMoistModel

        # Grouped struct access  
        @test model.hydrology isa CA.AtmosHydrology
        @test model.hydrology.moisture_model isa CA.NonEquilMoistModel
    end

    @testset "Error Handling" begin
        # Test invalid parameter error with helpful message
        @test_throws ErrorException CA.AtmosModel(; invalid_param = "test")
    end

    @testset "Internal Consistency" begin
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
    end
end
