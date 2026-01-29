using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaAtmos.RRTMGPInterface as RRTMGPI
using Dates

const FT = Float32
"""
    test_defaults(model, expected_defaults)

Test that each field in `model` matches the expected type or value specified in `expected_defaults`.
For each field:
- If expected type is `nothing`, test that field is `nothing`
- If expected type is a Type, test that field is an instance of that type
- Otherwise, test that field equals the expected value exactly
"""
function test_defaults(model, expected_defaults)
    for (field, expected_type) in expected_defaults
        actual_value = getproperty(model, field)
        if isnothing(expected_type)
            @test isnothing(actual_value)
        elseif expected_type isa Type
            @test actual_value isa expected_type
        else
            @test actual_value == expected_type
        end
    end
end

@testset "Sensible Defaults" begin
    @testset "Basic AtmosModel() creates working model with expected defaults" begin
        model = CA.AtmosModel()

        # Define expected defaults in a compact dictionary
        expected_defaults = Dict(
            # Core physics defaults
            :moisture_model => CA.DryModel,
            :microphysics_model => CA.NoPrecipitation,
            :cloud_model => Union{CA.GridScaleCloud, CA.QuadratureCloud},
            :surface_model => CA.PrescribedSST,
            :sfc_temperature => CA.ZonallySymmetricSST,
            :insolation => CA.IdealizedInsolation,
            :disable_surface_flux_tendency => false,

            # Advanced physics defaults (should be nothing/disabled)
            :radiation_mode => nothing,
            :turbconv_model => nothing,
            :non_orographic_gravity_wave => nothing,
            :orographic_gravity_wave => nothing,
            :viscous_sponge => nothing,
            :rayleigh_sponge => nothing,
            :hyperdiff => Union{Nothing, CA.Hyperdiffusion},
            :vertical_diffusion => nothing,
        )

        test_defaults(model, expected_defaults)

        # Test numerics structure separately due to nested fields
        @test model.numerics isa CA.AtmosNumerics
        @test model.numerics.diff_mode isa CA.Explicit
        @test model.numerics.hyperdiff isa Union{Nothing, CA.Hyperdiffusion}
    end

    @testset "User overrides work correctly" begin
        # Test various override scenarios including complex parameter types
        model = CA.AtmosModel(;
            moisture_model = CA.EquilMoistModel(),
            microphysics_model = CA.Microphysics1Moment(),
            cloud_model = CA.QuadratureCloud(),
            radiation_mode = RRTMGPI.ClearSkyRadiation(;
                idealized_h2o = false,
                add_isothermal_boundary_layer = false,
                aerosol_radiation = false,
                deep_atmosphere = false,
            ),
            hyperdiff = CA.Hyperdiffusion(;
                ν₄_vorticity_coeff = 1e15,
                divergence_damping_factor = 1.0,
                prandtl_number = 1.0,
            ),
            disable_surface_flux_tendency = true,
        )

        # Test customized values
        @test model.moisture_model isa CA.EquilMoistModel
        @test model.microphysics_model isa CA.Microphysics1Moment
        @test model.cloud_model isa CA.QuadratureCloud
        @test model.radiation_mode isa RRTMGPI.ClearSkyRadiation
        @test model.hyperdiff isa CA.Hyperdiffusion
        @test model.numerics.hyperdiff isa CA.Hyperdiffusion
        @test model.disable_surface_flux_tendency == true

        # Test that non-overridden defaults are preserved
        @test model.surface_model isa CA.PrescribedSST
        @test model.insolation isa CA.IdealizedInsolation
        @test model.numerics.diff_mode isa CA.Explicit
    end

    @testset "Convenience constructors work with defaults" begin
        # Test that convenience constructors properly override defaults
        models = [
            ("dry", CA.DryAtmosModel(), CA.DryModel),
            ("equil", CA.EquilMoistAtmosModel(), CA.EquilMoistModel),
            ("nonequil", CA.NonEquilMoistAtmosModel(), CA.NonEquilMoistModel),
        ]

        for (name, model, expected_moisture_type) in models
            @test model.moisture_model isa expected_moisture_type
            @test model.surface_model isa CA.PrescribedSST  # default preserved
            @test model.numerics.diff_mode isa CA.Explicit  # default preserved
            @test model.numerics.hyperdiff isa Union{Nothing, CA.Hyperdiffusion}  # default is nothing or Hyperdiffusion
        end
    end
end

@testset "Documentation Examples" begin
    @testset "Basic configurations work as documented" begin
        # Test basic dry model
        dry_model = CA.AtmosModel(;
            moisture_model = CA.DryModel(),
            surface_model = CA.PrescribedSST(),
        )
        @test dry_model.moisture_model isa CA.DryModel
        @test dry_model.surface_model isa CA.PrescribedSST

        # Test moist model with radiation
        moist_model = CA.AtmosModel(;
            moisture_model = CA.EquilMoistModel(),
            microphysics_model = CA.Microphysics0Moment(),
            radiation_mode = RRTMGPI.ClearSkyRadiation(;
                idealized_h2o = false,
                add_isothermal_boundary_layer = false,
                aerosol_radiation = false,
                deep_atmosphere = false,
            ),
        )
        @test moist_model.moisture_model isa CA.EquilMoistModel
        @test moist_model.microphysics_model isa CA.Microphysics0Moment
        @test moist_model.radiation_mode isa RRTMGPI.ClearSkyRadiation

        # Test HeldSuarezForcing as radiation mode
        held_suarez_model =
            CA.AtmosModel(; radiation_mode = CA.HeldSuarezForcing())
        @test held_suarez_model.radiation_mode isa CA.HeldSuarezForcing
    end
end

@testset "Interface Compatibility" begin
    # Test that both flat parameters and grouped struct access work
    model = CA.AtmosModel(moisture_model = CA.NonEquilMoistModel())

    # Flat parameter access
    @test model.moisture_model isa CA.NonEquilMoistModel

    # Grouped struct access  
    @test model.water isa CA.AtmosWater
    @test model.water.moisture_model isa CA.NonEquilMoistModel
end

@testset "Complete Grouped Struct Support" begin
    # Test passing complete grouped struct
    water = CA.AtmosWater(; moisture_model = CA.EquilMoistModel())
    model = CA.AtmosModel(; water = water)
    @test model.water === water
    @test model.moisture_model isa CA.EquilMoistModel
end

@testset "Error Handling" begin
    # Test invalid parameter error with helpful message
    @test_throws ErrorException CA.AtmosModel(; w = 2, invalid_param = "test")
end

@testset "Internal Consistency" begin
    # Ensure no conflicts between grouped arguments and direct AtmosModel fields
    grouped_args = Set(keys(CA.GROUPED_PROPERTY_MAP))
    grouped_struct_fields = Set([
        :water,
        :forcing,
        :radiation,
        :advection,
        :turbconv,
        :gravity_wave,
        :vert_diff,
        :sponge,
        :surface,
        :numerics,
    ])
    direct_args =
        Set(filter(fn -> fn ∉ grouped_struct_fields, fieldnames(CA.AtmosModel)))

    # Check for keyword argument conflicts
    overlap = intersect(grouped_args, direct_args)
    @test isempty(overlap)
end
