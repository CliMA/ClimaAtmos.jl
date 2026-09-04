using Test
import Adapt
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaAtmos.RRTMGPInterface as RRTMGPI
import ClimaCore: Fields
using Dates

const FT = Float32

const TEST_GRID = CA.ColumnGrid(FT; z_elem = 10, z_max = 3e3, z_stretch = false)
const TEST_PARAMS = CA.ClimaAtmosParameters(FT)
make_model(; kwargs...) = CA.AtmosModel(TEST_GRID; params = TEST_PARAMS, kwargs...)
bomex_setup() = CA.Setups.Bomex(;
    prognostic_tke = true,
    thermo_params = TEST_PARAMS.thermodynamics_params,
)

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
    @testset "Basic AtmosModel(grid) creates working model with expected defaults" begin
        model = make_model()

        expected_defaults = Dict(
            # Core physics defaults
            :microphysics_model => CA.DryModel,
            :cloud_model => Union{CA.GridScaleCloud, CA.QuadratureCloud},
            :surface => CA.AtmosSurface,
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
        model = make_model(;
            microphysics_model = CA.NonEquilibriumMicrophysics1M(),
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
        @test model.microphysics_model isa CA.NonEquilibriumMicrophysics1M
        @test model.cloud_model isa CA.QuadratureCloud
        @test model.radiation_mode isa RRTMGPI.ClearSkyRadiation
        @test model.hyperdiff isa CA.Hyperdiffusion
        @test model.numerics.hyperdiff isa CA.Hyperdiffusion
        @test model.disable_surface_flux_tendency == true

        # Test that non-overridden defaults are preserved
        @test model.surface.temperature isa CA.SurfaceConditions.AnalyticTemperature
        @test model.insolation isa CA.IdealizedInsolation
        @test model.numerics.diff_mode isa CA.Explicit
    end
end

@testset "Documentation Examples" begin
    # Test basic dry model
    dry_model = make_model(; microphysics_model = CA.DryModel())
    @test dry_model.microphysics_model isa CA.DryModel
    @test dry_model.surface isa CA.AtmosSurface

    # Test moist model with radiation
    moist_model = make_model(;
        microphysics_model = CA.EquilibriumMicrophysics0M(),
        radiation_mode = RRTMGPI.ClearSkyRadiation(;
            idealized_h2o = false,
            add_isothermal_boundary_layer = false,
            aerosol_radiation = false,
            deep_atmosphere = false,
        ),
    )
    @test moist_model.microphysics_model isa CA.EquilibriumMicrophysics0M
    @test moist_model.radiation_mode isa RRTMGPI.ClearSkyRadiation

    # Test HeldSuarezForcing as radiation mode
    held_suarez_model = make_model(; radiation_mode = CA.HeldSuarezForcing())
    @test held_suarez_model.radiation_mode isa CA.HeldSuarezForcing
end

@testset "Interface Compatibility" begin
    # Test that both flat parameters and grouped struct access work
    model = make_model(; microphysics_model = CA.NonEquilibriumMicrophysics1M())

    # Flat parameter access
    @test model.microphysics_model isa CA.NonEquilibriumMicrophysics1M

    # Grouped struct access
    @test model.water isa CA.AtmosWater
    @test model.water.microphysics_model isa CA.NonEquilibriumMicrophysics1M
end

@testset "Complete Grouped Struct Support" begin
    # Test passing complete grouped struct
    water = CA.AtmosWater(; microphysics_model = CA.EquilibriumMicrophysics0M())
    model = make_model(; water = water)
    @test model.water === water
    @test model.microphysics_model isa CA.EquilibriumMicrophysics0M
end

@testset "Binds grid/params/setup and derives components" begin
    setup = bomex_setup()
    model = make_model(; setup)

    @test model.grid === TEST_GRID
    @test model.params === TEST_PARAMS
    @test model.setup === setup

    # Bomex's component hooks are applied at construction (issue 06)
    @test model.subsidence isa CA.LargeScaleSubsidence
    @test model.ls_adv isa CA.LargeScaleAdvection
    @test !isnothing(model.scm_coriolis)
    @test model.surface.flux_scheme isa CA.SurfaceConditions.MoninObukhov
    @test model.surface.temperature isa
          CA.SurfaceConditions.AnalyticTemperature

    # Test Adapt
    stripped = Adapt.adapt(Array, model)
    @test isnothing(stripped.grid)
    @test isnothing(stripped.params)
    @test isnothing(stripped.setup)
    @test stripped.subsidence === model.subsidence
    @test stripped.surface === model.surface
end

@testset "Explicit kwarg wins over setup component (with warning)" begin
    setup = bomex_setup()

    # A leaf kwarg beats the component for that leaf only, and warns
    flux_scheme =
        CA.SurfaceConditions.ExchangeCoefficients{FT}(Cd = 0.001, Ch = 0.001)
    m1 = @test_logs (:warn, r"override values defined by the Bomex setup") make_model(;
        setup,
        flux_scheme,
    )
    @test m1.surface.flux_scheme === flux_scheme
    @test m1.subsidence isa CA.LargeScaleSubsidence  # other components still applied

    # A wholesale group object beats every component in that group (and warns)
    m2 = @test_logs (:warn, r"ls_adv, scm_coriolis, subsidence") make_model(;
        setup,
        scm_setup = CA.SCMSetup(),
    )
    @test isnothing(m2.subsidence)
    @test isnothing(m2.ls_adv)
    @test isnothing(m2.scm_coriolis)
    # ... but other groups still receive their components
    @test m2.surface.flux_scheme isa CA.SurfaceConditions.MoninObukhov

    # No warning when the setup defines no component for the overridden leaf:
    # DecayingProfile is component-free, and the generic surface-temperature
    # fallback does not count as case-defined
    @test_logs make_model(;
        temperature = CA.SurfaceConditions.AnalyticTemperature(
            CA.Setups.zonally_symmetric_temperature,
        ),
    )
end

@testset "Defaults tier: explicit > component > defaults > struct default" begin
    setup = bomex_setup()

    # defaults beat the struct default...
    m = make_model(;
        defaults = (; microphysics_model = CA.EquilibriumMicrophysics0M()),
    )
    @test m.microphysics_model isa CA.EquilibriumMicrophysics0M

    # ...but lose to a setup component (no warning: nothing is suppressed) ...
    preset_flux =
        CA.SurfaceConditions.ExchangeCoefficients{FT}(Cd = 0.002, Ch = 0.002)
    m = @test_logs make_model(; setup, defaults = (; flux_scheme = preset_flux))
    @test m.surface.flux_scheme isa CA.SurfaceConditions.MoninObukhov

    # The generic surface-temperature fallback is a default, not a case
    # component: with a component-free setup, a defaults-tier temperature wins
    m = make_model(;
        defaults = (;
            temperature = CA.SurfaceConditions.SlabOceanTemperature{FT}(),
        ),
    )
    @test m.surface.temperature isa CA.SurfaceConditions.SlabOceanTemperature

    # ...and lose to an explicit kwarg
    m = make_model(;
        defaults = (; microphysics_model = CA.EquilibriumMicrophysics0M()),
        microphysics_model = CA.DryModel(),
    )
    @test m.microphysics_model isa CA.DryModel

    # The defaults tier accepts leaf kwargs only
    @test_throws ErrorException make_model(;
        defaults = (; scm_setup = CA.SCMSetup()),
    )
end

@testset "Copy-with-changes and hash_physics" begin
    model = make_model(; microphysics_model = CA.EquilibriumMicrophysics0M())

    changed = CA.AtmosModel(model; disable_surface_flux_tendency = true)
    @test changed.disable_surface_flux_tendency == true
    # Untouched fields are preserved, not rebuilt from defaults
    @test changed.water === model.water
    @test changed.surface === model.surface
    @test changed.grid === model.grid

    # Leaf properties are rejected: rebuild the group instead
    @test_throws ErrorException CA.AtmosModel(
        model;
        microphysics_model = CA.DryModel(),
    )

    # hash_physics ignores `grid`, `params`, and `setup`: the same physics on a
    # different grid hashes identically
    other_grid = CA.ColumnGrid(FT; z_elem = 10, z_max = 3e3, z_stretch = false)
    other = CA.AtmosModel(
        other_grid;
        params = TEST_PARAMS,
        microphysics_model = CA.EquilibriumMicrophysics0M(),
    )
    @test CA.hash_physics(model) == CA.hash_physics(other)
    @test CA.hash_physics(model) != CA.hash_physics(changed)
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
