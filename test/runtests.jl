# When Julia 1.10+ is used interactively, stacktraces contain reduced type information to make them shorter.
# On the other hand, the full type information is printed when julia is not run interactively.
# Given that ClimaCore objects are heavily parametrized, non-abbreviated stacktraces are hard to read,
# so we force abbreviated stacktraces even in non-interactive runs.
# (See also Base.type_limited_string_from_context())
redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))
using SafeTestsets
using Test

# Download test artifacts
include("download_artifacts.jl")

# Get test group from environment variable (default: run all tests)
TEST_GROUP = get(ENV, "TEST_GROUP", "all")

#! format: off

# ============================================================================
# Infrastructure: Configuration, utilities, interfaces, and integration tests
# ============================================================================
if TEST_GROUP in ("infrastructure", "all")
    # Skip Aqua tests due to precompilation failures in old versions of SciMLBase
    import SciMLBase
    if pkgversion(SciMLBase) > v"2.12.1"
        @safetestset "Aqua" begin @time include("aqua.jl") end
    end

    @safetestset "Dependencies" begin @time include("dependencies.jl") end
    @safetestset "Callbacks" begin @time include("callbacks.jl") end
    @safetestset "Configuration tests" begin @time include("config.jl") end
    @safetestset "Grids" begin @time include("grids.jl") end
    @safetestset "Utilities" begin @time include("utilities.jl") end
    @safetestset "Variable manipulations" begin @time include("variable_manipulations_tests.jl") end
    @safetestset "Parameter tests" begin @time include("parameter_tests.jl") end

    @safetestset "Check TOML path" begin @time include("test_output_yaml_path.jl") end

    # Interface tests
    @safetestset "Radiation interface tests" begin @time include("rrtmgp_interface.jl") end
    @safetestset "Coupler compatibility" begin @time include("coupler_compatibility.jl") end
    @safetestset "Surface albedo tests" begin @time include("surface_albedo.jl") end

    # Solver tests
    @safetestset "Model getters" begin @time include("solver/model_getters.jl") end
    @safetestset "AtmosModel Constructor" begin @time include("solver/atmos_model_constructor.jl") end
    @safetestset "Topography tests" begin @time include("topography.jl") end
end

# ============================================================================
# Dynamics: Prognostic equations and conservation tests
# ============================================================================
if TEST_GROUP in ("dynamics", "all")
    @safetestset "Prognostic equations" begin @time include("prognostic_equations.jl") end
    @safetestset "Advection operators" begin @time include("prognostic_equations/advection_tests.jl") end
    @safetestset "Hyperdiffusion" begin @time include("prognostic_equations/hyperdiffusion_tests.jl") end
    @safetestset "Tendency computations" begin @time include("prognostic_equations/tendency_tests.jl") end
    @safetestset "Vertical water borrowing limiter" begin @time include("prognostic_equations/vertical_water_borrowing_tests.jl") end

    # Conservation tests
    @safetestset "Mass conservation" begin @time include("conservation/mass_conservation.jl") end
    @safetestset "Energy conservation" begin @time include("conservation/energy_conservation.jl") end
end

# ============================================================================
# Parameterizations: Parameterized tendency tests (excluding ERA5)
# ============================================================================
if TEST_GROUP in ("parameterizations", "all")
    # Sponge layers (combined for shared space setup)
    @safetestset "Sponge layers" begin @time include("parameterized_tendencies/sponge.jl") end

    # Microphysics tests
    @safetestset "Microphysics tendency tests" begin @time include("parameterized_tendencies/microphysics/tendency.jl") end
    @safetestset "Microphysics wrappers tests" begin @time include("parameterized_tendencies/microphysics/microphysics_wrappers.jl") end
    @safetestset "SGS quadrature tests" begin @time include("parameterized_tendencies/microphysics/sgs_quadrature.jl") end
    @safetestset "Tendency limiters tests" begin @time include("parameterized_tendencies/microphysics/tendency_limiters.jl") end
    @safetestset "Moisture fixers tests" begin @time include("parameterized_tendencies/microphysics/moisture_fixers.jl") end
    @safetestset "Cloud fraction tests" begin @time include("parameterized_tendencies/microphysics/cloud_fraction.jl") end
    @safetestset "SGS saturation tests" begin @time include("parameterized_tendencies/microphysics/sgs_saturation.jl") end
    @safetestset "BMT integration tests" begin @time include("parameterized_tendencies/microphysics/bmt_integration.jl") end
    @safetestset "Allocation tests" begin @time include("parameterized_tendencies/microphysics/allocations.jl") end

    # NOTE: Gravity wave visualization scripts (nogw_test_3d.jl, nogw_test_mima.jl,
    # nogw_test_single_column.jl, ogwd_3d.jl, ogwd_baseflux.jl) are not included
    # in the test suite because they have no @test assertions - they only generate
    # comparison plots for visual verification.
end

# ============================================================================
# Restarts: Restart and reproducibility tests
# ============================================================================
if TEST_GROUP in ("restarts", "all")
    @safetestset "Restarts" begin @time include("restart.jl") end
    @safetestset "Reproducibility infra" begin @time include("unit_reproducibility_infra.jl") end
    @safetestset "Init with file" begin @time include("test_init_with_file.jl") end
end

# ============================================================================
# ERA5: External forcing data tests (heavy)
# ============================================================================
if TEST_GROUP in ("era5", "all")
    @safetestset "ERA5 forcing" begin @time include("era5_tests.jl") end
end

#! format: on

nothing
