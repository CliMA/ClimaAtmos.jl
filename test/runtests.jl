# When Julia 1.10+ is used interactively, stacktraces contain reduced type information to make them shorter.
# On the other hand, the full type information is printed when julia is not run interactively.
# Given that ClimaCore objects are heavily parametrized, non-abbreviated stacktraces are hard to read,
# so we force abbreviated stacktraces even in non-interactive runs.
# (See also Base.type_limited_string_from_context())
redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))
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
        @testset "Aqua" begin @time include("aqua.jl") end
    end

    @testset "Dependencies" begin @time include("dependencies.jl") end
    @testset "Callbacks" begin @time include("callbacks.jl") end
    @testset "Configuration tests" begin @time include("config.jl") end
    @testset "Utilities" begin @time include("utilities.jl") end
    @testset "Variable manipulations" begin @time include("variable_manipulations_tests.jl") end
    @testset "Parameter tests" begin @time include("parameter_tests.jl") end

    # Interface tests
    @testset "Radiation interface tests" begin @time include("rrtmgp_interface.jl") end
    @testset "Coupler compatibility" begin @time include("coupler_compatibility.jl") end
    @testset "Surface albedo tests" begin @time include("surface_albedo.jl") end

    # Solver and restart tests
    @testset "Model getters" begin @time include("solver/model_getters.jl") end
    @testset "AtmosModel Constructor" begin @time include("solver/atmos_model_constructor.jl") end
    @testset "Topography tests" begin @time include("topography.jl") end
    @testset "Restarts" begin @time include("restart.jl") end
    @testset "Reproducibility infra" begin @time include("unit_reproducibility_infra.jl") end
    @testset "Init with file" begin @time include("test_init_with_file.jl") end
end

# ============================================================================
# Dynamics: Prognostic equations and conservation tests
# ============================================================================
if TEST_GROUP in ("dynamics", "all")
    @testset "Prognostic equations" begin @time include("prognostic_equations.jl") end
    @testset "Advection operators" begin @time include("prognostic_equations/advection_tests.jl") end
    @testset "Hyperdiffusion" begin @time include("prognostic_equations/hyperdiffusion_tests.jl") end
    @testset "Tendency computations" begin @time include("prognostic_equations/tendency_tests.jl") end

    # Conservation tests
    @testset "Mass conservation" begin @time include("conservation/mass_conservation.jl") end
    @testset "Energy conservation" begin @time include("conservation/energy_conservation.jl") end
end

# ============================================================================
# Parameterizations: All parameterized tendency tests
# ============================================================================
if TEST_GROUP in ("parameterizations", "all")
    @testset "ERA5 forcing" begin @time include("era5_tests.jl") end

    # Sponge layers
    @testset "Rayleigh sponge tests" begin @time include("parameterized_tendencies/sponge/rayleigh_sponge.jl") end
    @testset "Viscous sponge tests" begin @time include("parameterized_tendencies/sponge/viscous_sponge.jl") end

    # Microphysics
    @testset "Precipitation interface tests" begin @time include("parameterized_tendencies/microphysics/precipitation.jl") end

    # Gravity waves
    @testset "Non-orographic gravity wave (3D)" begin @time include("parameterized_tendencies/gravity_wave/non_orographic_gravity_wave/nogw_test_3d.jl") end
    @testset "Non-orographic gravity wave (MiMA)" begin @time include("parameterized_tendencies/gravity_wave/non_orographic_gravity_wave/nogw_test_mima.jl") end
    @testset "Non-orographic gravity wave (single column)" begin @time include("parameterized_tendencies/gravity_wave/non_orographic_gravity_wave/nogw_test_single_column.jl") end
end

#! format: on

nothing
