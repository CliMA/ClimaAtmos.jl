# # Running the tests
#
# Tests run in parallel across worker processes via ParallelTestRunner.jl. Each
# test file gets a fresh worker module, so files must be self-contained.
#
# Always go through `Pkg.test`: it resolves the test-only dependencies (Aqua,
# CairoMakie, ...) that `julia --project=. test/runtests.jl` would not see.
#
# Everything:
#
#     julia --project=. -e 'using Pkg; Pkg.test("ClimaAtmos")'
#
# `] test` from the REPL works too, but its only option is `--coverage`, so use
# `Pkg.test` when you want any of the filters or flags below.
#
# One CI group (see `TEST_GROUPS` below for the names). `TEST_GROUP` is read from
# the environment, so `ENV["TEST_GROUP"] = "dynamics"` also works from the REPL:
#
#     TEST_GROUP=dynamics julia --project=. -e 'using Pkg; Pkg.test("ClimaAtmos")'
#
# A subset, by test name. Names are paths relative to this directory without the
# `.jl` extension, matched with `startswith`, and `!` excludes:
#
#     Pkg.test("ClimaAtmos"; test_args = ["prognostic_equations"])
#     Pkg.test("ClimaAtmos"; test_args = ["conservation", "grids"])
#     Pkg.test("ClimaAtmos"; test_args = ["!era5"])
#
# Useful flags, passed the same way:
#
#     --list        print the selected test names and exit
#     --jobs=N      number of workers (default: from CPU count and free memory)
#     --quickfail   stop the whole run at the first failure
#     --verbose     per-test start times
#
# Tips:
#
#   - `--jobs=1` when a failure looks like a parallelism artifact, or when the
#     interleaved output of several workers is hard to read.
#   - Output from a test file is only printed if that file fails; use `--list`
#     plus a name filter to iterate on one file at a time.
#   - Workers are recycled when they exceed a memory threshold, but only between
#     files, so a single hungry file can still OOM the machine. The known ones
#     are listed in `SERIAL` below; add to it if a new test starts crashing runs.
#   - `JULIA_TEST_MAXRSS_MB=2000` recycles workers sooner, which helps when RSS
#     creeps up across many files (it cannot bound one oversized file).

using ClimaAtmos
using ParallelTestRunner

# Download test artifacts on the driver, before any worker is spawned.
include("download_artifacts.jl")

# Test suite, grouped so that CI can run one group per job via TEST_GROUP.
# Only files listed here run: the rest of `test/` holds shared helpers and
# plot-only scripts that have no `@test` assertions.
#! format: off
const TEST_GROUPS = Dict(
    # ========================================================================
    # Infrastructure: Configuration, utilities, interfaces, and integration tests
    # ========================================================================
    "infrastructure" => [
        "aqua",
        "dependencies",
        "callbacks",
        "config",
        "grids",
        "utilities",
        "variable_manipulations_tests",
        "tracer_processes_tests",
        "parameter_tests",
        "test_output_yaml_path",

        # Interface tests
        "rrtmgp_interface",
        "coupler_compatibility",
        "surface_albedo",
        "larcform1",

        # Config tests
        "config/model_from_config",
        "config/atmos_model_constructor",
        "presets",
        "topography",
    ],

    # ========================================================================
    # Diagnostics: Unit tests for diagnostic variables
    # ========================================================================
    "diagnostics" => [
        "diagnostics/unit_diagnostics",
        "diagnostics/diagnostics_config",
        "cosp/subcol_test",
    ],

    # ========================================================================
    # Dynamics: Prognostic equations and conservation tests
    # ========================================================================
    "dynamics" => [
        "prognostic_equations",
        "prognostic_equations/advection_tests",
        "prognostic_equations/hyperdiffusion_tests",
        "prognostic_equations/tendency_tests",
        "prognostic_equations/tracer_mass_consistency_tests",
        "prognostic_equations/correct_implicit_advection_tests",
        "prognostic_equations/vertical_diffusion_tests",
        "prognostic_equations/vertical_water_borrowing_tests",
        "prognostic_equations/enforce_physical_constraints_tests",
        "prognostic_equations/eddy_diffusion_closures_tests",

        # Conservation tests
        "conservation/mass_conservation",
        "conservation/energy_conservation",
    ],

    # ========================================================================
    # Parameterizations: Parameterized tendency tests (excluding ERA5)
    # ========================================================================
    "parameterizations" => [
        # Sponge layers (combined for shared space setup)
        "parameterized_tendencies/sponge",

        # Microphysics tests
        "parameterized_tendencies/microphysics/tendency",
        "parameterized_tendencies/microphysics/microphysics_wrappers",
        "parameterized_tendencies/microphysics/sgs_quadrature",
        "parameterized_tendencies/microphysics/sgs_moments",
        "parameterized_tendencies/microphysics/tendency_limiters",
        "parameterized_tendencies/microphysics/moisture_fixers",
        "parameterized_tendencies/microphysics/cloud_fraction",
        "parameterized_tendencies/microphysics/sgs_saturation",
        "parameterized_tendencies/microphysics/bmt_integration",
        "parameterized_tendencies/microphysics/allocations",

        # Chemistry tests
        "parameterized_tendencies/chemistry/chemistry_tendency",

        # Gravity wave: Beres convective NOGW pure-function unit tests (no
        # simulation build). The simulation-based Beres tests
        # (test_beres_single_column.jl, test_beres_sphere_integration.jl) run as
        # standalone Buildkite steps.
        "parameterized_tendencies/gravity_wave/non_orographic_gravity_wave/test_beres_unit",
    ],

    # ========================================================================
    # Restarts: Restart and reproducibility tests
    # ========================================================================
    "restarts" => [
        "restart",
        "unit_reproducibility_infra",
        "test_init_with_file",
    ],

    # ========================================================================
    # ERA5: External forcing data tests (heavy)
    # ========================================================================
    "era5" => [
        "era5_tests",
        "column_datasets_tests",
    ],
)
#! format: on

# Files that peak above the ~3.8 GB per-worker RSS ceiling on their own. Worker
# recycling only happens between files, so it cannot bound a single hungry one;
# and because tests are started longest-first, these would otherwise all launch
# in the same wave. They run one at a time before the parallel batch instead.
const SERIAL = [
    "restart",                                                # peaks ~6.4 GB
    "prognostic_equations/tracer_mass_consistency_tests",      # peaks ~6.4 GB
    "prognostic_equations/enforce_physical_constraints_tests", # peaks ~3.5 GB
]

const TEST_GROUP = get(ENV, "TEST_GROUP", "all")
if TEST_GROUP != "all" && !haskey(TEST_GROUPS, TEST_GROUP)
    groups = join(sort!(collect(keys(TEST_GROUPS))), ", ")
    error("Unknown TEST_GROUP `$TEST_GROUP`; expected `all` or one of: $groups")
end

selected =
    TEST_GROUP == "all" ? reduce(vcat, values(TEST_GROUPS)) : TEST_GROUPS[TEST_GROUP]

testsuite = find_tests(@__DIR__)
stale = setdiff([selected; SERIAL], keys(testsuite))
isempty(stale) || error("No such test file(s): $(join(sort!(stale), ", "))")
filter!(((name, _),) -> name in selected, testsuite)

# ClimaCore objects are heavily parametrized, so non-abbreviated stacktraces are
# hard to read. Julia only abbreviates them when running interactively, so force
# it here as well. (See also Base.type_limited_string_from_context())
init_worker_code = quote
    redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))
end

runtests(ClimaAtmos, ARGS; testsuite, init_worker_code, serial = SERIAL)
