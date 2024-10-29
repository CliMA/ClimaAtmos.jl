import Pkg
import Test

# Adding new dependencies to a Julia project has downsides. First, every
# dependency increases the precompilation and import time of a package. For
# complex packages, such as CUDA, this translates in several minutes of
# precompilation time that every single user of ClimaAtmos (including our CI
# runners) always pays. Second, dependencies can increase the cost of
# maintaining ClimaAtmos and the surface area of things that can go wrong (a
# buggy update in a dependency can break ClimaAtmos). Third, dependencies
# increase the complexity of the project's environment, potentially causing
# compatibility issues and restricting the versions we can use.

# This test follows the spirit of the reproducibility tests and tries to capture the
# cost of adding a new dependency by having developers explicitly come here and
# declare their intents.
#
# DO NOT ADD new dependencies if:

# - the feature you need can be easily implemented (e.g., do you need the might
#   of Distributions.jl to compute a guassian function?)
# - your dependency is heavy/has lots of dependencies (e.g., DataFrames, CUDA).
#   Instead, ask around, someone will help you accomplish what you need
# - your dependency implements code that logically should be implemented elsewhere
#   (e.g., if your importing CUDA, maybe we should extend ClimaComms instead).
#   Instead, ask around, someone will help you accomplish what you need

atmos_uuid = Pkg.project().dependencies["ClimaAtmos"]
direct_dependencies =
    keys(Pkg.dependencies(identity, atmos_uuid).dependencies) |> Set

known_dependencies = Set([
    # ArgParse is used to read --config_file and --job_id from command line
    "ArgParse",
    "Artifacts",
    "AtmosphericProfilesLibrary",
    "ClimaComms",
    "ClimaCore",
    "ClimaDiagnostics",
    "ClimaParams",
    "ClimaTimeSteppers",
    "ClimaUtilities",
    "CloudMicrophysics",
    "Dates",
    "DiffEqBase",
    "FastGaussQuadrature",
    "Insolation",
    "Interpolations",
    "LazyArtifacts",
    "LinearAlgebra",
    "Logging",
    # NCDatasets is used to read Earth topography, GCM driven initial conditions, orographic gravity wave data
    "NCDatasets",
    "NVTX",
    "RRTMGP",
    # Random is used to reset seed for random number generator used for cloudy RRTMGP runs to enable bit-wise reproducibility for tests
    "Random",
    "SciMLBase",
    "StaticArrays",
    # Statistics is used to call 'mean' on ClimaCore Fields
    "Statistics",
    "SurfaceFluxes",
    "Thermodynamics",
    # UnrolledUtilities is used to make some loops and maps in the gravity-wave code type-stable
    "UnrolledUtilities",
    "YAML",
])

diff = setdiff(direct_dependencies, known_dependencies)

if !isempty(diff)
    println("Detected new dependencies: $diff")
    println("Please, double check if you really need the new dependencies")
    println(
        "If you do, edit the dependencies.jl file adding a note about where the packages are used",
    )
end

otherdiff = setdiff(known_dependencies, direct_dependencies)
if !isempty(otherdiff)
    println("Detected stale dependencies: $otherdiff")
    println("Please, edit the dependencies.jl file")
end

Test.@test direct_dependencies == known_dependencies
