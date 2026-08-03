using Documenter
using Documenter: doctest
using ClimaAtmos
using Base.CoreLogging
using DocumenterCitations
using DocumenterInterLinks

disable_logging(Base.CoreLogging.Info) # Hide doctest's `@info` printing
bib = CitationBibliography(joinpath(@__DIR__, "bibliography.bib"))
links = InterLinks(
    "Julia" => "https://docs.julialang.org/en/v1/objects.inv",
    "ClimaComms" => "https://clima.github.io/ClimaComms.jl/stable/objects.inv",
    "ClimaCore" => "https://clima.github.io/ClimaCore.jl/stable/objects.inv",
    "ClimaDiagnostics" => "https://clima.github.io/ClimaDiagnostics.jl/stable/objects.inv",
    "ClimaTimeSteppers" => "https://clima.github.io/ClimaTimeSteppers.jl/stable/objects.inv",
    "CloudMicrophysics" => "https://clima.github.io/CloudMicrophysics.jl/stable/objects.inv",
    "RRTMGP" => "https://clima.github.io/RRTMGP.jl/stable/objects.inv",
    "Thermodynamics" => "https://clima.github.io/Thermodynamics.jl/stable/objects.inv",
    # ClimaUtilities, ClimaParams, and SurfaceFluxes do not publish inventories
    # (objects.inv); pages link to them with plain URLs checked by linkcheck.
)
include(joinpath(@__DIR__, "src", "config_table.jl"))
doctest(ClimaAtmos; plugins = [bib, links])
disable_logging(Base.CoreLogging.BelowMinLevel) # Re-enable all logging

makedocs(;
    plugins = [bib, links],
    modules = [ClimaAtmos],
    sitename = "ClimaAtmos.jl",
    authors = "Clima",
    checkdocs = :exports,
    # Validate external links on every build, but do not fail PR builds on
    # them: external sites can be transiently unreachable, which should not
    # block unrelated PRs. Broken links surface as warnings in local builds
    # and the CI log; set LINKCHECK_STRICT=1 (e.g. in a manual or scheduled
    # run) to turn them into build failures. The check costs seconds.
    linkcheck = true,
    warnonly = isempty(get(ENV, "LINKCHECK_STRICT", "")) ? [:linkcheck] :
               Symbol[],
    format = Documenter.HTML(
        prettyurls = !isempty(get(ENV, "CI", "")),
        collapselevel = 1,
        mathengine = MathJax3(),
        size_threshold_ignore = ["available_diagnostics.md"],
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => [
            "Installation" => "installation.md",
            "Your First Simulation" => "first_simulation.md",
            "Script vs Config Interface" => "interfaces.md",
        ],
        "How-to Guides" => [
            "Running Simulations" => [
                "Single Column Models" => "single_column.md",
                "Global Simulations" => "global_simulations.md",
                "Restarts and Checkpoints" => "restarts.md",
                "Running on GPUs and MPI" => "gpu_and_mpi.md",
            ],
            "Configuration & Parameters" => [
                "Scripting Simulations" => "scripting_simulations.md",
                "Creating Custom Configurations" => "configuration.md",
            ],
            "Computing and Saving Diagnostics" => "diagnostics.md",
            "Loading and Visualizing Output" => "visualizing_output.md",
        ],
        "Explanation" => [
            "The CliMA Ecosystem" => "ecosystem.md",
            "Dynamics & Numerics" => [
                "Governing Equations" => "equations.md",
                "Implicit Solver" => "implicit_solver.md",
                "Integer Time (ITime)" => "itime.md",
            ],
            "Physics & Parameterizations" => [
                "PROPHET: Prognostic Equations" => "edmf_equations.md",
                "Microphysics" => "microphysics.md",
                "Radiation" => "radiation.md",
                "Gravity Wave Drag" => [
                    "Non-orographic Gravity Waves" => "non_orographic_gravity_wave.md",
                    "Orographic Gravity Waves" => "orographic_gravity_wave.md",
                ],
                "Ocean Surface Albedo" => "surface_albedo.md",
                "Topography Representation" => "topography.md",
            ],
        ],
        "Reference" => [
            "API" => "api.md",
            "Configuration Options" => "configuration_options.md",
            "Setups (Initial Conditions & Cases)" => "setups.md",
            "Column Datasets" => "column_datasets_reference.md",
            "Grids" => "grids.md",
            "Surface Conditions" => "surface_conditions.md",
            "Passive Tracers" => "passive_tracers.md",
            "Trace Gases (Radiation)" => "trace_gases.md",
            "Available Diagnostics" => "available_diagnostics.md",
            "Glossary" => "glossary.md",
            "Bibliography" => "references.md",
        ],
        "Developer Guide" => [
            "Contributing" => "contributor_guide.md",
            "Extending ClimaAtmos" => [
                "Adding a Setup" => "extending_setups.md",
                "Adding a Diagnostic Variable" => "extending_diagnostics.md",
                "Adding a Passive Tracer" => "extending_tracers.md",
                "Surface Conditions Internals" => "surface_conditions_internals.md",
                "Adding a Column Dataset" => "extending_column_datasets.md",
            ],
        ],
    ],
)

deploydocs(
    repo = "github.com/CliMA/ClimaAtmos.jl.git",
    devbranch = "main",
    push_preview = all(
        !isempty,
        (get(ENV, "GITHUB_TOKEN", ""), get(ENV, "DOCUMENTER_KEY", "")),
    ),
    forcepush = true,
)
