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
    format = Documenter.HTML(
        prettyurls = !isempty(get(ENV, "CI", "")),
        collapselevel = 1,
        mathengine = MathJax3(),
        size_threshold_ignore = ["repl_scripts.md", "available_diagnostics.md"],
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
                "Single Column Models" => "single_column_prospect.md",
                "Radiative Equilibrium Example" => "radiative_equilibrium.md",
                "Restarts and Checkpoints" => "restarts.md",
                "REPL Debugging Workflow" => "repl_scripts.md",
            ],
            "Configuration & Parameters" => [
                "Custom Configurations" => "config.md",
                "Parameters" => "parameters.md",
            ],
            "Computing and Saving Diagnostics" => "diagnostics.md",
        ],
        "Explanation" => [
            "Dynamics & Numerics" => [
                "Governing Equations" => "equations.md",
                "Implicit Solver" => "implicit_solver.md",
                "Integer Time (ITime)" => "itime.md",
            ],
            "Physics & Parameterizations" => [
                "Microphysics" => "microphysics.md",
                "EDMF: Prognostic Equations" => "edmf_equations.md",
                "EDMF: Diagnostic Equations" => "diagnostic_edmf_equations.md",
                "Gravity Wave Drag" => "gravity_wave.md",
                "Ocean Surface Albedo" => "surface_albedo.md",
                "Topography Representation" => "topography.md",
            ],
        ],
        "Reference" => [
            "API" => "api.md",
            "Glossary" => "glossary.md",
            "Grids" => "grids.md",
            "Setups" => "setups.md",
            "Surface Conditions" => "surface_conditions.md",
            "Passive Tracers" => "tracers.md",
            "Trace Gases (Radiation)" => "trace_gases.md",
            "Available Diagnostics" => "available_diagnostics.md",
            "Bibliography" => "references.md",
        ],
        "Developer Guide" => [
            "Contributing" => "contributor_guide.md",
            "Buildkite Longrun Jobs" => "longruns.md",
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
