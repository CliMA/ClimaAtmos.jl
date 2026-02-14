using Documenter
using Documenter: doctest
using ClimaAtmos
using Base.CoreLogging
using DocumenterCitations

disable_logging(Base.CoreLogging.Info) # Hide doctest's `@info` printing
bib = CitationBibliography(joinpath(@__DIR__, "bibliography.bib"))
include(joinpath(@__DIR__, "src", "config_table.jl"))
doctest(ClimaAtmos; plugins = [bib])
disable_logging(Base.CoreLogging.BelowMinLevel) # Re-enable all logging

makedocs(;
    plugins = [bib],
    modules = [ClimaAtmos],
    sitename = "ClimaAtmos.jl",
    authors = "Clima",
    checkdocs = :exports,
    format = Documenter.HTML(
        prettyurls = !isempty(get(ENV, "CI", "")),
        collapselevel = 1,
        mathengine = MathJax3(),
        size_threshold_ignore = ["repl_scripts.md"],
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Contributor Guide" => "contributor_guide.md",
        "Equations" => "equations.md",
        "Microphysics" => "microphysics.md",
        "EDMF Equations" => "edmf_equations.md",
        "Diagnostics" => "diagnostics.md",
        "Grids" => "grids.md",
        "Setups" => "setups.md",
        "Available Diagnostics" => "available_diagnostics.md",
        "Diagnostic EDMF Equations" => "diagnostic_edmf_equations.md",
        "Gravity Wave Drag Parameterizations" => "gravity_wave.md",
        "Ocean Surface Albedo Parameterization" => "surface_albedo.md",
        "Topography Representation" => "topography.md",
        "Tracers" => "tracers.md",
        "Implicit Solver" => "implicit_solver.md",
        "Radiative Equilibrium" => "radiative_equilibrium.md",
        "Single Column Model" => "single_column_prospect.md",
        "Restarts and checkpoints" => "restarts.md",
        "REPL scripts" => "repl_scripts.md",
        "Configuration" => "config.md",
        "Parameters" => "parameters.md",
        "Longruns" => "longruns.md",
        "The time type" => "itime.md",
        "API" => "api.md",
        "References" => "references.md",
    ],
)

deploydocs(
    repo = "github.com/CliMA/ClimaAtmos.jl.git",
    devbranch = "main",
    push_preview = true,
    forcepush = true,
)
