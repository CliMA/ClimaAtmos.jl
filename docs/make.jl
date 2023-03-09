using Documenter
using Documenter: doctest
using ClimaAtmos
using Base.CoreLogging
using DocumenterCitations

disable_logging(Base.CoreLogging.Info) # Hide doctest's `@info` printing
doctest(ClimaAtmos)
disable_logging(Base.CoreLogging.BelowMinLevel) # Re-enable all logging

makedocs(
    CitationBibliography(joinpath(@__DIR__, "bibliography.bib")),
    modules = [ClimaAtmos],
    sitename = "ClimaAtmos.jl",
    authors = "Clima",
    strict = true,
    checkdocs = :exports,
    format = Documenter.HTML(
        prettyurls = !isempty(get(ENV, "CI", "")),
        collapselevel = 1,
        mathengine = MathJax3(),
    ),
    pages = [
        "Home" => "index.md",
        "Installation instructions" => "installation_instructions.md",
        "Contributor Guide" => "contributor_guide.md",
        "Equations" => "equations.md",
        "EDMF Equations" => "edmf_equations.md",
        "Gravity wave" => "gravity_wave.md",
        "REPL scripts" => "repl_scripts.md",
        "Command line arguments" => "cl_args.md",
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
