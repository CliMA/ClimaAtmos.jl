if joinpath(@__DIR__, "..") ∉ LOAD_PATH
    push!(LOAD_PATH, joinpath(@__DIR__, ".."))
end
using Documenter
using ClimaAtmos

makedocs(
    modules = [ClimaAtmos],
    sitename = "ClimaAtmos.jl",
    authors = "Clima",
    strict = true,
    format = Documenter.HTML(
        prettyurls = !isempty(get(ENV, "CI", "")),
        collapselevel = 1,
        mathengine = MathJax3(),
    ),
    pages = [
        "Home" => "index.md",
        "Installation instructions" => "installation_instructions.md",
        "Running instructions" => "running_instructions.md",
        "Contributor Guide" => "contributor_guide.md",
        "Function Index" => "function_index.md",
        "Equations" => "equations.md",
        "Command line arguments" => "cl_args.md",
    ],
)

deploydocs(
    repo = "github.com/CliMA/ClimaAtmos.jl.git",
    devbranch = "main",
    push_preview = true,
    forcepush = true,
)
