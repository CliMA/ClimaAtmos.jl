using Documenter
using ClimaAtmos

models = ["models.md"]

makedocs(
    sitename = "ClimaAtmos.jl",
    authors = "Clima",
    format = Documenter.HTML(collapselevel = 1, mathengine = MathJax3()),
    pages = [
        "Home" => "index.md",
        "Installation instructions" => "installation_instructions.md",
        "Running instructions" => "running_instructions.md",
        "Contributor Guide" => "contributor_guide.md",
        "Function Index" => "function_index.md",
    ],
    modules = [ClimaAtmos],
)

deploydocs(repo = "github.com/CliMA/ClimaAtmos.jl.git", devbranch = "main")
