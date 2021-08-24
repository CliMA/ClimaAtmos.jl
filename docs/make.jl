using Documenter
using ClimaAtmos

abstractions = ["abstractions.md"]
models = ["models.md"]

makedocs(
    sitename = "ClimaAtmos.jl",
    authors = "Andre, Toby, Jia",
    format = Documenter.HTML(collapselevel = 1, mathengine = MathJax3()),
    pages = [
        "Home" => "index.md",
        "Abstractions" => abstractions,
        "Contributor Guide" => "contributor_guide.md",
        "Function Index" => "function_index.md",
    ],
    modules = [ClimaAtmos],
)

deploydocs(repo = "github.com/CliMA/ClimaAtmos.jl.git", devbranch = "main")
