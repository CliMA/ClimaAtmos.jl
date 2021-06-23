using Documenter
using Aleph


abstractions = ["abstractions.md"]

makedocs(
    sitename = "Aleph",
    authors = "Andre, Tobi, Jia",
    format = Documenter.HTML(collapselevel = 1, mathengine = MathJax3()),
    pages = [
    "Home" => "index.md",
    "Abstractions" => abstractions,
    "Contributor Guide" => "contributor_guide.md",
    "Function Index" => "function_index.md",
    ],
    modules = [Aleph]
)

deploydocs(repo = "github.com/CliMA/Aleph.jl.git", devbranch = "main")

