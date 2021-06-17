using Documenter
using Aleph

makedocs(
    sitename = "Aleph",
    format = Documenter.HTML(collapselevel = 1),
    pages = [
    "Home" => "index.md",
    "Function Index" => "function_index.md",
    ],
    modules = [Aleph]
)

deploydocs(repo = "github.com/CliMA/Aleph.jl.git")

