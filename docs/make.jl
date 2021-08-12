using Documenter
using ClimaAtmos


abstractions = ["abstractions.md"]
models = ["3d_dry_euler_total_energy.md",
          "3d_moist_euler_total_energy.md",
          "barotropic_fluid.md",
         ]

makedocs(
    sitename = "ClimaAtmos",
    authors = "Andre, Tobi, Jia",
    format = Documenter.HTML(collapselevel = 1, mathengine = MathJax3()),
    pages = [
    "Home" => "index.md",
    "Abstractions" => abstractions,
    "Models" => models,
    "Contributor Guide" => "contributor_guide.md",
    "Function Index" => "function_index.md",
    ],
    modules = [ClimaAtmos]
)

deploydocs(repo = "github.com/CliMA/ClimaAtmos.jl.git", devbranch = "main")

