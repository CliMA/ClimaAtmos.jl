## ClimaCore Examples

* `bickleyjet/`: 2D spectral element implementations of the bickleyjet testcase
* `column/`: 1D column tests for finite difference operators
* `hybrid/`: 2D slice model testcases for the hybrid spectral / fd domains
* `sphere/`: 2D dycore shallow water testcases on the cubed sphere
* `sphere3d/`: 3D dycore testcases on the hybrid spectral / fd cubed sphere domain


### Use ClimaCore Examples from the ClimaCore parent package

Instantiate the examples environment (this will download and install the necessary packages)

    cd ClimaCore
    julia --project=examples -e 'using Pkg; Pkg.instantiate()'

The default Manifest.toml uses [`Pkg.develop`](https://pkgdocs.julialang.org/v1/api/#Pkg.develop)
to track the current versions of ClimaCore.jl and subpackages in the `lib` directory.

You can now execute the examples in the example projects, and ClimaCore changes will be tracked in the env:

    julia --project=examples
    julia> include("examples/bickleyjet/bickleyjet_cg.jl")

Or alternatively:

    julia --project=examples examples/bickleyjet/bickleyjet_cg.jl
