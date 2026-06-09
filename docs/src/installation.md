# Installation

## Julia

Download and install Julia from [julialang.org/downloads](https://julialang.org/downloads/).
If you are new to Julia's package manager, the official guides on
[environments](https://pkgdocs.julialang.org/v1/environments/) and
[managing packages](https://pkgdocs.julialang.org/v1/managing-packages/) are worth a look.

## Installing ClimaAtmos

ClimaAtmos is a registered Julia package. To install it, open the REPL, type `]` to enter
the package manager, and add it:

```julia-repl
pkg> add ClimaAtmos
```

(equivalently, `import Pkg; Pkg.add("ClimaAtmos")` from the Julia prompt).

This is the right approach when you want to use ClimaAtmos as a library in your own
project environment -- for example, to script simulations or post-process output alongside
other packages. Add it to the environment you are working in, exactly as you would any
other dependency.

## Running from a cloned repository

Most standalone simulations are launched from a clone of the repository using its bundled 
`.buildkite` environment, which pins the exact dependency versions used in the
continuous integration (CI) pipeline. Clone the repository and instantiate that environment:

```bash
git clone https://github.com/CliMA/ClimaAtmos.jl.git
cd ClimaAtmos.jl
julia --project=.buildkite -e 'using Pkg; Pkg.instantiate()'
```

You can then run any configuration through the driver:

```bash
julia --project=.buildkite .buildkite/ci_driver.jl \
    --config_file config/model_configs/<config>.yml --job_id <job_id>
```

The `.buildkite` project `dev`-depends on the checked-out source, so local edits take
effect immediately. See [Script vs Config Interface](@ref) for the configuration format
and [Single Column Models](@ref) for ready-made example configurations.

## GPU support (optional)

ClimaAtmos selects its compute device through
[ClimaComms](https://clima.github.io/ClimaComms.jl/stable/). To run on an NVIDIA GPU,
add `CUDA.jl` to your environment and load the backend with
[`ClimaComms.@import_required_backends`](@extref):

```julia
import Pkg
Pkg.add("CUDA")

import ClimaComms
ClimaComms.@import_required_backends   # loads CUDA.jl when it is installed
ClimaComms.device()   # `CUDADevice` when a GPU is active, a CPU device otherwise

import ClimaAtmos as CA
```

The device is auto-detected: when a CUDA-capable GPU is available it is used by default.
You can force the choice with the `CLIMACOMMS_DEVICE` environment variable (`"CUDA"` or
`"CPU"`), or, in a YAML configuration, with the `device` key.

If these load without errors, you're ready to go. Continue to [Your First Simulation](@ref).
