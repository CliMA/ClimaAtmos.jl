# Installation

## Julia

Download and install Julia from [julialang.org/downloads](https://julialang.org/downloads/).
We recommend Julia v1.11, the version used to build our documentation; CI also tests v1.10.
If you are new to Julia's package manager, the official guides on
[environments](https://pkgdocs.julialang.org/v1/environments/) and
[managing packages](https://pkgdocs.julialang.org/v1/managing-packages/) are worth a look.

## Installing ClimaAtmos

ClimaAtmos is a registered Julia package. To install it, open the REPL, type `]` to enter
the package manager, and add it:

```Julia-repl
pkg> add ClimaAtmos
```

(equivalently, `import Pkg; Pkg.add("ClimaAtmos")` from the Julia prompt).

This is the right approach when you want to use ClimaAtmos as a library in your own
project environment, for example, to script simulations or post-process output alongside
other packages. Add it to the environment you are working in, as you would any
other dependency.

## Running from a cloned repository

To run standalone simulations or develop the model, clone the repository and
instantiate its environment:

```bash
git clone https://github.com/CliMA/ClimaAtmos.jl.git
cd ClimaAtmos.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
```

The first instantiation downloads and precompiles the full dependency stack
and can take tens of minutes; later sessions start quickly.

You can then run the model either interactively, through Julia scripts, or by using
YAML configuration files. See [Script vs Config Interface](@ref) for an overview
of the available workflows, and [Your First Simulation](@ref) for a quick start.

## GPU support (optional)

ClimaAtmos selects its compute device through
[ClimaComms](https://clima.github.io/ClimaComms.jl/stable/). To run on an NVIDIA GPU,
add `CUDA.jl` to your environment and load the backend with
[`ClimaComms.@import_required_backends`](@extref). When working from a clone
with `julia --project`, the active project is the package itself; add CUDA to
your default environment instead (start `Pkg.add` from a plain `julia`
session), where the load path picks it up, so the package `Project.toml`
stays unchanged:

```julia
import Pkg
Pkg.add("CUDA")

import ClimaComms
ENV["CLIMACOMMS_DEVICE"] = "CUDA"
ClimaComms.@import_required_backends   # loads CUDA.jl when CLIMACOMMS_DEVICE="CUDA"
ClimaComms.device()   # `CUDADevice` when CLIMACOMMS_DEVICE="CUDA", a CPU device otherwise

import ClimaAtmos as CA
```

The device defaults to CPU. Select a GPU with the `CLIMACOMMS_DEVICE` environment
variable (`"CUDA"`; set it before `@import_required_backends`), or, in a YAML
configuration, with the `device` key (`device: CUDADevice`).

If these load without errors, you're ready to go. Continue to [Your First Simulation](@ref).
