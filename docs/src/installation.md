# Installation

## Julia

Download and install Julia from [julialang.org/downloads](https://julialang.org/downloads/).

## ClimaAtmos

ClimaAtmos is a registered Julia package:

```julia
using Pkg
Pkg.add("ClimaAtmos")
```

To develop or track the latest changes, clone the repository instead:

```bash
git clone https://github.com/CliMA/ClimaAtmos.jl.git
cd ClimaAtmos.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## GPU support (optional)

To run on GPU, install the appropriate backend and set the device:

```julia
using Pkg
Pkg.add("CUDA")
```


## Verify

```julia
import ClimaAtmos as CA
import CUDA
```

If this loads without errors, you're ready to go. Continue to [Your First Simulation](@ref).
