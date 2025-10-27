## Installation instructions

ClimaAtmos.jl is a [registered Julia package](https://julialang.org/packages/). To install

```julia
julia> using Pkg

julia> Pkg.add("ClimaAtmos")
```

Alternatively, you can clone the `ClimaAtmos`
[repository](https://github.com/CliMA/ClimaAtmos.jl) with:

```
$ git clone https://github.com/CliMA/ClimaAtmos.jl.git
```
This is useful if you want to keep up with bleeding-edge changes between major releases.

Now change into the `ClimaAtmos.jl` directory with 

```
$ cd ClimaAtmos.jl
```

To use ClimaAtmos, you need to instantiate all dependencies with:

```
$ julia --project
julia> ]
(ClimaAtmos) pkg> instantiate
```

## Some common terminology in ClimaAtmos
The following terms are frequently used within the source code and between collaborators. Feel free to open a GitHub issue if you come across any other key terms that we've missed here.
- `Yₜ`: The tendency state vector, where `Yₜ.sfc` components are modified.
- `Y`: The current state vector.
- `p`: Cache containing parameters, precomputed fields (radiation fluxes, surface
       conditions, precipitation fluxes), atmospheric model configurations, and
       slab model properties.
- `t`: Current simulation time.
