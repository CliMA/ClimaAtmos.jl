# Installation instructions

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
