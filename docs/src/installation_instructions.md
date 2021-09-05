# Installation instructions

Download the `ClimaAtmos`
[source](https://github.com/CliMA/ClimaAtmos.jl) with:

```
$ git clone https://github.com/CliMA/ClimaAtmos.jl.git
```

Now change into the `ClimaAtmos.jl` directory with 

```
$ cd ClimaAtmos.jl
```

To use ClimaAtmos, you need to add the `ClimaCore` package and instantiate all dependencies with:

```
$ julia --project
julia>]
(v1.6) pkg> add https://github.com/CliMA/ClimaCore.jl.git
(v1.6) pkg> instantiate
```
