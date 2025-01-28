## Introduction
To ensure reporoducibility, packages are managed using (Pkg.jl)[pkgdocs.julialang.org]. If we want to run a configurations from
`ClimaAtmos/config/`, we will need to load the packages listed in `ClimaAtmos/examples/Project.toml`.
This can be done one of two ways:
1) Within the Julia REPL
2) Using the Julia command line interface (CLI)
The latter option is particularlly useful when submitting jobs to a remote machine, such as a High-Performance Cluster (HPC).

## Environment setup:
### REPL 
Change to the ClimaAtmos directory
`$ cd ~/ClimaAtmos.jl`
invoke Julia
`$ julia`
open the built-in package manager
`julia> ]`
`@v1.10) pkg>`
Above we see the base Julia environment (for Julia version v1.10). 
We can activate the project environment in the current directory with
`@v1.10) pkg> activate .`
`(ClimaAtmos) pkg>`
Note, however, that this is the "bare bones" environment available in base directory of ClimaAtmos. 
To run an example configuration without error, we activate the example environment with
` @v1.10) pkg> activate ./examples`
Next, instantiate the environment
`(examples) pkg> instantiate`
This will install and precompile the necssary packages and their dependencies. 

### CLI
(TODO)
`julia --project=$EXAMPLES $DRIVER --config_file $CONFIG`

## Creating a simulation
```import ClimaAtmos as CA
import ClimaComms
ClimaComms.@import_required_backends

simulation = CA.get_simulation(CA.AtmosConfig("myconfig.yml"))```

## Running simulation
`CA.solve_atmos!(simulation)`
