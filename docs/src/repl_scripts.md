# REPL script (debugging workflow example)


```@example

ca_dir = joinpath(@__DIR__, "..", "..")
buildkite_env = joinpath(ca_dir, ".buildkite")
using Pkg
Pkg.activate(buildkite_env)

# julia> using Revise # This is useful For REPL debugging. See also: Infiltrator.jl and Main.@infiltrate

import ClimaAtmos as CA
import SciMLBase: step!

# If you wish to run your simulation on a `CUDA` device, use

# import ClimaComms
# ClimaComms.@import_required_backends
# ENV["CLIMACOMMS_DEVICE"]="CUDA" 

# (Note that the example below runs on a single CPU, which uses 
# ENV["CLIMACOMMS_DEVICE"]="CPU")

config_file = joinpath(ca_dir, "config/model_configs/baroclinic_wave.yml")
config = CA.AtmosConfig(config_file)

# Generate temporary directory for Documenter run-script, clear after 
# demo is completed. 

temp_output_dir = mktempdir(ca_dir, cleanup=true)
config.parsed_args["output_dir"]=temp_output_dir
simulation = CA.AtmosSimulation(config) 

# Example: Advance a single timestep and explore the solution
# stored in `simulation.integrator.u`
step!(simulation.integrator) 

# Example: Update command line argument, reset simulation, re-run to completion
# Note that you can also to update the configuration `.yml` and 
# load the simulation again from the `config_file` as above in the same REPL session. 
# Note that you'd need to use `Revise` at the start of your session to apply changes 
# to the source code within your REPL session.
# e.g. 
# julia> simulation = CA.AtmosSimulation(config_file)

@info "----------------------------------"
@info "Update config arguments and re-run"
@info "----------------------------------"

config.parsed_args["dt"]="400secs"
config.parsed_args["t_end"]="800secs" 
simulation = CA.AtmosSimulation(config)
CA.solve_atmos!(simulation) 

@info "----------------------------------"
@info "Reactivate docs environment"
@info "----------------------------------"

# The final step, which is not part of the standard workflow
# resets the environment to `docs`.
# (it is necessary for the documentation generation only)
ca_dir = joinpath(@__DIR__, "..", "..")
docs_env = joinpath(ca_dir, "docs")
Pkg.activate(docs_env) 

nothing # hide
```

# Julia scripts per Buildkite job

```@example
include("repl_scripts.jl")
```
