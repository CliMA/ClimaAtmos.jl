# Parameters

## Overview

Parameters in ClimaAtmos.jl are handled by [ClimaParams.jl](https://github.com/CliMA/ClimaParams.jl). The repository stores all default values of parameters used in ClimaAtmos and provides utilities for handling parameters. It lets you override parameters without changing the source code. For more information, see the [docs](https://clima.github.io/ClimaParams.jl/dev/).

## How to add your own parameters to ClimaAtmos

First, create a TOML file with the parameters you want to add/override. Here is the basic format for a single parameter:

```
[parameter_name]
value = <value>
type = "<type>"
```

The possible types are: `bool`, `float`, `integer`, `string`, or `datetime`. The `type` field is optional.

### Basic example for gravitational acceleration

```
[gravitational_acceleration]
value = 9.81
type = "float"
```

For more info on formatting the TOML, see [here](https://clima.github.io/ClimaParams.jl/dev/toml/).

Once you have created your parameter file (`parameters.toml`), you must create a separate YAML configuration file (`config.yaml`).
In the config file, enter:

```
toml: [parameters.toml]
```

To run the model:

```julia
import ClimaAtmos as CA

config = CA.AtmosConfig("config.yaml"; job_id = "my_job")
simulation = CA.AtmosSimulation(config)
CA.solve_atmos!(simulation)
```

`AtmosConfig` also accepts a vector of configuration files (later files
override earlier ones), so the `toml` entry can live in its own file alongside
an existing configuration. Alternatively, add the `toml` key to the existing
configuration file.
