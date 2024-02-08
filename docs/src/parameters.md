## Overview
Parameters in ClimaAtmos.jl are handled by [CLIMAParameters.jl](https://github.com/CliMA/CLIMAParameters.jl). The repository stores all default values of parameters used in ClimaAtmos and has various utilities for handling parameters. It allows for easy parameter overriding without needing to change the source code directly. For more information, see the [docs](https://clima.github.io/CLIMAParameters.jl/dev/).

## How to add your own parameters to ClimaAtmos:
First, create a TOML file with the parameters you want to add/override. Here is the basic format for a single parameter:
```
[descriptive name]
value = <value>
type = "<type>"
```
The possible types are: `bool`, `float`, `integer`, or `string`.

#### Basic example for gravitational acceleration:
```
[gravitational_acceleration]
value = 9.81
type = "float"
```
For more info on formatting the TOML, see [here](https://clima.github.io/CLIMAParameters.jl/dev/toml/).

Once you have created your parameter file (`parameters.toml`), you must create a separate YAML configuration file (`config.yaml`).
In the config file, enter:
```
toml: parameters.toml
```
In order to run the model, type: `julia --project=examples --config_file config.yaml`.
Note that the `--config_file` argument can take several config files, so if you have a separate config file you would like to use,
you can simply add it to the end of the command line arguments. Alternatively, you can just add your TOML config to the existing config file.
