## Overview
Parameters in ClimaAtmos.jl are handled by [CLIMAParameters.jl](https://github.com/CliMA/CLIMAParameters.jl). The repository stores all default values of parameters used in ClimaAtmos and has various utilities for handling parameters. It allows for easy parameter overriding without needing to change the source code directly. For more information, see the [docs](https://clima.github.io/CLIMAParameters.jl/dev/).

## How to add your own parameters to ClimaAtmos:
First, create a TOML file with the parameters you want to add/override. Here is the basic format for a single parameter:
```
[descriptive name]
alias = "<short name>"
value = <value>
type = "<type>"
```
The possible types are: `bool`, `float`, `integer`, or `string`.

#### Basic example for gravitational acceleration:
```
[gravitational_acceleration]
alias = "grav"
value = 9.81
type = "float"
```
For more info on formatting the TOML, see [here]([https://clima.github.io/CLIMAParameters.jl/dev/toml/](https://clima.github.io/CLIMAParameters.jl/dev/toml/)). 


# After CLI Removal:
Once you have created your parameter file (`parameters.toml`), you must create a separate YAML configuration file (`config.yaml`).
In the config file, enter:
```
toml: parameters.toml
```
In order to run the model, type: `julia --project=examples --config_file config.yaml`.
Note that the `--config_file` argument can take several config files, so if you have a separate config file you would like to use,
you can simply add it to the end of the command line arguments. Alternatively, you can just add your TOML config to the existing config file.

# Pre-CLI removal:
To run the model, you can either initialize the model interactively from the Julia REPL or from the command-line.
From the command-line at the top-level CA directory, just run `julia --project=examples --toml path/to/toml`

If running the model interactively, you can change `ARGS` and append `["--toml", "path/to/toml"]`.
Alternatively, you can create the `parsed_args` dictionary and manually alter it yourself: `parsed_args["toml"] = "path/to/toml"`.

If you are just overriding a parameter, you can just create the model config and run the model! 
If you add a new parameter, you need to alter the [ClimaAtmosParameters]([https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/parameters/Parameters.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/parameters/Parameters.jl#L13)) struct to include your parameter. This can temporarily be done at the top level if you need to test something quickly, but if you want to merge the code, it is best to place your new parameter inside one of the nested parameter structs.
