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

To run the model, type:

```bash
julia --project=.buildkite .buildkite/ci_driver.jl --config_file config.yaml --job_id my_job
```

Note that the `--config_file` argument can take several config files, so if you have a separate config file you would like to use,
you can add it to the end of the command line arguments. Alternatively, you can add your TOML config to the existing config file.
