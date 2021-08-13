# Contributor's Guide
This guide is heavily based on the [MetPy contributors guide](https://github.com/Unidata/MetPy/blob/main/CONTRIBUTING.md)

## Introduction
First of all, thank you for considering contributing to ```ClimaAtmos.jl```! Please use the following as a guideline on how to best contribute to the project.  One core philosophy in ```ClimaAtmos.jl``` is to reuse as much of Julia's native infrastructure as possible.

## What Can I Do?
Create an issue or pull request on the main ```ClimaAtmos.jl``` repostory. 

## Ground Rules
Be nice.

## Reporting a bug
Please file an issue with the tag ```Bug``` and provide a minimum working example of the bug.

## Pull Requests
Create a pull request with a descriptive title and a few sentences explaining the purpose of the pull request.

## Documentation
Complex functions should have docstrings. For example,

```julia

"""
complex_function(a,b; key_a = nothing)

# Description
This function does a complex calculation

# Arguments
- `a`: String. a string for complex names
- `b`: Number. a number for complex operations

# Keyword Arguments
- `key_a`: default = nothing. a keyword argument for nothing

# Return 
- nothing
"""
function complex_function(a,b; key_a = nothing)
    # do complex things
    return nothing
end

```

## Tests
Unit tests and example scripts that run are welcome. 