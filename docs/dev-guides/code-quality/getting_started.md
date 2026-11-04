# Writing Code Compatible with CliMA Models

This document covers the conventions for writing code that is compatible with CliMA models. It is intended for developers who are new to the CliMA codebase. Code should also follow the guidelines in [code_style.md](code_style.md) and in [software_design_patterns.md](software_design_patterns.md).

## CliMA models and ClimaCore Fields

CliMA simulations are evaluated on a discretized space, so code must be compatible with the data structure that represents a value at every point in that space: `ClimaCore.Fields.Field`.

CliMA models evaluate functions at every point in the domain by broadcasting, which behaves like broadcasting over an `Array` with one difference that matters here: code does not index into a `ClimaCore.Fields.Field`. Write functions in a pointwise style and broadcast them over the domain. For example, to compute `par` as a function of shortwave radiation `sw_d`:

```julia
# ❌ Not compatible with Fields
function compute_PAR!(par, sw_d)
    for i in eachindex(par)
        par[i] = 0.5 * sw_d[i]
    end
end
```

```julia
# ✅ Compatible with Fields
function compute_PAR(sw_d)
    return 0.5 * sw_d
end

# Then broadcast over the entire domain:
par .= compute_PAR.(sw_d)
```

When a function is broadcasted over one or more `ClimaCore.Fields.Field`s and the `ClimaComms.device` is a `CUDADevice`, the function runs as a CUDA kernel on the GPU, so it must be type-stable and use only operations the GPU supports. See [performance/gpu_performance.md](../performance/gpu_performance.md) for writing GPU-compatible code, and [performance/branchless_code.md](../performance/branchless_code.md) for why such pointwise functions should avoid data-dependent branches.

### Spatial Derivatives

For spatial derivatives, use the operators provided by `ClimaCore.Operators` (gradient, divergence, curl, interpolation). They match the discretization and run on the GPU.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
