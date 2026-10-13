# Writing Code Compatible with CliMA Models

This document covers the conventions for writing code that is compatible with CliMA models. It is intended for developers who are new to the CliMA codebase. Code should also follow the guidelines in [code_style.md](code_style.md) and in [software_design_patterns.md](software_design_patterns.md).

## CliMA models and ClimaCore Fields

CliMA simulations are evaluated on a discretized space, so code must be compatible with the data structure that represents a value at every point in that space: `ClimaCore.Fields.Field`.

In general, CliMA models will evaluate functions at every point in the domain using broadcasting, which behaves similarly to broadcasting over an Array but with some important differences. For example, users should not directly index into `ClimaCore.Fields.Field`s. Instead, code should be written in a pointwise style that can be broadcasted over the entire domain. For example, if you wanted to compute `par` as a function of shortwave radiation `sw_d`:

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

When a function is broadcasted over one or more `ClimaCore.Fields.Field`(s), and the `ClimaComms.device` is a `CUDADevice`, the function will be executed as a CUDA kernel on the GPU. This means that the function must be type-stable and must not perform any operations that are not supported on the GPU. See [performance/gpu_performance.md](../performance/gpu_performance.md) for more details on writing GPU-compatible code, and [performance/branchless_code.md](../performance/branchless_code.md) for why such pointwise functions should avoid data-dependent branches.

### Spatial Derivatives

When writing code that computes spatial derivatives, use the operators provided by `ClimaCore` (e.g., `ClimaCore.grad`, `ClimaCore.div`, `ClimaCore.curl`) rather than manually indexing into fields. This ensures that the code is compatible with the discretization and can be efficiently executed on the GPU.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
