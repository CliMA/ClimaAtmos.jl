# Docstring Standard

This guide defines the layout and conventions for docstrings across CliMA repositories.

## Structure

1. **One-line summary**: a single sentence explaining what the function or struct does.
2. **Details (optional)**: a brief paragraph with additional context or mathematical formulas.
3. **Arguments / Fields**: a list under `# Arguments` or `# Fields`.
4. **Returns**: a `# Returns` section is required for any function whose return value is not a simple scalar of an obvious type (for example, a `NamedTuple`, multiple values, or a `Field` with non-obvious units). Use it whenever the function appears in user-facing documentation built by Documenter.
5. **Signature**: explicitly include the function signature at the top of the docstring, especially for functions with complex dispatch or many arguments.

## Example: function

```julia
\"\"\"
    gauss_hermite(FT, N)

Return Gauss-Hermite quadrature nodes and weights for order N.

The nodes are roots of the Hermite polynomial Hₙ(x).
Weights are standard Gauss-Hermite weights for the physicists' Hermite polynomials.

# Arguments
- `FT`: floating point type for result
- `N`: quadrature order (1-5)
\"\"\"
function gauss_hermite(::Type{FT}, N::Int) where {FT}
    # ...
end
```

## Example: struct

```julia
\"\"\"
    GaussianSGS <: AbstractSGSDistribution

Gaussian (normal) distribution for SGS fluctuations.
Uses Gauss-Hermite quadrature for bivariate integration.
\"\"\"
struct GaussianSGS <: AbstractSGSDistribution end
```

## Example: functor

For callable structs, attach the docstring to the call method:

```julia
\"\"\"
    (eval::MicrophysicsEvaluator)(T_hat, q_tot_hat)

Evaluate microphysics tendencies at a quadrature point.

# Arguments
- `T_hat`: temperature at quadrature point (K)
- `q_tot_hat`: total specific humidity (kg/kg)
\"\"\"
function (eval::MicrophysicsEvaluator)(T_hat, q_tot_hat)
    # ...
end
```

## Guidelines

- **Conciseness**: avoid overly verbose descriptions. Let variable names and formulas do the work.
- **Math**: use LaTeX (in backticks or blocks) for mathematical variables or relationships.
- **Prefixes**: use library prefixes (for example, `TD.`, `SA.`) to clarify the source of external types and functions.
- **Generic math**: reference generic type constructs (for example, `one(FT)`) if relevant to implementation details.

## Documenter.jl pitfalls

### Markdown link ambiguity

Be careful with `[kg/m^3] (description)` formats in docstrings. Documenter's markdown parser interprets `[text](text)` as a link and will produce `:cross_references` errors if the parenthetical text is not a URL.

**Fix**: use parentheses for units — `(kg/m^3)` — or separate brackets and parentheses with punctuation or a line break.

Do **not** attempt to escape square brackets with backslashes (`\[...\]`) in Julia string literals; this causes invalid escape sequence errors during precompilation.

### Missing docstrings

If `makedocs` fails with "Missing docstrings" errors, ensure every exported symbol with a docstring is included in a documentation page via an `@docs` or `@autodocs` block.

### Undefined symbols

Use fully qualified names in docstrings (for example, `Thermodynamics.ThermodynamicsParameters`) to ensure Documenter's link generator can resolve them across package boundaries.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
