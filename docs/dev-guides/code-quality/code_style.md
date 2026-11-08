# Code Style Guide

This guide covers formatting and naming conventions for CliMA repositories. For Git workflow and feature-removal protocol, see [onboarding.md §§5, 7](../workflow/onboarding.md).

## 1. JuliaFormatter

The root `.JuliaFormatter.toml` is authoritative for code formatting. Run the formatter locally before committing:

```bash
julia -e 'using JuliaFormatter; format(".")'
```

or, on Julia 1.12+, install JuliaFormatter as an app (`Pkg.Apps` is absent on 1.11 and earlier, including the 1.10 LTS; check with `isdefined(Pkg, :Apps)`):

```julia-repl
julia> import Pkg; Pkg.Apps.add("JuliaFormatter")
```

add `~/.julia/bin/` to your PATH, and run it from the command line:

```bash
jlfmt -i .
```

### Version consistency

Match the JuliaFormatter major version CI uses; a different version reformats unrelated lines. Repos use the `julia-actions/julia-format` GitHub Action and pin the major version via the `version:` input:

```yaml
- uses: julia-actions/julia-format@v4
  with:
    version: '1'   # JuliaFormatter major version; check the repo's workflow file
```

The pinned major version varies across CliMA repos: some pin `'1'`, others `'2'`, and some leave the default. Check `.github/workflows/JuliaFormatter.yml` (or `julia_formatter.yml`) in the repo you are working in before formatting.

### Pre-commit hooks with prek

To run formatter checks on each commit, follow the pattern used in ClimaAtmos.jl:

- add a `.pre-commit-config.yaml` at repo root,
- optionally use a dedicated formatter environment (for example `.dev/format/`),
- keep the CI formatter check and the local hook aligned.

Use [`prek`](https://prek.j178.dev) to manage hooks:

```sh
# Install uv (https://docs.astral.sh/uv/getting-started/installation/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install prek
uv tool install prek

# From your repo root: install git hooks once
prek install
```

Hooks then run on each `git commit`, on staged files. For manual runs:

- `prek run` checks the files selected by normal hook matching.
- `prek run --all-files` checks the whole repository.

`pre-commit` also works if you already use it; `prek` is a drop-in replacement.

> [!NOTE]
> If a hook reformats staged files, the commit is aborted and files are left
> modified on disk. Review, `git add`, and commit again.

### Avoiding formatting noise

Hand-formatting that disagrees with the formatter is undone on its next run. If the formatter produces unwanted results, adjust `.JuliaFormatter.toml`.

`git checkout -- .` to undo formatting changes also discards uncommitted functional changes. Use `git checkout -p` or `git add -i` to select hunks.

## 2. Variable locality

Constants specific to a physical algorithm should be defined as local variables inside the function, not as global module constants:

```julia
function compute_gradient(x, y)
    # Algorithmic constant local to this function
    ε = 1e-8
    # ... logic ...
end
```

This keeps the module namespace small.

## 3. File organization

For large source files, use visual section headers to group related functionality:

```julia
# ============================================================================
# Quadrature Evaluators
# ============================================================================
```

The `test/` directory structure should mirror `src/`:

- **Source**: `src/parameterized_tendencies/microphysics/tendency.jl`
- **Test**: `test/parameterized_tendencies/microphysics/tendency.jl`

## 4. Naming and syntax conventions

### Capitalization

- Modules, structs, and types use `TitleCase`.
- Functions and variables use `snake_case` (lowercase, words separated by underscores).
- Constants use `SCREAMING_SNAKE_CASE`.
- Functions that mutate one of their arguments (conventionally the first) end in `!`, e.g. `update!`, `compute_tendency!`.

### Function names

- **Prefer full words over abbreviations.** `compute_strain_rate_full!` is better than `csrf!`. The author types the full name once; a reader decodes an abbreviation on every encounter.
- **Acceptable abbreviations** are standard physics/math symbols (`Φ`, `ρ`, `χ`, `θ`) and acronyms standard in the relevant subfield (`EDMF`, `RRTMGP`, `SGS`, `PDF`, `LES`). When in doubt, spell it out.
- **Staggered-grid prefixes (ClimaCore-based repos):** `ᶜ` marks a name whose value is a cell-center field and `ᶠ` a cell-face field, including functions that return one; pointwise names have no prefix. See [ecosystem_conventions.md §3](../architecture/ecosystem_conventions.md).

### Type names

- **Abstract types: use the bare concept name, not an `Abstract`-prefixed form.** Prefer `CloudModel`, `SpongeModel`, `JacobianAlgorithm` over `AbstractCloudModel`, `AbstractSpongeModel`, `AbstractJacobianAlgorithm`. The concept name reads better in dispatch signatures (`f(x::CloudModel)`) and in documentation. Some legacy code uses `AbstractFoo`; keep it consistent within an existing module, but new hierarchies should drop the prefix.
- **Common suffixes** signal what kind of type a struct is:
  - `…Model`: dispatch tag for a parameterization choice (e.g. `SmagorinskyLilly`, `EDMFModel`).
  - `…Method` / `…Algorithm`: algorithmic choice (e.g. `JacobianAlgorithm`, `TracerNonnegativityMethod`).
  - `…Parameters` or `…Params`: immutable bag of numerical parameters (e.g. `ThermodynamicsParameters`).
  - `…Cache`: mutable workspace or precomputed state (e.g. `AtmosCache`).
- **Avoid generic `…Type` or `…Helper` suffixes**: they say nothing about the type's role.

### Variables

- Follow the conventions in the [Variable List](variable_list.md).
- Avoid one-character names like `l` (lowercase el), `O` (uppercase oh), or `I` (uppercase eye); they are visually ambiguous.
- One-letter names from physics/math (`T`, `ρ`, `χ`, `Φ`) are fine when they match standard notation in the surrounding code.

### Unicode

- Limit use of Unicode. Avoid combining accents (dot, hat, vec) that create visually ambiguous characters.
- Use only standard Greek letters (`α`, `β`, `Δ`, `χ`, `ρ`, …) and common math symbols (`∇`, `∂`, `∫`, `≤`).
- Exception: the staggered-grid prefixes `ᶜ` and `ᶠ` (see "Function names" above).

### Line length

The JuliaFormatter margin is the authoritative line-length limit. Most repos use `margin = 92`; check the repo's `.JuliaFormatter.toml`.

### Imports

Group `using`/`import` statements in the following order, separated by blank lines:

1. Standard library imports.
2. Related third-party imports.
3. Local/application-specific imports.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
