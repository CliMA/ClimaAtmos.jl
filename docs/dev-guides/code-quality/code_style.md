# Code Style Guide

This guide covers formatting and naming conventions for CliMA repositories. For Git workflow and feature-removal protocol, see [onboarding.md §§5, 7](../workflow/onboarding.md).

## 1. JuliaFormatter

The root `.JuliaFormatter.toml` is the authoritative source of truth for code formatting. Run the formatter locally before committing:

```bash
julia -e 'using JuliaFormatter; format(".")'
```

or install JuliaFormatter as an app and use directly from the command-line:

```julia-repl
julia> import Pkg; Pkg.Apps.add("JuliaFormatter")
```

and add `~/.julia/bin/` to your PATH.

Then you can run the formatter directly:

```bash
jlfmt -i .
```

### Version consistency

Match the JuliaFormatter version used in CI to prevent unnecessary diff churn. Repos use the `julia-actions/julia-format` GitHub Action and pin a JuliaFormatter major version via the `version:` input:

```yaml
- uses: julia-actions/julia-format@v4
  with:
    version: '1'   # JuliaFormatter major version; check the repo's workflow file
```

Repos subscribed to DeveloperGuides pin JuliaFormatter to **v1** via two enforced files that are kept in lockstep: `.github/workflows/julia_formatter.yml` (`version: '1'`) and the dedicated `.dev/format/Project.toml` environment used by the pre-commit hook below. The surest way to match CI locally is to use that hook (next section), which formats from the pinned env regardless of the JuliaFormatter version in your base environment — `Pkg.add("JuliaFormatter")` now installs v2 by default and produces a different diff. If you are in a repo that is not yet subscribed, cross-check `.github/workflows/JuliaFormatter.yml` (or `julia_formatter.yml`) before formatting.

### Pre-commit hooks (recommended)

To avoid ever seeing a formatter-only CI failure, set up the git pre-commit hooks defined in the repo's `.pre-commit-config.yaml`. They run on each `git commit` against your staged files and:

- run `JuliaFormatter` from a dedicated, version-pinned environment (`.dev/format/`) so the result matches the `.github/workflows/julia_formatter.yml` CI check regardless of which `JuliaFormatter` version is in your base environment, and
- trim trailing whitespace that `JuliaFormatter` leaves behind (for example in comments).

The hooks are managed with [`prek`](https://prek.j178.dev), a fast drop-in replacement for `pre-commit`. `prek` is a Python tool; the easiest way to get it without touching your Julia setup is via [`uv`](https://docs.astral.sh/uv/):

```sh
# Install uv (see https://docs.astral.sh/uv/getting-started/installation/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install prek as a standalone tool
uv tool install prek
```

Then, from the repository root, install the git hooks once:

```sh
prek install
```

That's it — the hooks now run automatically on every commit. `julia` must be on your `PATH`; the first run instantiates and precompiles `.dev/format/`, which takes a minute, and is fast thereafter.

To format and clean the whole repository on demand (handy after a large change):

```sh
prek run --all-files
```

The original `pre-commit` works too if you already have it (`pip install pre-commit` / `uv tool install pre-commit`, then `pre-commit install`); the config file is shared.

> [!NOTE]
> When a hook reformats a staged file, the commit is aborted and the file is left changed on disk — this is expected. Review the changes, `git add` them, and commit again.

### Avoiding formatting noise

Do not manually format code inconsistently with the formatter. If the formatter produces unwanted results, adjust `.JuliaFormatter.toml` rather than overriding manually.

Be cautious with `git checkout -- .` to undo formatting changes — this also undoes any uncommitted functional changes. Prefer `git checkout -p` or `git add -i` for selective staging.

## 2. Variable locality

Constants specific to a physical algorithm should be defined as local variables inside the function, not as global module constants:

```julia
function compute_gradient(x, y)
    # Algorithmic constant local to this function
    ε = 1e-8
    # ... logic ...
end
```

This minimizes global namespace pollution and improves code clarity.

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
- Functions that mutate one of their arguments (conventionally the first) end in `!` — e.g. `update!`, `compute_tendency!`.

### Function names

- **Prefer full words over abbreviations.** `compute_strain_rate_full!` is better than `csrf!`. A few extra characters at the definition site are a vanishingly small cost compared to the cost of decoding an unfamiliar abbreviation every time a reader encounters it.
- **Acceptable abbreviations** are universally-understood physics/math symbols (`Φ`, `ρ`, `χ`, `θ`) and well-established acronyms used widely in the relevant subfield (`EDMF`, `RRTMGP`, `SGS`, `PDF`, `LES`). When in doubt, spell it out.
- **Lazy field prefixes (ClimaCore-based repos):** functions that return a lazy cell-center–valued field are prefixed with `ᶜ`; those that return a lazy cell-face–valued field are prefixed with `ᶠ`. Unprefixed functions are understood to be pointwise. For example, `ᶜρ` is a lazy field at cell centers; `ρ` (no prefix) is a pointwise scalar.

### Type names

- **Abstract types: use the bare concept name, not an `Abstract`-prefixed form.** Prefer `CloudModel`, `SpongeModel`, `JacobianAlgorithm` over `AbstractCloudModel`, `AbstractSpongeModel`, `AbstractJacobianAlgorithm`. The concept name reads more naturally in dispatch signatures (`f(x::CloudModel)`) and in documentation. Some legacy code uses `AbstractFoo`; keep it consistent within an existing module, but new hierarchies should drop the prefix.
- **Common suffixes** signal what kind of type a struct is. Use them to make intent obvious at the call site:
  - `…Model`: dispatch tag for a parameterization choice (e.g. `SmagorinskyLilly`, `EDMFModel`).
  - `…Method` / `…Algorithm`: algorithmic choice (e.g. `JacobianAlgorithm`, `TracerNonnegativityMethod`).
  - `…Parameters` or `…Params`: immutable bag of numerical parameters (e.g. `ThermodynamicsParameters`).
  - `…Cache`: mutable workspace or precomputed state (e.g. `AtmosCache`).
- **Avoid generic `…Type` or `…Helper` suffixes** — they don't tell the reader what kind of thing they are looking at.

### Variables

- Follow the conventions in the [Variable List](variable_list.md).
- Avoid one-character names like `l` (lowercase el), `O` (uppercase oh), or `I` (uppercase eye) — they are visually ambiguous.
- One-letter names from physics/math (`T`, `ρ`, `χ`, `Φ`) are fine when they match standard notation in the surrounding code.

### Unicode

- Limit use of Unicode. Avoid combining accents (dot, hat, vec) that create visually ambiguous characters.
- Use only standard Greek letters (`α`, `β`, `Δ`, `χ`, `ρ`, …) and common math symbols (`∇`, `∂`, `∫`, `≤`).
- Exception: the modifier-letter prefixes `ᶜ` and `ᶠ` are idiomatic in ClimaCore-based repos for lazy center/face field functions (see "Function names" above). They are visually distinct and unambiguous.

### Line length

The JuliaFormatter margin is the authoritative line-length limit. Most repos use `margin = 92`; check the repo's `.JuliaFormatter.toml`.

### Imports

Group `using`/`import` statements in the following order, separated by blank lines:

1. Standard library imports.
2. Related third-party imports.
3. Local/application-specific imports.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
