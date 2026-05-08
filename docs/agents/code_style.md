# Code Style Guide

This guide covers formatting, variable conventions, and Git workflow for CliMA repositories.

## 1. JuliaFormatter

The root `.JuliaFormatter.toml` is the authoritative source of truth for code formatting. Run the formatter locally before committing:

```bash
julia -e 'using JuliaFormatter; format(".")'
```

### Version consistency

Match the major version of `JuliaFormatter` used in CI to prevent unnecessary diff churn. The CI workflow typically pins the major version:

```yaml
- uses: julia-actions/julia-format@v3
  with:
    version: '1'
```

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

## 4. Git workflow

### Rebasing over merging

Prefer **rebasing** over merging to maintain a linear commit history:

```bash
git fetch origin main
git rebase origin/main
```

### Starting a new task

Ensure your branch is based on the latest remote `main`:

```bash
git stash
git checkout main
git pull origin main
git checkout -b your/branch-name
git stash pop
```

### Functional commits

Each commit should represent a logical unit of work and maintain model compilability.

## 5. Feature removal

When a feature is deprecated or removed, follow the full cleanup protocol:

1. **Source removal**: delete implementation code, structs, and methods.
2. **Configuration purge**: remove options from config files and parsers. Ensure that choosing a removed option triggers a clear `error` listing valid alternatives.
3. **Test suite cleanup**: delete targeted tests; update integration tests to use supported alternatives. Mirror changes between `src/` and `test/`.
4. **Dependency slimming**: remove packages that were exclusively used by the removed feature from `Project.toml`. See [Dependency Management Guide](dependency_management.md).
5. **Documentation update**: update docstrings and docs to reflect the removal.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
