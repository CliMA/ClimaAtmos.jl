# Architectural Boundaries

This guide defines the layered architecture used across CliMA model repositories and the rules that keep boundaries clean. Each repo's `*_specific.md` (linked from [AGENTS.md](../../AGENTS.md)) maps these layers to its concrete directories.

## 1. Layer diagram

```
┌─────────────────────────────────────────────────────┐
│  Parameterizations layer                            │
│  Defines HOW a physical tendency is computed.       │
└────────────────────┬────────────────────────────────┘
                     │ results (scalars / NamedTuples)
┌────────────────────▼────────────────────────────────┐
│  Infrastructure layer                               │
│  Defines WHERE results are stored and HOW the model │
│  time-steps them.                                   │
└─────────────────────────────────────────────────────┘
```

**Rule**: if a file defines how a physical tendency is calculated, it belongs in the parameterizations layer. If it only defines where a result is stored or how it is integrated, it belongs in the infrastructure layer. Do not put orchestration logic in physics files.

## 2. Parameter container design

- Containers should be focused on the prognostic tendencies they serve.
- Do not add "zombie" forward-compatibility fields to support not-yet-refactored callers; refactor the callers instead.
- Exclude diagnostic or calibration parameters from physics containers; pass them explicitly from the infrastructure layer.

## 3. Avoid hidden field dependencies

Do not access internal or undocumented fields of a sub-package's parameter struct directly (for example, `cm2p.internal_field`). Use the documented public accessor or the primary parameter source.

This makes physics refactors in sub-packages safe without cascading breakage in the model.

Bad:

```julia
# Brittle: depends on internal field names of a microphysics struct
w_sed = cm2p.rtv + cm2p.ctv
```

Preferred:

```julia
# Robust: access from the primary, stable parameter source
w_sed = cmc.Ch2022.rain + cmc.stokes.liquid
```

## 4. Module import rules

Inside `src/`, do not add local `using` or `import` patterns between submodules. See [SDP 2](software_design_patterns.md). Prefer explicit qualification or project-established module patterns.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
