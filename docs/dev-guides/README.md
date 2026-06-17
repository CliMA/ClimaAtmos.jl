# DeveloperGuides

Shared engineering standards, architectural patterns, and development guidelines for human and AI developers across the [CliMA](https://clima.caltech.edu) ecosystem.

|             |                                        |
|------------:|:---------------------------------------|
| **License** | [![license][license-img]][license-url] |

[license-img]: https://img.shields.io/github/license/CliMA/DeveloperGuides
[license-url]: https://github.com/CliMA/DeveloperGuides/blob/main/LICENSE

## Usage

DeveloperGuides is included as a **Git subtree** in CliMA repositories at the standardized path `docs/dev-guides/`. The consuming repo keeps its own `AGENTS.md` at the root, which references `docs/dev-guides/AGENTS.md` (the shared guide index) plus a repo-specific guide (e.g. `docs/clima_atmos_specific.md`). See the [`AGENTS.md`](AGENTS.md) for the full guide index, and [`templates/`](templates/) for ready-to-copy starter files (root `AGENTS.md`, repo-specific guide skeleton, monthly sync workflow).

```bash
# Add the subtree to a new consumer repo
git subtree add --prefix docs/dev-guides \
    https://github.com/CliMA/DeveloperGuides.git main --squash

# Pull the latest guides manually (most repos automate this monthly via update_dev_guides.yml)
git subtree pull --prefix docs/dev-guides \
    https://github.com/CliMA/DeveloperGuides.git main --squash \
    -m "chore: sync dev guides from central repo"
```

> [!NOTE]
> **Subtree pitfalls.**
> 
> - DeveloperGuides ships its own `AGENTS.md`, `LICENSE`, and `README.md` at the repo root, which conflict with the consumer's own root files during `git subtree add`. Resolve by keeping the consumer's versions: `git checkout --ours AGENTS.md LICENSE README.md && git add … && git rebase --continue`.
> - `git subtree pull` exits with an error when there are no new commits upstream. In an automated workflow, append `|| true` so the step does not fail on months with no DeveloperGuides changes.

### Contributing back

Edits to shared guidelines belong here, not in the vendored copy inside a consumer repo. Open PRs against `CliMA/DeveloperGuides`; once merged, the next subtree pull propagates them to every consumer.

## Directory Structure

```text
├── AGENTS.md                  # Master index for AI agents
├── architecture/              # System design, layering, contracts
├── performance/               # GPU, type stability, numerics, AD
├── code-quality/              # Style, docstrings, changelogs
├── infrastructure/            # Testing, device abstraction
├── workflow/                  # Agent autonomy, PR review
└── templates/                 # Starter files for consumer repos
```

## Integration with the CliMA Ecosystem

DeveloperGuides is the central source of truth for engineering standards across [CliMA](https://github.com/CliMA), including:

- [ClimaAtmos](https://github.com/CliMA/ClimaAtmos.jl)
- [ClimaCore](https://github.com/CliMA/ClimaCore.jl)
- [ClimaLand](https://github.com/CliMA/ClimaLand.jl)
- [ClimaOcean](https://github.com/CliMA/ClimaOcean.jl)
- [ClimaCoupler](https://github.com/CliMA/ClimaCoupler.jl)
- [Thermodynamics](https://github.com/CliMA/Thermodynamics.jl)
- [CloudMicrophysics](https://github.com/CliMA/CloudMicrophysics.jl)
- [SurfaceFluxes](https://github.com/CliMA/SurfaceFluxes.jl)
- [ClimaTimeSteppers](https://github.com/CliMA/ClimaTimeSteppers.jl)

## Contributing

- Each guide has a **Self-correction** section: if you discover a guide is stale or missing a pattern, update it directly.
- New guides should be placed in the appropriate category directory and added to [`AGENTS.md`](AGENTS.md).
- Cross-references between guides should use relative paths (e.g., `../performance/gpu_performance.md`).

## Getting Help

For questions or suggestions, open an issue on [GitHub](https://github.com/CliMA/DeveloperGuides/issues).
