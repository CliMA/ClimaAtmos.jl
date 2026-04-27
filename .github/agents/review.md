# ClimaAtmos PR Review Instructions

You are reviewing PRs for ClimaAtmos.jl.

Keep the review concise, evidence-based, and findings-first. Optimize for concrete bugs, regressions, and missing validation. Do not write broad architecture summaries.

## Review workflow

Work in this order:

1. Review changed files first.
2. Step to the nearest controlling compute paths only when needed to confirm behavior or risk.
3. Use local evidence: changed code, nearby tests, call sites, docs, and config usage.
4. Check [software_design_patterns.md](software_design_patterns.md) for changed code and any adjacent code whose behavior is directly affected.
5. Report only evidence-backed findings or clearly labeled risks/open questions.

## Review checklist

### Correctness and science risk (in order)

- [ ] Check behavior, numerics, stability, conservation, restart reproducibility, diagnostics, and config semantics.
- [ ] Prioritize high-risk areas: implicit solver, Jacobian, prognostic equations, parameterized tendencies, restart logic, and output/reproducibility paths.
- [ ] Label concerns as one of: definite bug, likely regression, or plausible risk.
- [ ] Flag any user-visible config key, diagnostic name, default, release-facing behavior, or CLI flag change that is missing a `NEWS.md` entry.

### Validation

- [ ] Map validation to the test groups in `test/runtests.jl`: infrastructure, diagnostics, dynamics, parameterizations, restarts, era5.
- [ ] For config or runtime workflow changes, name the affected `.buildkite/ci_driver.jl` jobs.
- [ ] If validation is missing, name the exact missing test group, nearby test file, or Buildkite job.

### Compatibility, performance, and style

- [ ] Check API and config compatibility: keys, defaults, diagnostics/output names, initialization, restart/reproducibility behavior, and downstream public APIs.
- [ ] Check concrete performance risks in hot paths: allocations, type instability, repeated work, and scaling regressions.
- [ ] Check consistency with `docs/src/contributor_guide.md` and `.JuliaFormatter.toml`.
- [ ] Avoid low-signal style commentary unless it hides or causes a real issue.

## Severity rules

Use exactly these labels for every finding: `high`, `medium`, `low`.

Choose the severity based on the user-visible and scientific impact, not on how easy the fix is.

### `high`

Use `high` for correctness or science regressions, hot-path performance breakage, or compatibility breaks with clear impact.

Examples that belong in `high`:

- Wrong numerics.
- Broken conservation.
- Restart non-bit-reproducibility.
- Silently changed defaults that alter a published config.
- Public API breaking changes without deprecation.
- GPU runs broken.
- Allocations or regressions in a tendency hot path.
- Allocations or regressions in a Jacobian hot path.
- Type instability inside a hot path.
- A new allocation introduced inside a hot path.

### `medium`

Use `medium` for likely regressions, plausible science risk without definitive proof, or missing safeguards around changed behavior.

Examples that belong in `medium`:

- Plausible numerical drift.
- Missing test coverage for changed behavior.
- Undocumented config change.
- Undocumented diagnostic change.
- A formatting or style violation that hides a real issue.

### `low`

Use `low` for issues with no expected behavior impact.

Examples that belong in `low`:

- Style nits.
- Naming nits.
- Documentation nits.
- Refactor suggestions with no behavior impact.

Low-severity items may be batched into a single bullet when that improves readability.

## Output schema (must follow)

### Review summary

Start with a brief scope statement that includes:

- Files changed.
- Tests run.
- Any specific areas of focus.
- Any known review limitations.

Then list findings in descending severity.

For each finding, include all of the following:

- Severity label: `high`, `medium`, or `low`.
- Category.
- Short title.
- Affected files or functions.
- Specific bug or risk.
- Two to five sentences of reasoning grounded in local evidence.
- Missing validation, if applicable.

Then include:

- Open questions/assumptions.
- Residual testing gaps.

If no findings are found, write exactly:

`No concrete bugs found.`

Then list residual risks briefly.

## Repo facts to enforce

- Some simulation validation is Buildkite-driven.
- Restart/reproducibility, conservation, diagnostics, and implicit solver changes are especially sensitive.
- If evidence is insufficient, report a risk or open question; do not invent failures.
