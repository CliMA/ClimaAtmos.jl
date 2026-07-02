# PR Review Instructions

You are reviewing PRs for a CliMA package.

The output stays concise, evidence-based, and findings-first: concrete bugs, regressions, and missing validation, ordered by impact. Do not write broad architecture summaries. Concise output does not mean shallow analysis, though. Every review must do the deep work below, especially the mathematical, physical, and performance-guideline checks, even though only the resulting findings appear in the report.

## Review workflow

Two passes are mandatory on every PR, no matter how small or how "obviously correct" the change looks. A review that skips either one is incomplete:

1. **Mathematical and physical correctness and consistency** (section 1). Check every mathematical relation in the changed code and in its documentation, and confirm that code and docs agree.
2. **Conformance to the performance guides** (section 2). Check the changed hot-path and kernel code against the specific written performance rules, not by general impression.

Work in this order:

1. Read the changed files first, then step to the nearest controlling compute paths only when needed to confirm behavior or risk.
2. Run the two mandatory passes: mathematical and physical correctness and consistency (section 1), then conformance to the performance guides (section 2).
3. Use local evidence: changed code, nearby tests, call sites, docs, and config usage. Where a mathematical or codegen claim is in doubt, generate the evidence: derive the relation, check equivalence numerically, or inspect `@code_typed`/`@code_llvm`/`@allocated`.
4. Report only evidence-backed findings or clearly labeled risks and open questions.

## 1. Mathematical and physical correctness and consistency (mandatory, in depth)

This is the heart of a CliMA review. Code can be clean, allocation-free, and GPU-safe and still be scientifically wrong. Do all of the following.

### Verify every mathematical relation

- [ ] Check **every** formula, algebraic identity, exponent, coefficient, unit, and physical constant in the changed code against its source: the cited paper or textbook, the governing equation, or the pre-change code. Re-derive it; do not assume a refactor preserved the math.
- [ ] For each rewritten expression (a performance or clarity refactor: an exponent rewrite, a precomputed coefficient, a log-space reformulation, a fused `muladd`, a factored constant), **prove equivalence to the original** rather than eyeballing it. Confirm bit-identity with `===`, or agreement within a stated tolerance, across both `Float32` and `Float64`, on representative and extreme inputs. Recent example: `x^(2//3) -> cbrt(x)^2`, and `a_vent_0_coeff = av*cbrt(36)` replacing the inline `av / 6^(-2/3)`, were each confirmed identical before the change was accepted.
- [ ] Watch for equivalence that holds algebraically but fails numerically: a reordering that overflows or underflows an intermediate (`cbrt(x^2)` vs `cbrt(x)^2`), catastrophic cancellation, a dropped `max(x, floor)` guard, or a broken sign assumption (`cbrt` accepts negatives, `sqrt` does not). See [numerical_robustness.md](../performance/numerical_robustness.md).
- [ ] Check every **citation to a paper or reference**. Confirm the reference resolves (DOI, author-year, or equation/table/section number) and, more importantly, that it actually **supports the specific point being made**: the equation as written, the numerical value of a coefficient or threshold, the functional form of a parameterization, or the range of validity. Read the cited passage, do not trust the label. Flag a citation that supports something weaker or different than the code claims, an equation or table number that does not match the cited edition, a constant whose value disagrees with the source, and any new physics, empirical formula, or parameter value introduced without a citation. Keep citations consistent between code and docs (see [documentation_policy.md §3.4](../code-quality/documentation_policy.md)).

### Keep documentation consistent with code

- [ ] Treat docstrings, code comments, and LaTeX/Documenter math as part of the artifact under review. Every relation they state must match the code **exactly**.
- [ ] Flag as findings: a docstring formula that no longer matches the implementation, a comment describing a previous implementation, a field docstring whose "pre-computed ..." description disagrees with the constructor, an `@ref`/`@docs` pointing at a renamed or removed symbol, and units or symbol definitions that have drifted from the code. A doc/code inconsistency is a finding even when the code itself is correct. See [documentation_policy.md](../code-quality/documentation_policy.md).

### Physical reasoning

- [ ] Check conservation (mass, energy, number), sign conventions, dimensional and unit consistency, monotonicity where the physics requires it, and boundedness of rates and states.
- [ ] Check physical limits explicitly: zero input yields zero output, thresholds (freezing/melting, saturation) switch at the right point and in the right direction, rates are non-negative where required, and quantities stay finite at the edges of their domain. Cross-check against the limit, round-trip, and analytic tests catalogued in [testing_and_validation.md](../infrastructure/testing_and_validation.md).

### Accuracy of approximations

- [ ] For a changed quadrature rule, series truncation, root-finder, iteration count, or table lookup, confirm accuracy is preserved with an **offline convergence or error study** against a high-order or analytic reference, not with "the tests still pass." A reduced node count or iteration count needs that evidence.
- [ ] Watch for integrand or integrand-derivative non-smoothness (a cusp or kink, for example at `|v1 - v2| = 0`) that caps the convergence order. A node count tuned for a smooth integrand silently loses accuracy on a non-smooth one.

## 2. Conformance to the performance guides (mandatory)

Check the changed hot-path and kernel code (both defined in [gpu_performance.md](../performance/gpu_performance.md)) against the specific rules in the performance guides. This is a checklist against written guidance, not a general impression.

- [ ] **Thread divergence**: data-dependent `if/else`, `&&`/`||` on per-point conditions, convergence loops (`while err > tol`), and early `break` on a per-grid-point path. Prefer `ifelse` and fixed iteration counts. See [branchless_code.md](../performance/branchless_code.md), [gpu_performance.md §1 and §6](../performance/gpu_performance.md), and SDP 17 and 19.
- [ ] **Elementary-function cost**: `pow` from a floating-point exponent, rational-literal exponents (`x^(2//3)`, which construct a `Rational{Int64}` and run a 64-bit `gcd` per thread on the GPU), float-literal exponents (`x^2.0`), and repeated transcendentals inside an integrand. Fractional powers should use `sqrt`/`cbrt` with integer-literal powers, and constant powers should be precomputed once on the host. See [gpu_performance.md §10](../performance/gpu_performance.md).
- [ ] **Numerical robustness in kernels**: `log`, `sqrt`, and division guarded *before* an `ifelse` (both arms always evaluate), with the correct floor (not a blind `eps(FT)`). See [numerical_robustness.md](../performance/numerical_robustness.md) and [gpu_performance.md §1](../performance/gpu_performance.md).
- [ ] **Type stability and Float32**: unwrapped `Float64` literals (`1.0`, `Inf`), integers or `Rational`s in a Float32 path, and any promotion that pulls a Float32 kernel into Float64. See [type_stability.md](../performance/type_stability.md) and SDP 15 and 16.
- [ ] **Allocations and kernel safety**: a new `Field` allocated inside a tendency or cache setter, a non-`isbits`-after-adapt kernel argument, a closure where a functor is required (flag closures passed to `integrate`, `quadrature`, or `bycolumn` calls), keyword arguments inside a kernel, and `Dict`/`String`/`@assert` in kernels. See [allocation_debugging.md](../performance/allocation_debugging.md), [gpu_performance.md §2 and §8](../performance/gpu_performance.md), and SDP 11, 13, 18, 20, 21.
- [ ] **AD compatibility**: a new `where {FT}` on a non-constructor physics function, or a floating-point type taken from a `where` clause instead of inferred from values. See [ad_compatibility.md](../performance/ad_compatibility.md) and SDP 14 and 15.
- [ ] **Confirm claimed optimizations with evidence.** When a PR claims a codegen property (no `pow`, no `Rational`, no allocation, type stable, faster), verify it: `@code_typed`/`@code_llvm`/`@code_warntype` for the claim, `@allocated` for allocations, and run the affected benchmark or test. A claim in the PR description is not evidence.

## 3. Validation

- [ ] Map validation to the test groups in the repo's `test/runtests.jl`. The repo-specific guide (linked from [AGENTS.md](../AGENTS.md)) lists the groups and example jobs.
- [ ] For any changed mathematical relation or physical behavior, confirm a test exists that would actually catch a wrong value: a limit test, a round-trip, or a comparison to an analytic result ([testing_and_validation.md](../infrastructure/testing_and_validation.md)). If a refactor is claimed "bit-identical," check whether an existing test truly pins the value or merely exercises the code path.
- [ ] For config or runtime workflow changes, name the affected CI driver and Buildkite/GitHub Actions jobs explicitly.
- [ ] Allocation benchmarks (typically under `perf/`) are not run in CI; catch allocation regressions in review with the `@allocated == 0` pattern from [allocation_debugging.md](../performance/allocation_debugging.md).
- [ ] If validation is missing, name the exact missing test group, nearby test file, or CI job.

## 4. Compatibility and versioning

- [ ] Check API and config compatibility: config keys, defaults, diagnostic and output names, initialization, restart/reproducibility behavior, and downstream public APIs.
- [ ] Flag any user-visible config key, diagnostic name, default, release-facing behavior, or CLI flag change that is missing a `NEWS.md` entry. See [changelogs_and_versions.md](../code-quality/changelogs_and_versions.md).

## 5. Consistency and style

- [ ] Check consistency with [software_design_patterns.md](../code-quality/software_design_patterns.md) for the changed code and any adjacent code whose behavior it affects.
- [ ] Flag misleading or colliding names: a variable or field that reuses an established name in the package for an unrelated quantity (a recent review caught a ventilation exponent field named `β_va`, the established name of the mass-dimension exponent). Names must not mislead a future reader.
- [ ] Check formatting against the repo's `.JuliaFormatter.toml` and contributor guide ([code_style.md](../code-quality/code_style.md)).
- [ ] Avoid low-signal style commentary unless it hides or causes a real issue.

## Output schema (must follow)

### Review summary

Start with a brief scope statement:

- Files changed.
- Tests run, and their result.
- Areas of focus, and any review limitations.

Then list findings ordered by impact, most consequential first. For each finding, include:

- A short title and a category (for example: correctness, physics, docs/code consistency, performance, validation, compatibility, style).
- A confidence classification: **definite bug**, **likely regression**, or **plausible risk / open question**. Base this on evidence and scientific impact, not on how easy the fix is.
- Affected files or functions.
- The specific bug or risk.
- Two to five sentences of reasoning grounded in local evidence: the derivation, the equivalence check, or the guideline violated.
- Missing validation, if applicable.

Then include open questions and assumptions, and residual testing gaps.

If no findings are found, write exactly:

`No concrete bugs found.`

Then list residual risks briefly.

## Repo facts to enforce

- Some simulation validation is Buildkite-driven; the repo-specific guide names the canonical CI driver.
- Restart/reproducibility, conservation, diagnostics, and implicit-solver changes are especially sensitive.
- Allocation benchmarks (typically under `perf/`) are not run in CI; allocation regressions must be caught in review with the `@allocated` pattern.
- If evidence is insufficient, report a risk or open question; do not invent failures.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
