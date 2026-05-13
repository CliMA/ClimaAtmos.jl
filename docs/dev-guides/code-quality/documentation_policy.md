# Software Documentation Policy

This guide defines the standards for repository-level documentation and docstrings across CliMA repositories.

## 1. Goal and Purpose

CliMA is committed to producing high-quality, well-documented software. **Our goals are to foster shared code ownership and to prevent siloed knowledge**.

Documentation should focus on explaining the **design, purpose, and behavior** of code. It should be embedded in the code and consist of the "minimally viable documentation" that allows a technically versed programmer who is not an expert in the subject matter to understand and use it.

- **Do not** document mechanical implementation details ("What a code does in detail should be as self-explanatory as possible").
- **Do** document interfaces, expected behavior, and provide short examples.

## 2. Repository Documentation

All repositories must include the following high-priority documentation sections (typically in `docs/src/` or `README.md`):

1. **Home**: Briefly describe the repository and include links to important subcomponents.
2. **Examples**: Simple examples showing main uses.
3. **API reference**: Interface concepts, purpose, and function signatures.
4. **Contribution guidelines**: How to contribute (PRs, style guide, CI).

### Organizing documentation by user need

Good documentation serves distinct user needs. The [Diátaxis](https://diataxis.fr/) framework identifies four:

- **Tutorials / walkthroughs** — learning-oriented material that guides a newcomer through a meaningful exercise. The reader should *do* something and gain confidence. State the goal up front ("In this tutorial we will compute saturation-adjusted profiles for a moist atmosphere"), deliver visible results at every step, and minimize theoretical digressions. In practice, tutorials often interleave brief explanations of the underlying physics — this is fine, as long as the doing remains the spine of the narrative. Test tutorials in CI (e.g., via Literate.jl) so they never silently break.
- **How-to guides** — task-oriented directions for someone who already knows what they want to achieve. Title them as verb phrases ("How to add a new parameterization," "How to run on GPU," not "GPU support"). Focus on action, not theory; link to explanatory pages when background is needed. A how-to guide that only works for one narrow case is rarely useful — show how to adapt the approach.
- **Reference** — information-oriented material (API docs, configuration options, data formats). Keep entries structured consistently. Documenter's [`@autodocs`](https://documenter.juliadocs.org/stable/man/syntax/#@autodocs-block) blocks can generate reference pages from docstrings and are convenient for fast-moving internal modules, configuration dumps, and other "just show me everything" surfaces. Prefer **manual `@docs` curation** for the public API page when you want to control symbol ordering, separate public from internal API, add prose between groups of related functions, or keep stable URL anchors as the codebase evolves.
- **Explanation** — understanding-oriented discussion: derivations, design rationale, trade-offs. This is the right place for mathematical formulations and theory.

These categories are a guide, not a rigid partition. In CliMA repos, theory and worked examples are often interleaved; for instance, Thermodynamics.jl pairs a *Mathematical Formulation* page with a *How-To Guide* and *Temperature Profiles* walkthrough, while SurfaceFluxes.jl blends *Surface Fluxes Theory*, *Universal Functions*, and *Physical Scales* pages alongside its *API Reference*. What matters is that each page has a clear primary purpose and that the reader can quickly find what they need.

### Tools

- Use [Documenter.jl](https://juliadocs.github.io/Documenter.jl/stable/) for rendering docstrings on documentation pages.
- Use [Literate.jl](https://fredrikekre.github.io/Literate.jl/stable/) to generate markdown and Jupyter-notebook-style examples. Literate.jl scripts are ideal for tutorials because they can be tested in CI.
- Documentation sources live in `docs/src/`; tutorials in `tutorials/` (if present).

### Licensing

All repositories must include a `NOTICE` file and a `LICENSE` file (Apache License 2.0) in the repository root.

## 3. Docstring Standard

Every docstring lives next to the code it describes and is the first thing a future reader (human or AI) will look at. Docstrings should be useful, not decorative — they exist so the next reader can understand and safely modify the code without re-deriving its purpose.

This section follows [Julia's documentation conventions](https://docs.julialang.org/en/v1/manual/documentation/) and uses [Documenter.jl](https://documenter.juliadocs.org/) features for rendering. Anything that is repository- or framework-specific is called out as such.

### 3.1 Calibrate depth to the function's role

Not every function needs the same level of detail. Match the docstring to where the function sits in the codebase. **The more central or widely-called a function is, the more documentation it earns** — extra explanation, more thorough I/O coverage, and `See also` pointers to dependent code.

#### User-facing API

Anything exported, registered on a `docs/src/api.md` page, or that users construct directly (model setups, parameter holders, top-level entry points) is part of the public surface. **Always document inputs and outputs explicitly:**

- `# Arguments` and `# Keyword Arguments` for every parameter, with types and units.
- `# Returns` describing structure and units of the result.
- `# Examples` — at least one runnable snippet.
- Citations to source papers via `[Key](@cite)` (see §3.4).
- ``See also [`related_function`](@ref)`` linking to neighbours in the same public surface.

#### Hot-path / heavily-used internal functions

Tendency functions, cache builders, core precomputed quantities, Jacobian routines, and fundamental physics helpers are read often and edited carefully. Treat them like public API, and **add more:**

- **Extra explanation** of the algorithm, numerical considerations, or sign conventions — enough that a reader can verify correctness without re-deriving the math.
- **Side effects spelled out:** which fields of `Y`, `Yₜ`, `p` (or analogous state) are read, mutated, or assumed pre-populated by an earlier stage of the timestep.
- **`See also` pointers to dependent code:** the major callers and related helpers it composes with. The more places a function is used, the more important these breadcrumbs become for the next reader.

#### Internal helpers with one or two call sites

A short helper extracted for readability does not need exhaustive I/O documentation. A one-line summary is enough, **but it should still reference its caller** so a reader landing on the helper can navigate back up:

~~~julia
"""
    e_tot_0M_precipitation_sources_helper(thp, T, q_liq, q_ice, Φ)

Compute the specific energy carried away by precipitation in the 0-moment scheme.

Called from [`Microphysics0MEvaluator`](@ref) and [`microphysics_tendencies_0m`](@ref).
"""
~~~

Mention any non-obvious assumption the helper relies on (e.g. "assumes `q_tot ≥ 0`"), but skip the `# Arguments` section unless the names or units are not obvious.

#### Trivial private helpers

For one- or two-line functions with self-documenting names (`_clamp_positive`, anonymous closures, single-use inline functions), a docstring is optional. If you skip it, a brief comment above the definition explaining *why* the helper exists is welcome.

### 3.2 Anatomy of a docstring

Every docstring shares the same skeleton: an indented signature, a blank line, a one-line summary in imperative mood, and (optionally) more detail.

~~~julia
"""
    function_name(arg1, arg2; kwarg = default)

One-line summary in imperative mood ("Compute X", "Return Y", "Build Z").

Optional longer prose. Wrap at ~92 chars.
"""
function function_name(arg1, arg2; kwarg = default)
    ...
end
~~~

Rules that apply to every docstring:

- **Signature line is indented 4 spaces** and lives on the first line after `"""`. This is what Julia's `?function_name` and Documenter render as the call signature; do not omit it. For functions with complex dispatch or many arguments, the signature is even more important — it is the first thing the reader sees.
- **One-line summary** follows the signature after a blank line. Imperative mood ("Compute the buoyancy ...") not third-person ("Computes the buoyancy ..."). Be consistent within a docstring.
- **Backtick `code`** for variable names, type names, and option strings.
- **Sentences end with periods**, including bullet items.
- **Conciseness:** let names and formulas do the work. A docstring is not a tutorial.

### 3.3 Section headings

Use the following section headings, in this order, with a single `#` (level 1). Only include the sections you actually need.

| Heading | When to include |
|---|---|
| `# Arguments` | Positional arguments of a function (skip if signature is self-explanatory and ≤ 2 args). |
| `# Keyword Arguments` | Keyword arguments. Always document defaults in the bullet, not just in the signature. |
| `# Returns` | Whenever the return value is non-obvious, has structure (`NamedTuple`, multiple values, a `Field` with non-obvious units). |
| `# Fields` | For struct types, to describe each field. See §3.5 for when this is required. |
| `# Constructor` | When an abstract or parametric type has a meaningful outer constructor. |
| `# Examples` | At least one short example for any user-facing function, type, or setup. |
| `# Notes` | Caveats, performance notes, related implementations. |
| `# Extended help` | Optional appendix shown only via `??function_name` in the REPL. See §3.10. |

Use the plural form (`# Arguments`, `# Examples`) — Julia's official convention. Do not mix in `## Example`, `## Inputs`, `Arguments:` (with a colon), or other variants when writing new docs; fix those when you encounter them.

#### Argument and field bullet format

```
- `name`: One-line description. Units in square brackets at the end, e.g. [kg/m³].
  Continuation lines indented two spaces under the bullet.
```

- Each bullet starts with the backticked identifier.
- Default values for keyword arguments go in the description: ``` `z_elem = 10`: number of vertical points. ```
- For complex options, list valid values as a nested bullet list.

### 3.4 Units, math, and references

#### Units

Atmospheric and physics code is full of dimensional quantities; units carry semantic content. Always include units where they apply.

- **Use SI** unless the underlying library exposes another unit (in which case match it and say so).
- **Square brackets at the end of the bullet/description:** `[K]`, `[kg/m³]`, `[m/s²]`, `[J/kg]`, `[W/m²]`, `[kg/kg]` for specific humidities, `[s]` for time, `[m]` for length.
- **Dimensionless** quantities: `[-]` or `[dimensionless]` if it could be mistaken for unitful.
- **Be consistent within a docstring** — if `Y` is `[kg/m³]` in one bullet, it is `[kg/m³]` everywhere.
- Do not put `(...)` immediately after `[...]` — Documenter parses `[text](text)` as a markdown link and will error. See §4.

#### Math

Documenter renders math with [KaTeX](https://katex.org/).

- **Prefer Unicode for simple expressions.** KaTeX handles Greek letters (α, β, χ, ρ, …), differential operators (`∂`, `∇`), comparison and set operators (`≤`, `∈`, `∪`), arrows, and basic operators inline. Unicode is far easier to read in source than the equivalent LaTeX commands, and matches the variable naming used in the code.
- **Use LaTeX for complex layout** — fractions, integrals with bounds, multi-line alignment, large operators, super/subscripts on Unicode bases that KaTeX won't render cleanly.
- **Inline math:** double backticks, ``` ``α \cdot β`` ``` or ``` ``α · β`` ```.
- **Display math:** fenced ` ```math ` block.

~~~markdown
```math
\frac{∂χ}{∂t} = -β\, ∇·(∇χ), \quad z > z_d
```
~~~

If a docstring has many backslashes, use `raw"""..."""` so Julia does not try to interpret them as string escapes.

#### Citations

Use Documenter's bibliography integration. The bibliography lives at `docs/src/bibliography.bib` (or the repo equivalent). Cite with:

```markdown
Described in [Smith2020](@cite).
The scheme of [Stevens2005](@cite) is extended by [Ackerman2009](@cite).
```

The key (`Smith2020`) is the BibTeX entry name. **Add the BibTeX entry first** if you cite a paper that isn't in the bibliography — the docs build will fail otherwise.

#### Cross-references

**Every function, type, or method name you mention in a docstring should be linked.** A bare backtick mention like `` `compute_strain_rate_face_full!` `` produces no link and forces the reader to grep — the `@ref` form costs one extra `(@ref)` and gives the reader a click-through.

Use Documenter's `@ref` syntax for in-package references:

```markdown
See also [`compute_strain_rate_face_full!`](@ref) for the face-centered version.
Constructed via [`SphereGrid`](@ref) or [`ColumnGrid`](@ref).
```

The target must be documented and registered on a docs page (or exported from the package).

##### Cross-package references

Use [DocumenterInterLinks.jl](https://juliadocs.org/DocumenterInterLinks.jl/stable/) to link to symbols in other CliMA repositories — or any package whose docs are built with Documenter. Configure the inter-link inventory once in `docs/make.jl`, then cite cross-repo symbols with the `@extref` syntax:

```markdown
Wraps [`Thermodynamics.air_temperature`](@extref).
```

This produces working URLs into the dependency's own documentation. Without DocumenterInterLinks configured, fall back to fully qualified names in backticks (`` `Thermodynamics.ThermodynamicsParameters` ``) so the reader at least sees the package qualifier — but prefer to set up DocumenterInterLinks so the link works.

### 3.5 Documenting structs

**Use a `# Fields` section in the docstring body** to document fields:

~~~julia
"""
    ViscousSponge{FT} <: SpongeModel

Viscous sponge model; damp variables in proportion to the value of their Laplacian.

# Fields
- `zd`: Lower damping height [m].
- `κ₂`: Damping coefficient [m²/s²].
"""
@kwdef struct ViscousSponge{FT} <: SpongeModel
    zd::FT
    κ₂::FT
end
~~~

> **Why `# Fields` and not inline field strings?**
> Julia supports docstrings attached directly to struct fields (a string literal above each field declaration). These are accessible via `Docs.fielddoc(MyStruct, :field)` and shown by `?` in the REPL, but **Documenter does not render them in built documentation pages** unless you also use `DocStringExtensions.@TYPEDFIELDS`. Putting field documentation in a `# Fields` section keeps a single source of truth that renders in both REPL help and built docs.

A `# Fields` section is **required** when:
- the struct is part of the public API (constructed by users),
- it has more than a few fields,
- or any field's meaning, units, or invariants are not obvious from the name and type.

It is optional for marker structs (no fields) and tiny internal helpers with self-evident fields.

#### Type parameters

If a struct is parameterized, explain the type parameters either in prose or with a nested list. Include them in the signature on the docstring's first line:

~~~julia
"""
    SmagorinskyLilly{AXES}

Smagorinsky-Lilly eddy viscosity model.

`AXES` is a symbol indicating along which axes the model is applied. It can be
- `:UVW` (all axes),
- `:UV`  (horizontal axes),
- `:W`   (vertical axis),
- `:UV_W` (horizontal and vertical axes treated separately).

# Examples
```julia
sl = SmagorinskyLilly(; axes = :UV_W)
```
"""
struct SmagorinskyLilly{AXES} <: EddyViscosityModel end
~~~

#### Functors (callable structs)

For callable structs, document the type and its call method separately. The type docstring describes what the struct represents and its fields; the call-method docstring describes the behavior of invoking it:

~~~julia
"""
    MicrophysicsEvaluator

Functor for evaluating microphysics tendencies at a quadrature point.

# Fields
- `cm_params`: cloud microphysics parameters.
- `thermo_params`: thermodynamics parameters.
"""
struct MicrophysicsEvaluator{CMP, TP}
    cm_params::CMP
    thermo_params::TP
end

"""
    (eval::MicrophysicsEvaluator)(T_hat, q_tot_hat)

Evaluate microphysics tendencies at a quadrature point.

# Arguments
- `T_hat`: temperature at quadrature point [K].
- `q_tot_hat`: total specific humidity [kg/kg].
"""
function (eval::MicrophysicsEvaluator)(T_hat, q_tot_hat)
    ...
end
~~~

### 3.6 Documenting abstract types

Abstract types are the entry point for understanding a type hierarchy. The docstring should:

1. State the role of the hierarchy in one sentence.
2. Enumerate concrete subtypes with a one-line description of each.
3. Document the outer constructor if there is one (`# Constructor` section).
4. Note any interface methods subtypes must implement.

~~~julia
"""
    CloudModel

Strategy for computing the cloud fraction.

Subtypes:
- [`GridScaleCloud`](@ref): cloud fraction based on grid-mean conditions.
- [`QuadratureCloud`](@ref): cloud fraction from an SGS-quadrature integral.
- [`SGSML`](@ref): ML-based diagnostic cloud fraction.
"""
abstract type CloudModel end
~~~

### 3.7 Documenting functions

#### Plain function

~~~julia
"""
    geopotential(grav, z)

Compute the geopotential Φ at height `z` using gravitational acceleration `grav`.

```math
Φ = g · z
```
"""
geopotential(grav, z) = grav * z
~~~

#### Multiple methods

When several methods share a docstring, list each signature on its own indented line:

~~~julia
"""
    microphysics_tendencies_0m(SG_quad, cmp, thp, ρ, T, q_tot, T′T′, q′q′, corr_Tq, Φ, tst, dt)
    microphysics_tendencies_0m(cmp, thp, ρ, T, q_tot, q_liq, q_ice, Φ, tst, dt)

Compute 0-moment microphysics tendencies. The SGS-quadrature method integrates over the
joint PDF of (T, q_tot); the point-value method assumes a single representative state.

# Arguments
- ...
"""
~~~

#### In-place / cache-mutating functions

Functions of the form `f!(Yₜ, Y, p, t, ...)` are pervasive in atmospheric solvers. Document them explicitly:

- State that the function **modifies the first (or specified) argument in place**.
- State the return value (`nothing`, by convention).
- For each model-selector argument, list which concrete subtypes the function dispatches on.
- If the function reads precomputed quantities from the cache, mention which ones — the goal is greppability, not exhaustiveness.

~~~julia
"""
    microphysics_tendency!(Yₜ, Y, p, t, microphysics_model, turbconv_model)

Apply microphysics tendencies, reading `mp_tendency` values precomputed in the cache.

# Arguments
- `Yₜ`: tendency state vector (mutated in place).
- `Y`: current state vector.
- `p`: cache; reads precomputed `mp_tendency` set up earlier in the timestep.
- `t`: current simulation time [s].
- `microphysics_model`: dispatch tag, e.g. `EquilibriumMicrophysics0M`,
  `NonEquilibriumMicrophysics1M`.
- `turbconv_model`: e.g. `PrognosticEDMFX`, `DiagnosticEDMFX`, or `nothing`.

# Returns
`nothing`. Modifies `Yₜ` in place.
"""
~~~

#### Returning structured values

If a function returns a `NamedTuple` or other composite, sketch its shape:

~~~julia
"""
    build_nonneg_borrow_ledger(Y, atmos, FT) -> NamedTuple or nothing

Allocate the per-species, per-subdomain borrow ledger consumed by
`tracer_nonnegativity_constraint!`.

# Returns
`nothing` if no `tracer_nonnegativity_method` is configured, otherwise a NamedTuple:

    (gridmean = (ρq_lcl = …, ρq_rai = …, …),
     sgsʲs    = ((q_lcl = …, …), …))

Fields are zeroed. Each call to the constraint overwrites them with `Y_post − Y_pre`.
"""
~~~

The `-> ReturnType` annotation in the signature line is useful when the return shape is the most important thing the caller needs to know.

### 3.8 Examples

Include at least one example for any user-facing API. Examples for purely internal helpers are optional but welcome.

~~~markdown
# Examples
```julia
sponge = ViscousSponge(Float32; zd = 20_000, κ₂ = 1e6)
```
~~~

- Use a plain fenced ` ```julia ` block.
- Examples should be runnable in a fresh REPL after `using YourPackage`. Spell out any non-trivial setup (parameter creation, thermo params) so a reader can reproduce.
- One short example is better than three sprawling ones.

### 3.9 Admonitions

Use [Documenter admonitions](https://documenter.juliadocs.org/stable/showcase/#Admonitions) sparingly, for genuinely important caveats. Available kinds: `note`, `warning`, `tip`, `info`, `compat`, `danger`.

```markdown
!!! warning
    Calling this function outside `set_precomputed_quantities!` will read stale `p.scratch`
    values. Run after `set_implicit_precomputed_quantities_part1!`.
```

- Keep admonitions to 1–3 sentences.
- Title them (`!!! note "Title"`) only when the docstring has multiple admonitions or the title adds context.
- If you find yourself writing more than two admonitions in one docstring, the docstring is doing too much — split it or move content into the documentation under `docs/src/`.

### 3.10 `# Extended help`

Documenter and the Julia REPL recognize `# Extended help` as a special section: its content is shown only when the user passes `??function_name`, while `?function_name` shows everything before it. Use it for:

- Long worked-out boundary cases that would dilute the main docstring.
- Implementation notes for maintainers (rather than callers).
- Detailed `See also` lists for cross-cutting helpers.

~~~julia
"""
    saturation_adjustment(thp, h, ρ, q_tot)

Compute the saturation-adjusted thermodynamic state for given enthalpy, density, and total
water specific humidity.

# Arguments
- ...

# Extended help

The iteration uses Newton's method with bracketed bisection fallback. Convergence
tolerances and edge-case handling are documented in [`SaturationAdjustmentSolver`](@ref).
...
"""
~~~

`# Extended help` should appear last in the docstring. Long mathematical derivations belong on a *Mathematical Formulation* page in `docs/src/`, not in extended help — see §3.11.

### 3.11 What we don't use

- **`DocStringExtensions`** (`$(TYPEDEF)`, `$(FIELDS)`, `$(SIGNATURES)`, etc.). Spell out signatures, field lists, and types by hand. Readability in the source file matters more than DRY here.
- **`jldoctest`** blocks. Most CliMA repos do not run doctests in CI; using `jldoctest` causes silent rot. Use plain ` ```julia ` blocks instead.
- **Long mathematical derivations inline** in docstrings. The docstring should give the reader enough to use the function correctly; the derivation belongs on an *Explanation* page in `docs/src/` (see §2) or on the source paper. Link to it from the docstring.
- **Generated docstrings via metaprogramming** (macros that splice docstrings into `@eval`'d definitions) unless absolutely necessary — they are hard to grep, hard to read, and easy to break.

### 3.12 Verifying your docstring

After writing or editing a docstring, do at least one of:

- **In the REPL** — `?your_thing` shows everything up to (but not including) `# Extended help`; `??your_thing` shows the full docstring including the extended-help section. From the shell:
  ~~~sh
  julia --project -e 'using YourPackage; @doc YourPackage.your_thing'
  ~~~

- **Build the docs locally:**
  ~~~sh
  julia --project=docs/ -e 'using Pkg; Pkg.instantiate(); Pkg.develop(PackageSpec(path=pwd()))'
  JULIA_DEBUG=Documenter julia --project=docs/ docs/make.jl
  ~~~
  Then open `docs/build/index.html`. Warnings about broken `@ref` / `@cite` mean cross-references are wrong; fix them before merging.

For new exported symbols, also add an entry to the appropriate page under `docs/src/` (typically `api.md`); otherwise the docstring will not be rendered on the API page (it will still be discoverable via `?` in the REPL).

### 3.13 Anti-patterns

- **Missing signature line.** Writing `"""Compute X..."""` with no indented signature breaks REPL help and Documenter rendering.
- **Restating the obvious.** `"""Return the input."""` for `identity(x) = x` is noise.
- **Documenting *how* instead of *why*.** The implementation is right below the docstring; callers want to know what to pass in, what they get back, and what assumptions are baked in.
- **Out-of-date signatures.** When renaming an argument or changing defaults, update the signature line *and* the `# Arguments` bullets. CI does not catch divergence here.
- **Inconsistent units.** If `Y` is `[kg/m³]` in one bullet and `[g/cm³]` in another, the docstring is worse than no docstring.
- **`Arguments:`, `Inputs:`, `Returns:` as plain prose lines** instead of `# Arguments`, `# Returns`. Documenter renders the section-heading form; the prose form is invisible structure.
- **Mixing third-person and imperative within the same docstring.** Pick one.
- **Multi-paragraph docstrings on internal helpers.** If you are writing more than ~15 lines to document a helper that is called from one place, ask whether the prose belongs in a block comment near the call site or in `docs/src/`.

### 3.14 Quick reference (cheat sheet)

For functions:

~~~julia
"""
    function_name(positional1, positional2; kwarg1 = default1)

One-line imperative summary.

Optional 1–3 paragraph elaboration. Math:

```math
y = f(x)
```

# Arguments
- `positional1`: description [units].
- `positional2`: description [units].

# Keyword Arguments
- `kwarg1 = default1`: description [units].

# Returns
Description of return value, with units / structure if non-obvious.

# Examples
```julia
result = function_name(a, b; kwarg1 = c)
```

# Notes
- Caveats, performance notes, related implementations.

See also [`related_function`](@ref). Described in [Smith2020](@cite).
"""
function function_name(positional1, positional2; kwarg1 = default1)
    ...
end
~~~

For structs:

~~~julia
"""
    MyStruct{T}

One-line summary. Longer description, including what `T` parameterizes.

# Fields
- `field`: description [units].

# Examples
```julia
s = MyStruct{Float64}(; field = 1.0)
```
"""
@kwdef struct MyStruct{T}
    field::T
end
~~~

For abstract types:

~~~julia
"""
    Foo

Strategy for doing foo.

Subtypes:
- [`ConcreteFooA`](@ref): one-line description.
- [`ConcreteFooB`](@ref): one-line description.
"""
abstract type Foo end
~~~

## 4. Documenter.jl pitfalls

### Markdown link ambiguity

Be careful with `[kg/m^3](description)` formats in docstrings. Documenter's markdown parser interprets `[text](text)` as a link and will produce `:cross_references` errors if the parenthetical text is not a URL.

**Fix**: use parentheses for units — `(kg/m^3)` — or separate brackets and parentheses with punctuation or a line break.

Do **not** attempt to escape square brackets with backslashes (`\[...\]`) in Julia string literals; this causes invalid escape sequence errors during precompilation.

### Missing docstrings

If `makedocs` fails with "Missing docstrings" errors, ensure every exported symbol with a docstring is included in a documentation page via an `@docs` or `@autodocs` block.

### Undefined symbols

Use fully qualified names in docstrings (for example, `Thermodynamics.ThermodynamicsParameters`) to ensure Documenter's link generator can resolve them across package boundaries.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
