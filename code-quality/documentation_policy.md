# Software Documentation Policy

Standards for repository-level documentation and docstrings across CliMA repositories.

## 1. Goal

CliMA is committed to producing high-quality, well-documented software so that knowledge is shared and not siloed. Documentation should explain the **design, purpose, and behavior** of code — not its mechanical implementation. Aim for "minimally viable documentation": enough for a technically capable reader who is not a subject-matter expert to understand and use the code.

- **Do** document interfaces, expected behavior, and short examples.
- **Do not** narrate what the code does line by line — the code itself should be self-explanatory.

## 2. Repository documentation

Every repository must include the following pages (typically under `docs/src/` or in `README.md`):

1. **Home** — a brief description with links to important subcomponents.
2. **Examples** — simple runnable examples covering the main uses.
3. **API reference** — interface concepts, purpose, and function signatures.
4. **Contribution guidelines** — how to contribute (PRs, style, CI).

All repositories must also include a `LICENSE` file (Apache 2.0) and a `NOTICE` file in the repository root.

### 2.1 Organizing documentation by user need

The [Diátaxis](https://diataxis.fr/) framework distinguishes four documentation modes; each page should have a clear primary mode:

- **Tutorials** — learning-oriented walkthroughs. State the goal up front, deliver visible results at every step, and minimize digressions. Test tutorials in CI (e.g. via Literate.jl) so they cannot silently break.
- **How-to guides** — task-oriented directions for someone who already knows what they want. Title as verb phrases ("How to add a parameterization"), not nouns ("Parameterizations").
- **Reference** — API docs, configuration options, data formats. Documenter's [`@autodocs`](https://documenter.juliadocs.org/stable/man/syntax/#@autodocs-block) blocks are convenient for fast-moving internal modules; prefer **manual `@docs` curation** for the public API so you control symbol ordering, separation of public from internal, and stable URL anchors.
- **Explanation** — derivations, design rationale, trade-offs. This is the right place for mathematical formulations and theory.

These modes are guides, not strict partitions: CliMA repos commonly interleave theory and worked examples (e.g. Thermodynamics.jl pairs a *Mathematical Formulation* page with a *How-To Guide*). What matters is that each page has one primary purpose and the reader can find what they need.

### 2.2 Tools

- [Documenter.jl](https://juliadocs.github.io/Documenter.jl/stable/) renders docstrings into documentation pages.
- [Literate.jl](https://fredrikekre.github.io/Literate.jl/stable/) generates markdown and notebook-style examples from Julia scripts and runs them in CI.
- Sources live in `docs/src/`; tutorials in `tutorials/` if present.
- For local iteration, use `LiveServer.servedocs()` — see [onboarding.md §6](../workflow/onboarding.md).

## 3. Docstrings

Every docstring lives next to the code it describes and is the first thing a future reader (human or AI) sees. Docstrings should be useful, not decorative.

This section follows [Julia's documentation conventions](https://docs.julialang.org/en/v1/manual/documentation/) and uses [Documenter.jl](https://documenter.juliadocs.org/) features for rendering.

### 3.1 Calibrate depth to the function's role

The more central or widely-called a function is, the more documentation it earns:

| Role                                                                                                 | Minimum content                                                                                                                                                                                                            |
|:-----------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **User-facing API** (exported, on `api.md`, constructed by users)                                    | Signature, summary, `# Arguments`, `# Returns` with units, at least one `# Examples` block, citations via `[Key](@cite)` where applicable, ``See also [`neighbour`](@ref)``.                                               |
| **Hot-path internals** (tendency functions, cache builders, Jacobian routines, core physics helpers) | Treat as public API and add: algorithmic explanation, sign conventions, side effects spelled out (which `Y`/`Yₜ`/`p` fields are read or mutated), `See also` to major callers.                                             |
| **Helpers with one or two call sites**                                                               | One-line summary plus a back-pointer to the caller (``Called from [`caller`](@ref)``). Skip `# Arguments` unless names/units are ambiguous.                                                                                |
| **Trivial private helpers** (e.g. `_clamp_positive`)                                                 | Docstring optional; a short comment explaining *why* the helper exists is welcome.                                                                                                                                         |

### 3.2 Anatomy of a docstring

Every docstring shares the same skeleton: an indented signature, a blank line, a one-line summary in imperative mood, and optionally more detail.

~~~julia
"""
    function_name(arg1, arg2; kwarg = default)

One-line summary in imperative mood ("Compute X", "Return Y").

Optional longer prose. Wrap at ~92 chars.
"""
function function_name(arg1, arg2; kwarg = default)
    ...
end
~~~

Universal rules:

- **Signature line is indented 4 spaces** on the first line after `"""`. Julia's `?` and Documenter render this as the call signature; do not omit it.
- **One-line summary** follows after a blank line, in imperative mood ("Compute…", not "Computes…").
- **Backtick `code`** for variable names, type names, and option strings.
- **Sentences end with periods**, including in bullet items.
- **Be concise.** Names and formulas do the work — a docstring is not a tutorial.

### 3.3 Section headings

Use these headings, in this order, with a single `#`. Include only what you need.

| Heading                | When to include                                                                                               |
|:-----------------------|:--------------------------------------------------------------------------------------------------------------|
| `# Arguments`          | Positional arguments. Skip if the signature is self-explanatory and there are ≤ 2 args.                       |
| `# Keyword Arguments`  | Keyword arguments. Document defaults in the bullet, not just in the signature.                                |
| `# Returns`            | When the return value is non-obvious or has structure (`NamedTuple`, multiple values, a `Field` with non-obvious units). |
| `# Fields`             | For struct types — see §3.5.                                                                                  |
| `# Constructor`        | When an abstract or parametric type has a meaningful outer constructor.                                       |
| `# Examples`           | At least one short example for any user-facing function, type, or setup.                                      |
| `# Notes`              | Caveats, performance notes.                                                                                   |
| `# Extended help`      | Optional appendix shown only via `??function_name` — see §3.7.                                                |

Use the plural form (`# Arguments`, `# Examples`) — Julia's official convention. Fix variants (`## Example`, `Arguments:`) when you encounter them.

**Argument and field bullet format:**

~~~text
- `name`: One-line description. Units in square brackets at the end, e.g. [kg/m³].
  Continuation lines indented two spaces under the bullet.
~~~

Each bullet starts with the backticked identifier. For complex options, list valid values as a nested bullet list.

### 3.4 Units, math, references

**Units.** Atmospheric and physics code is dimensional; units carry meaning. Use SI unless the underlying library exposes another unit (then match it and say so). Put units in square brackets at the end of the description: `[K]`, `[kg/m³]`, `[m/s²]`, `[W/m²]`, `[kg/kg]` for specific humidities. Dimensionless quantities: `[-]`. Be consistent within a docstring. Do **not** put `(...)` immediately after `[...]` — Documenter parses `[text](text)` as a markdown link and will error (see §4).

**Math.** Documenter renders math with [KaTeX](https://katex.org/).

- **Prefer Unicode for simple expressions** — α, β, ρ, ∂, ∇, ≤, ∈ all render inline and read more naturally than `\alpha`, `\beta`, etc., matching the variable names in the code.
- **Use LaTeX for complex layout** — fractions, integrals with bounds, multi-line alignment.
- **Inline:** double backticks, ``` ``α · β`` ```. **Display:** fenced ` ```math ` block.

~~~markdown
```math
\frac{∂χ}{∂t} = -β\, ∇·(∇χ), \quad z > z_d
```
~~~

If a docstring has many backslashes, use `raw"""..."""` so Julia does not interpret them as escapes.

**Citations.** Cite via Documenter's bibliography integration. The bibliography lives at `docs/src/bibliography.bib`:

~~~markdown
Described in [Smith2020](@cite). The scheme of [Stevens2005](@cite) is extended by [Ackerman2009](@cite).
~~~

Add the BibTeX entry before citing — the docs build fails otherwise.

**Cross-references.** Every function, type, or method name you mention should be linked, not just backticked. The `@ref` form costs one extra `(@ref)` and gives the reader a click-through:

~~~markdown
See also [`compute_strain_rate_face_full!`](@ref) for the face-centered version.
~~~

The target must be documented and registered on a docs page (or exported). For cross-repo references, use [DocumenterInterLinks.jl](https://juliadocs.org/DocumenterInterLinks.jl/stable/) and the `@extref` syntax:

~~~markdown
Wraps [`Thermodynamics.air_temperature`](@extref).
~~~

Without DocumenterInterLinks configured, fall back to fully qualified names in backticks so the reader at least sees the package qualifier — but prefer to set up DocumenterInterLinks.

### 3.5 Structs

Use a `# Fields` section to document fields:

~~~julia
"""
    ViscousSponge{FT} <: SpongeModel

Viscous sponge model; damps variables in proportion to the value of their Laplacian.

# Fields
- `zd`: Lower damping height [m].
- `κ₂`: Damping coefficient [m²/s²].
"""
@kwdef struct ViscousSponge{FT} <: SpongeModel
    zd::FT
    κ₂::FT
end
~~~

> **Why `# Fields` and not inline field docstrings?** Julia supports docstrings attached directly to struct fields, but Documenter does not render them in built documentation pages unless you also use `DocStringExtensions.@TYPEDFIELDS`. A `# Fields` section is a single source of truth for both REPL help and built docs.

A `# Fields` section is **required** when the struct is part of the public API, has more than a few fields, or any field's meaning/units/invariants are not obvious. Optional for marker structs and tiny internal helpers.

If a struct is parameterized, explain the type parameters in prose or a nested list, and include them in the signature line:

~~~julia
"""
    SmagorinskyLilly{AXES}

Smagorinsky-Lilly eddy viscosity model.

`AXES` is a symbol indicating the axes the model is applied along:
- `:UVW` — all axes,
- `:UV`  — horizontal axes only,
- `:W`   — vertical axis only,
- `:UV_W` — horizontal and vertical treated separately.
"""
struct SmagorinskyLilly{AXES} <: EddyViscosityModel end
~~~

For **callable structs** (functors), document the type and its call method separately: the type docstring describes what it represents and its fields; the call-method docstring describes the behavior of invoking it. See [SDP 18](software_design_patterns.md) for the functor pattern itself.

### 3.6 Abstract types

Abstract types are the entry point for understanding a hierarchy. The docstring should:

1. State the role in one sentence.
2. Enumerate concrete subtypes with a one-line description of each.
3. Document the outer constructor if there is one.
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

### 3.7 Mutating functions, multiple methods, structured returns

A few patterns recur often enough to call out:

- **In-place / cache-mutating functions** (`f!(Yₜ, Y, p, t, ...)`): state explicitly that the function mutates its first argument; state the return value (`nothing` by convention); for each model-selector argument list the concrete subtypes it dispatches on; mention which precomputed quantities it reads from the cache.
- **Multiple methods sharing a docstring**: list each signature on its own indented line, then write one body that covers them all.
- **Structured returns** (`NamedTuple`, composite): use a `-> ReturnType` annotation in the signature line and sketch the shape under `# Returns`.

See the cheat sheet (§3.10) for the full skeleton.

### 3.8 Examples and admonitions

**Examples** — at least one for any user-facing API; optional for internal helpers. Use a plain fenced ` ```julia ` block; examples should be runnable in a fresh REPL after `using YourPackage` (spell out non-trivial setup). One short example is better than three sprawling ones.

**Admonitions** ([Documenter syntax](https://documenter.juliadocs.org/stable/showcase/#Admonitions)) — use sparingly for genuinely important caveats. Kinds: `note`, `warning`, `tip`, `info`, `compat`, `danger`.

~~~markdown
!!! warning
    Calling this function outside `set_precomputed_quantities!` reads stale `p.scratch`
    values. Run after `set_implicit_precomputed_quantities_part1!`.
~~~

Keep to 1–3 sentences. If you find yourself writing more than two admonitions in one docstring, the docstring is doing too much — split it or move content into `docs/src/`.

### 3.9 `# Extended help`

Documenter and the Julia REPL recognize `# Extended help` as a special trailing section: `?function_name` shows everything before it; `??function_name` shows the full docstring. Use it for long boundary cases, implementation notes for maintainers, or detailed cross-cutting `See also` lists. It must appear last. Long derivations belong on a *Mathematical Formulation* page in `docs/src/`, not in extended help.

### 3.10 Cheat sheet

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

### 3.11 What we don't use, and other anti-patterns

- **`DocStringExtensions`** (`$(TYPEDEF)`, `$(FIELDS)`, `$(SIGNATURES)`, …). Spell out signatures, field lists, and types by hand. Readability in source matters more than DRY.
- **`jldoctest`** blocks. Most CliMA repos don't run doctests in CI, so jldoctests silently rot. Use plain ` ```julia ` blocks.
- **Long mathematical derivations inline.** The docstring should give the reader enough to use the function; derivations belong on an *Explanation* page or in the source paper. Link to it.
- **Generated docstrings via metaprogramming** (macros that splice docstrings into `@eval`'d definitions) unless unavoidable — hard to grep, hard to read, easy to break.
- **Missing signature line** (`"""Compute X..."""` with no indented signature) — breaks REPL help and Documenter rendering.
- **Restating the obvious** (`"""Return the input."""` for `identity`) — noise.
- **Documenting *how* instead of *why*** — the implementation is right below; callers want what to pass in, what they get back, what assumptions apply.
- **Out-of-date signatures.** When renaming an argument, update both the signature line and the `# Arguments` bullets. CI does not catch divergence.
- **Inconsistent units** within one docstring — worse than no docstring.
- **`Arguments:`, `Inputs:`, `Returns:` as plain prose lines** — these are invisible structure; Documenter renders only the `# Heading` form.
- **Mixing imperative and third person** within one docstring — pick one.
- **Multi-paragraph docstrings on internal helpers.** If you are writing more than ~15 lines for a helper called from one place, the prose belongs in a block comment near the call site or on a docs page.

## 4. Documenter.jl pitfalls

**Markdown link ambiguity.** `[kg/m^3](description)` is parsed as a markdown link and produces `:cross_references` errors if the parenthetical text is not a URL. Fix: use parentheses for units (`(kg/m^3)`), or separate brackets and parentheses with punctuation. Do not attempt to escape brackets with backslashes in Julia string literals — that causes invalid-escape-sequence errors during precompilation.

**Missing docstrings.** If `makedocs` fails with "Missing docstrings", ensure every exported symbol with a docstring is included on a documentation page via `@docs` or `@autodocs`.

**Undefined cross-package symbols.** Use fully qualified names (`Thermodynamics.ThermodynamicsParameters`) so Documenter's link generator can resolve them across package boundaries. For cross-repo `@ref`-style links, configure DocumenterInterLinks (§3.4).

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
