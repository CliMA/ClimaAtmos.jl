# Code Comments and Prose Style

Conventions for writing, editing, and reviewing inline comments, docstrings, and documentation prose across source, tests, docs, and CI config. For the structure and rendering of docstrings and repository-level documentation, see [documentation_policy.md](documentation_policy.md).

## 1. Inline comments

### 1.1 The governing rule

**A comment explains the code that is there, and why it is that way.** It is not a changelog, not a narrative of the work that produced it, and not a restatement of the line below it.

Match the comment density and voice of the surrounding file. A new file dense with prose in a codebase whose neighbours are sparse is out of place, and the extra prose is almost always saying what the code already says.

A header on a new file is justified when it states what the file does or asserts and why that is the right thing to do or assert. Keep it to a few lines. It is not the place for:

- which precisions, resolutions, or cases are swept — the loop shows this;
- which sibling files cover adjacent ground — navigation belongs in the README;
- a restatement of the function or testset names below.

### 1.2 Failure modes worth documenting

Document a hazard when a future developer could plausibly reintroduce it — but state it as a property of the current code, not as an incident report. Hazards worth a comment look like:

- `if: ${{ a }} && ${{ b }}` in GitHub Actions interpolates each `${{ }}` separately and leaves the string `"true && false"`, which is truthy either way, so the condition must be written as one expression.
- `--check-bounds=yes` disables `@inbounds` elision, so zero-allocation sentinels report allocations that are not there.
- An inline `@allocated f(...)` depends on the caller's inlining budget; measure from a `@noinline` wrapper.
- Weak-form operator values at element-boundary nodes are incomplete before DSS (or a DG interface flux) completes them, so a pre-DSS jump diagnostic applies only to the strong form.
- GPU kernel launches allocate host-side wrappers, so byte-count sentinels are CPU-only.

Each states a constraint that is still true of the code as written, and each would cost someone a debugging session to rediscover.

### 1.3 Comment mechanics

- **The formatter does not reflow comments.** JuliaFormatter leaves comment text alone, so rewrapping after an edit is manual: check line lengths yourself. Keep comments within the repo's margin (see [code_style.md §4](code_style.md)); wrapping narrower than the code margin is common and fine, as long as it is consistent within a file.
- **No trailing whitespace.**
- **Prefer the declarative to the imperative.** An imperative opener ("Split the branch outside the closure") reads as an instruction to change the code. Write what is true: "The branch is outside the closure so that ...".
- **In `#! format: off` blocks, alignment is manual.** Realign the whole column after editing any entry.

## 2. Prose style (comments, docstrings, and docs)

The rules in this section apply across `#` comments, docstrings, `docs/src/` pages, and YAML configuration.

### 2.1 Never describe a previous state

The single most common defect, and the one to look for first. All of these are wrong in a committed comment:

- "this used to fail on GPUs because ..."
- "the fluxes were previously mislabeled ..."
- "their former example files are deleted"
- "One job per case, matching how `main` ran these"
- "the former per-step-start callback produced this schedule"
- "these were already soft-failing before the split"

Rewrite to the present. Keep the hazard, drop the history:

```julia
# BEFORE
# The upwinded stencil, written as a fused broadcast. This used to fail on
# GPUs because `Base.Fix1(convert, T)` stores a `Type` field and so is not a
# bitstype; `ConvertTo{T}` is an empty struct, so the fused form now compiles.

# AFTER
# The conversion goes through `ConvertTo{T}`, an empty struct, rather than
# `Base.Fix1(convert, T)`: the latter stores a `Type` field and so is not a
# bitstype, which a fused broadcast cannot compile for the GPU.
```

The history belongs in the commit message, where it is dated and retrievable. A reader of the code a year from now needs the constraint, not the story of how it was found.

### 2.2 State what is the case

The second most common defect. A negative tail — "rather than X", "instead of X", "without X", "not X" — earns its place only when X is information. Cut it when the positive clause already excludes X, or when X is an alternative no reader would assume:

| Instead of | Write |
|:---|:---|
| "raises an error rather than silently doing nothing" | "raises an error" |
| "start from a preset rather than assembling the list by hand" | "start from a preset" |
| "evaluated in two calls rather than one" | "evaluated in two calls" |
| "resolved without uniform spacing aloft" | "has high vertical resolution that becomes coarser aloft" |
| "The setup and the physics are not independent: the setup sets…" | "The setup sets…" |
| "Nothing in the equations assumes convection is unresolved" | "The equations hold whether or not convection is resolved" |
| "is not a matter of taste: the choice determines…" | "the choice decides…" |

Keep a negative when it *is* the content: an absent guard ("Nothing rejects that configuration, so…"), a TODO ("not yet in the code"), a correctness statement ("potential temperature is not conserved"), a real design alternative ("advective rather than flux form"; "signed; the other two are absolute values"), or a symmetry the reader would otherwise assume ("the horizontal combination is rejected; the vertical one is not").

**Contrasts with a paper, another model, or an earlier version are the same defect in disguise.** "``z_d`` is an absolute height, unlike in Yatunin et al." becomes "The onset height is an absolute altitude"; "Unlike in earlier EDMF formulations, it is not structurally distinct" becomes "It is an ordinary subdomain, distinguished only by having its equations eliminated". Drop "unlike in [Cohen2020]" and state the physics. The one contrast worth keeping is an actionable code-vs-formulation discrepancy, and that goes in a `!!! note "TODO: …"` or `"Departure from the formulation"` admonition where the reader who compares against the paper will look for it.

Also convert passive "X is not subsided / are not diffused" to the active intransitive when one exists: "rain and snow do not subside", "sedimenting species do not diffuse".

### 2.3 Words to cut

**Adverbs and hedges.** simply, just, actually, essentially, effectively, merely, basically, clearly, obviously, carefully, properly, correctly, very, particularly, especially, notably, importantly, deliberately, intentionally.

**Weak adjectives.** simple, easy, clean, nice, proper, robust, comprehensive, thorough, genuine, sharp, obvious, trivial, straightforward, crucial, critical, essential, powerful, elegant.

**Connectives and filler openers.** "Note that", "It's worth noting", "In other words", "That said", "Keep in mind", "As such", "By design", "The whole point is", "exactly what we want", "Unlike X, …" (see §2.2).

**Verbs and nouns that read as machine-written.**

- "sits / sits on" → "lives on", "stays near", "falls on"
- "consumes / consumed" (data flow) → "uses", "reads"
- "carries" — fine for physical transport ("the flux carries enthalpy"); thin it where it means *has*, *includes*, *uses*, or *passes*
- the emphatic cleft "X is what makes Y" / "which is what lets" → "X makes Y" / "which lets" (keep true free relatives such as "which is what ClimaCore works in")
- "configuration surface" → "configuration options" (*surface* already means the lower boundary in an atmosphere or land model)
- "developer territory" → "a developer task"; "payload" → "terms"; "recipe" → "prescription" or "the minimum is"; "knob" → "parameter"; "in flux" → "under active development"; "for free" → drop; "wires X into" → "passes X to"
- "load-bearing", "foundational", "memoize", "tractable" — avoid

**Keep a flagged word when it carries meaning.** `silently` in "fails silently rather than erroring" *is* the content of the sentence. `sharp` describing a physical gradient is a measurement, not praise. "used to compare" means *employed to*, not *formerly*. These lists are grep targets for review, not find-and-replace rules.

### 2.4 Name the mechanism, not its price

Avoid borrowed vocabulary — economics-speak especially — that sounds knowing and says little:

| Instead of | Write |
|:---|:---|
| "shards the suite" | "splits the suite" |
| "instrumenting the others buys nothing" | "only this job's coverage is uploaded" |
| "coverage is not free: it costs the job" | "instrumented runs are killed partway through; uninstrumented ones finish" |
| "these tests cost minutes apiece" | "this marks the multi-minute tests" |

### 2.5 Numbers over characterization

A measured number beats an adjective, and it dates the claim implicitly:

| Instead of | Write |
|:---|:---|
| "takes a while" | "~2.5x as long" |
| "uses a lot of memory" | "peaks at 3.8 GB" |
| "converges well" | "measured 2.82 against design 3" |
| "roughly stable across builds" | "within ~10–20% across builds" |

When a tolerance or baseline is pinned in code, say where the number came from — the resolution, configuration, and margin it was measured at — so the next person knows whether they are allowed to move it.

### 2.6 Punctuation

- **Comma after a sentence-initial adverbial, however short.** "In the code, this is…", "Here, ``\Delta x`` is…", "By default, a single iteration…". Not after a gerund subject ("Setting ``A`` to zero recovers…").

## 3. Sweeping a branch and catching stale prose

### 3.1 Diff-based sweep workflow

When reviewing or cleaning up a branch, work from the diff rather than by reading files.

**Know which diff you are reading.** `git diff origin/main` reads the **working tree**; `git diff origin/main <branch>` reads the **committed tip**. Sweeping the wrong one silently reports stale results.

**Scan every added line, not only `#` comments.** Docstrings and config values have the same defects, and a comment-only filter never sees them. A stale value in a CI config is the dangerous case (§3.2).

```bash
git diff origin/main HEAD > /tmp/d.diff
awk '/^\+\+\+ b\// { f=substr($0,7) }
     /^\+/ && !/^\+\+\+/ { print f"\t"substr($0,2) }' /tmp/d.diff
```

Narrow to `^[[:space:]]*#` only for line-length and density checks. Then grep the added lines for:

- the word lists in §2.3;
- historical phrasing: `former|used to |previously|no longer|has since|were dropped`;
- negative tails and contrasts: `rather than|instead of|without|never|not yet|nothing|is not|are not|does not|cannot|unlike|by contrast`;
- lines over the margin.

**Review each hit rather than replacing blindly.** The negation grep is a candidate list, not a hit list: in a typical sweep, roughly a tenth of the matches are redundant and the rest carry information. The false-positive rate is high across all of these patterns: "this **pr**operty" matches `this PR` case-insensitively, and "**used to** compare" is not historical.

Finish by checking that filenames cited in comments still exist — extract every `*.jl` / `*.yml` / `*.md` reference and confirm each resolves, since renames and deletions strand them. Package names (`HDF5.jl`), globs (`*_utils.jl`), URL fragments, and documented placeholders are expected non-matches.

### 3.2 Two ways comments and prose go stale

**Comments and prose go stale across branches, not just across time.** A comment or docstring that is accurate on one branch can be false on another that renames or deletes what it describes. When rebasing a stack, a conflict inside prose is a signal to check which branch's description is still true — not just to pick one.

**Renaming a symbol, flag, or test tier strands its old name in prose and config.** The rename itself compiles and passes; the name survives in a docstring list, in an example, and in CI config values that select on it. Those config values then select something different, changing which tests run with nothing failing to signal it. Before calling such a rename done, grep the old name across source, config, and docs, and confirm that any config value naming it still selects what it did before.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
