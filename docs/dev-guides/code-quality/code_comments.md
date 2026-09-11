# Code Comments

Conventions for writing, editing, and reviewing comments in source, tests, and CI config. Docstrings are covered separately in [documentation_policy.md §3](documentation_policy.md); this guide is about the `#` comments that sit next to code, though the defects below appear in docstrings and YAML just as often.

## 1. The governing rule

**A comment explains the code that is there, and why it is that way.** It is not a changelog, not a narrative of the work that produced it, and not a restatement of the line below it.

Match the comment density and voice of the surrounding file rather than importing a different style. A new file dense with prose in a codebase whose neighbours are sparse is out of place, and the extra prose is almost always saying what the code already says.

A header on a new file is justified when it states what the file does or asserts and why that is the right thing to do or assert. Keep it to a few lines. It is not the place for:

- which precisions, resolutions, or cases are swept — the loop shows this;
- which sibling files cover adjacent ground — navigation belongs in the README;
- a restatement of the function or testset names below.

## 2. Never describe a previous state

The single most common defect, and the one to look for first. All of these are wrong in a committed comment:

- "this used to fail on GPUs because ..."
- "the fluxes were previously mislabeled ..."
- "their former example files are deleted"
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

## 3. Failure modes are worth documenting

This is the exception that makes the rule useful. Document a hazard when a future developer could plausibly reintroduce it — but state it as a property of the current code, not as an incident report. Hazards worth a comment look like:

- `if: ${{ a }} && ${{ b }}` in GitHub Actions interpolates each `${{ }}` separately and leaves the string `"true && false"`, which is truthy either way, so the condition must be written as one expression.
- `--check-bounds=yes` disables `@inbounds` elision, so zero-allocation sentinels report allocations that are not there.
- An inline `@allocated f(...)` depends on the caller's inlining budget; measure from a `@noinline` wrapper.
- GPU kernel launches allocate host-side wrappers, so byte-count sentinels are CPU-only.

Each states a constraint that is still true of the code as written, and each would cost someone a debugging session to rediscover.

## 4. Words to cut

**Adverbs and hedges.** simply, just, actually, essentially, effectively, merely, basically, clearly, obviously, carefully, properly, correctly, very, particularly, especially, notably, importantly, deliberately, intentionally.

**Weak adjectives.** simple, easy, clean, nice, proper, robust, comprehensive, thorough, genuine, obvious, trivial, straightforward, crucial, critical, essential, powerful, elegant.

**Connectives and filler openers.** "Note that", "It's worth noting", "In other words", "That said", "Keep in mind", "As such", "By design", "The whole point is", "exactly what we want".

**Keep a flagged word when it carries meaning.** `silently` in "fails silently rather than erroring" *is* the content of the sentence. `sharp` describing a physical gradient is a measurement, not praise. "used to compare" means *employed to*, not *formerly*. These lists are grep targets for review, not find-and-replace rules.

## 5. Name the mechanism, not its price

Avoid borrowed vocabulary — economics-speak especially — that sounds knowing and says little:

| Instead of | Write |
|:---|:---|
| "shards the suite" | "splits the suite" |
| "instrumenting the others buys nothing" | "only this job's coverage is uploaded" |
| "coverage is not free: it costs the job" | "instrumented runs are killed partway through; uninstrumented ones finish" |
| "these tests cost minutes apiece" | "this marks the multi-minute tests" |

## 6. Numbers over characterization

A measured number beats an adjective, and it dates the claim implicitly:

| Instead of | Write |
|:---|:---|
| "takes a while" | "~2.5x as long" |
| "uses a lot of memory" | "peaks at 3.8 GB" |
| "converges well" | "measured 2.82 against design 3" |
| "roughly stable across builds" | "within ~10–20% across builds" |

When a tolerance or baseline is pinned in code, say where the number came from — the resolution, configuration, and margin it was measured at — so the next person knows whether they are allowed to move it.

## 7. Mechanics

- **The formatter does not reflow comments.** JuliaFormatter leaves comment text alone, so rewrapping after an edit is manual: check line lengths yourself. Keep comments within the repo's margin (see [code_style.md §4](code_style.md)); wrapping narrower than the code margin is common and fine, as long as it is consistent within a file.
- **No trailing whitespace.**
- **Prefer the declarative to the imperative.** An imperative opener ("Split the branch outside the closure") reads as an instruction to change the code. Write what is true instead: "The branch sits outside the closure so that ...".
- **In `#! format: off` blocks, alignment is manual.** Realign the whole column after editing any entry.

## 8. Sweeping a branch for comment defects

When reviewing or cleaning up a branch, work from the diff rather than by reading files.

**Know which diff you are reading.** `git diff origin/main` reads the **working tree**; `git diff origin/main <branch>` reads the **committed tip**. Sweeping the wrong one silently reports stale results.

**Scan every added line, not only `#` comments.** Docstrings and config values carry the same defects, and a comment-only filter never sees them. A stale value in a CI config is the dangerous case: it changes which jobs run without failing anything.

```bash
git diff origin/main HEAD > /tmp/d.diff
awk '/^\+\+\+ b\// { f=substr($0,7) }
     /^\+/ && !/^\+\+\+/ { print f"\t"substr($0,2) }' /tmp/d.diff
```

Narrow to `^[[:space:]]*#` only for line-length and density checks. Then grep the added lines for:

- the word lists in §4;
- historical phrasing: `former|used to |previously|no longer|has since|were dropped`;
- lines over the margin.

**Review each hit rather than replacing blindly.** The false-positive rate is high: "this **pr**operty" matches `this PR` case-insensitively, and "**used to** compare" is not historical.

Finish by checking that filenames cited in comments still exist — extract every `*.jl` / `*.yml` / `*.md` reference and confirm each resolves, since renames and deletions strand them. Package names (`HDF5.jl`), globs (`*_utils.jl`), URL fragments, and documented placeholders are expected non-matches.

## 9. Two ways comments go stale

**Comments go stale across branches, not just across time.** A comment that is accurate on one branch can be false on another that renames or deletes what it describes. When rebasing a stack, a conflict inside a comment is a signal to check which branch's description is still true — not just to pick one.

**Renaming a symbol, flag, or test tier strands its old name in prose and config.** The rename itself compiles and passes; the name survives in a docstring list, in an example, and in CI config values that select on it. Those config values then quietly select something different from what they used to — silently changing which tests run, with nothing failing to signal it. Before calling such a rename done, grep the old name across source, config, and docs, and confirm that any config value naming it still selects what it did before.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
