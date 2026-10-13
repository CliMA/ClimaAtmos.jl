#!/usr/bin/env python3
"""Flag square brackets that Documenter will parse as a markdown link.

Documenter parses ``[text](target)`` as a link. Its parser is more permissive
than CommonMark: **any** run of whitespace between the closing bracket and the
opening parenthesis still forms a link, including a line break. So all four of
these become links whose destination is the parenthetical text, and the docs
build fails with a ``:cross_references`` error:

    Density [kg/m³](as measured at cloud base).
    Density [kg/m³] (as measured at cloud base).
    Density [kg/m³]  (as measured at cloud base).
    Density [kg/m³]
    (as measured at cloud base).

Separating the two with punctuation, or wrapping the units in backticks, is
safe:

    Density [kg/m³], as measured at cloud base.
    Density [kg/m³]. (As measured at cloud base.)
    Density `[kg/m³]` (as measured at cloud base).

This matters even for docstrings that no page renders today: `docs/make.jl`
uses ``checkdocs = :exports``, so an unrendered docstring's broken link stays
latent until someone adds that symbol to a page, and the build then fails in a
seemingly unrelated PR.

Usage:
    .dev/check_markdown_link_ambiguity.py [paths...]

With no arguments, scans `src/` and `docs/src/`. Exits non-zero if any
ambiguity is found.

Only prose is scanned: docstring bodies in `.jl`, and in `.md` everything
outside fenced blocks. Julia code, code comments, ``` and ~~~ fences, and
`inline code spans` are all skipped, so array indexing such as `a[i](x)`, and
documentation that deliberately quotes the bad pattern as an example, do not
produce false positives.
"""

import pathlib
import re
import sys

# A link target is a Documenter directive, a URL, an anchor, or a path. Prose is
# none of those. Rather than enumerate file extensions -- which wrongly flagged
# real links such as `[src/ClimaAtmos.jl](src/ClimaAtmos.jl)` -- classify the
# whole parenthetical.
TARGET_SCHEME = re.compile(r"^(@ref\b|@extref\b|@cite\b|<|#|https?://|ftps?://|mailto:)")
TARGET_PATH = re.compile(r"^[\w.@/+~%#-]+$")  # a path or filename: no whitespace
TARGET_EXTENSION = re.compile(r"\.[A-Za-z0-9]{1,8}$")


def is_link_target(target):
    """Whether the parenthesised text is a plausible link target, not prose.

    A bare word (``(required)``) and anything containing whitespace
    (``(at cloud base)``) are prose. A path needs a file extension or a
    trailing slash, so that units such as ``(kg/kg)`` are still reported.
    """
    target = target.strip()
    if not target:
        return False
    if TARGET_SCHEME.match(target):
        return True
    if not TARGET_PATH.match(target):
        return False
    path = target.split("#", 1)[0]  # `api.md#Grids` is a path plus an anchor
    return bool(TARGET_EXTENSION.search(path)) or path.endswith("/")

# A closing bracket followed by any whitespace (including newlines) and "(".
AMBIGUITY = re.compile(r"\][ \t\r\n]*\(", re.S)

DEFAULT_ROOTS = ("src", "docs/src")


def docstring_spans(text):
    """Return (start, stop) offsets of the bodies of triple-quoted strings."""
    spans = []
    i = 0
    while True:
        start = text.find('"""', i)
        if start < 0:
            break
        stop = text.find('"""', start + 3)
        if stop < 0:
            break
        spans.append((start + 3, stop))
        i = stop + 3
    return spans


FENCE = re.compile(r"^\s*(```|~~~)")
INLINE_CODE = re.compile(r"`[^`\n]*`")


def strip_fenced_blocks(text):
    """Blank out fenced code blocks, preserving line numbering.

    Both ``` and ~~~ fences are recognised; documentation about this very
    pitfall necessarily contains examples of it inside fences.
    """
    out = []
    in_fence = False
    for line in text.split("\n"):
        if FENCE.match(line):
            in_fence = not in_fence
            out.append("")
        else:
            out.append("" if in_fence else line)
    return "\n".join(out)


def strip_inline_code(text):
    """Blank out `inline code spans`, preserving length and line numbering.

    Text inside backticks is a code span, not prose, so Documenter does not
    parse a link there. Guides that quote the bad pattern inline rely on this.
    """
    return INLINE_CODE.sub(lambda m: " " * len(m.group(0)), text)


def parenthetical(text, open_paren_index):
    """Return the text between the parentheses starting at `open_paren_index`."""
    close = text.find(")", open_paren_index)
    return "" if close < 0 else text[open_paren_index + 1 : close]


def report(path, text, offset, match_start, match_end):
    line = text.count("\n", 0, offset + match_start) + 1
    excerpt = text[offset + max(0, match_start - 60) : offset + match_end + 40]
    excerpt = " ".join(excerpt.split())
    print(f"{path}:{line}: ambiguous link: …{excerpt}…")


def scan_julia(path):
    text = path.read_text()
    found = 0
    for start, stop in docstring_spans(text):
        body = strip_inline_code(strip_fenced_blocks(text[start:stop]))
        for m in AMBIGUITY.finditer(body):
            if is_link_target(parenthetical(body, m.end() - 1)):
                continue
            report(path, text, start, m.start(), m.end())
            found += 1
    return found


def scan_markdown(path):
    prose = strip_inline_code(strip_fenced_blocks(path.read_text()))
    found = 0
    for m in AMBIGUITY.finditer(prose):
        if is_link_target(parenthetical(prose, m.end() - 1)):
            continue
        report(path, prose, 0, m.start(), m.end())
        found += 1
    return found


def main(argv):
    paths = []
    for arg in argv or DEFAULT_ROOTS:
        p = pathlib.Path(arg)
        if p.is_dir():
            paths.extend(sorted(p.rglob("*.jl")))
            paths.extend(sorted(p.rglob("*.md")))
        elif p.suffix in (".jl", ".md"):
            paths.append(p)

    found = 0
    for path in paths:
        # dev-guides are vendored from CliMA/DeveloperGuides; fix them upstream.
        if "dev-guides" in path.parts:
            continue
        if path.suffix == ".jl":
            found += scan_julia(path)
        else:
            found += scan_markdown(path)

    if found:
        print(
            f"\n{found} ambiguous link(s). Separate the bracket and the "
            f"parenthesis with punctuation, or wrap the units in backticks.\n"
            f"See docs/dev-guides/code-quality/documentation_policy.md.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
