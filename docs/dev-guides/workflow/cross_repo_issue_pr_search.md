# Searching Issues & PRs Across CliMA

CliMA spans dozens of repositories with no built-in cross-repo view of
issues and PRs. GitHub's **org-scoped search** is that view: one query
returns matches from every repo in the org at once.

## 1. The org-scoped search pattern

Scope to the org with `org:CliMA` and narrow with state/type/label
qualifiers. The canonical example (from the group-meeting discussion on
getting an overview across CliMA) is open issues/PRs labelled "bug 🐞",
shown here in all three forms:

- **Web UI**: paste into the box at <https://github.com/search>:

  ```text
  org:CliMA state:open label:"bug 🐞"
  ```

- **Bookmarkable URL**: spaces become `+`; add `&type=issues` to land on
  the Issues tab:

  ```text
  https://github.com/search?q=org:CliMA+state:open+label:"bug+🐞"&type=issues
  ```

- **CLI / agents**: `gh search` returns the same set as machine-readable
  text; use it for scripts and scheduled jobs:

  ```bash
  gh search issues --owner CliMA --state open --label "bug 🐞"
  gh search prs    --owner CliMA --state open --label "dependencies"
  ```

Labels with spaces or emoji **must be quoted** (`label:"bug 🐞"`); the
emoji is part of the string. Copy it from an existing labelled issue.

## 2. Filter recipes

Combine `org:CliMA` with these qualifiers (all work in the web UI, the URL
form, and `gh search` with the equivalent flags):

| Goal                          | Query                                               |
| ----------------------------- | --------------------------------------------------- |
| Open bugs across the org      | `org:CliMA state:open label:"bug 🐞"`               |
| PRs awaiting your review      | `org:CliMA is:pr is:open review-requested:@me`      |
| Issues assigned to you        | `org:CliMA is:issue is:open assignee:@me`           |
| Stale PRs (untouched 30 days) | `org:CliMA is:pr is:open updated:<2026-04-16`       |
| Dependabot update PRs         | `org:CliMA is:pr is:open label:dependencies`        |
| Untriaged issues              | `org:CliMA is:issue is:open no:label`               |
| Mentions needing a reply      | `org:CliMA is:open mentions:@me`                    |
| One repo only                 | replace `org:CliMA` with `repo:CliMA/ClimaAtmos.jl` |

Other qualifiers: `author:`, `-label:` (negation), `created:>YYYY-MM-DD`,
`draft:false`, `review:required`; stack multiple `label:` to require all.
[Full reference](https://docs.github.com/en/search-github/searching-on-github/searching-issues-and-pull-requests).

## Self-correction

If this guide is discovered to be stale or missing a useful query, update it.
