# Contributors Guide

Thank you for considering contributions to ClimaAtmos! We hope this guide
helps you make a contribution.

Feel free to ask us questions and chat with us at any time about any topic at all
by:

  - [Opening a GitHub issue](https://github.com/CliMA/ClimaAtmos.jl/issues/new)

!!! note "Shared CliMA engineering standards"

    The authoritative CliMA-wide engineering guides (code style, documentation
    policy, software design patterns, performance, testing, and workflows) are
    vendored into this repository under
    [`docs/dev-guides/`](https://github.com/CliMA/ClimaAtmos.jl/blob/main/docs/dev-guides/README.md)
    (from [CliMA/DeveloperGuides](https://github.com/CliMA/DeveloperGuides)).
    This page covers the ClimaAtmos-specific and community aspects of
    contributing and defers to those guides for everything else.

## Creating issues

The simplest way to contribute to ClimaAtmos is to create or comment on issues.

The most useful bug reports:

  - Provide an explicit code snippet --- not just a link --- that reproduces the bug in the latest tagged version of ClimaAtmos. This is sometimes called the ["minimal working example"](https://en.wikipedia.org/wiki/Minimal_working_example). Reducing bug-producing code to a minimal example can decrease the time it takes to resolve an issue.

  - Paste the _entire_ error received when running the code snippet, even if it's unbelievably long.

  - Use triple backticks (e.g., ````` ```some_code; and_some_more_code;``` `````) to enclose code snippets, and other [markdown formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax) to make your issue easy and quick to read.

  - Report the ClimaAtmos version, Julia version, machine (especially if using a GPU) and any other possibly useful details of the computational environment in which the bug was created.

Discussions are recommended for asking questions about (for example) the user interface, implementation details, science, and life in general.

## But I want to _code_!

  - New users help write ClimaAtmos code and documentation by [forking the ClimaAtmos repository](https://docs.github.com/en/github/collaborating-with-pull-requests/working-with-forks), [using git](https://guides.github.com/introduction/git-handbook/) to edit code and docs, and then creating a [pull request](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork). Pull requests are reviewed by ClimaAtmos collaborators.

  - A pull request can be merged once it is reviewed and approved by collaborators. If the pull request author has write access, they have the responsibility of merging their pull request. Otherwise, ClimaAtmos.jl collaborators will execute the merge with permission from the pull request author.

  - Note: for small or minor changes (such as fixing a typo in documentation), the [GitHub editor](https://docs.github.com/en/github/managing-files-in-a-repository/managing-files-on-github/editing-files-in-your-repository) is super useful for forking and opening a pull request with a single click.

  - Write your code with love and care. In particular, conform to existing ClimaAtmos style and formatting conventions. It's worth a few extra minutes of our time to leave future generations with well-written, readable code.

### General coding guidelines

Naming, formatting, and structural conventions are defined in the shared
[code style guide](https://github.com/CliMA/ClimaAtmos.jl/blob/main/docs/dev-guides/code-quality/code_style.md);
design principles (composability, dispatch patterns, cache discipline) are in the
[software design patterns guide](https://github.com/CliMA/ClimaAtmos.jl/blob/main/docs/dev-guides/code-quality/software_design_patterns.md).
Please skim both before writing new code.

## What is a "collaborator" and how can I become one?

  - Collaborator status allows a contributor to review pull requests in addition to opening them. Collaborators can also create branches in the main ClimaAtmos repository.

  - We ask that new contributors try their hand at forking ClimaAtmos, and opening and merging a pull request before requesting collaborator status.

## What's a good way to start developing ClimaAtmos?

  - Tackle an existing issue. We keep a list of [good first issues](https://github.com/CLiMA/ClimaAtmos.jl/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
    that are self-contained and suitable for a newcomer to try and work on.

  - Try to run ClimaAtmos and play around with it to simulate your favorite
    fluids and atmosphere physics. If you run into any problems or find it difficult
    to use or understand, please open an issue!

  - Write up an example or tutorial on how to do something useful with
    ClimaAtmos, like how to set up a new physical configuration.

  - Improve documentation or comments if you found something hard to use.

  - Implement a new feature if you need it to use ClimaAtmos.

If you're interested in working on something, let us know by commenting on existing issues or
by opening a new issue. This is to make sure no one else is working on the same issue and so
we can help and guide you in case there is anything you need to know beforehand.

## Ground Rules

  - Each pull request should consist of a logical collection of changes. You can
    include multiple bug fixes in a single pull request, but they should be related.
    For unrelated changes, please submit multiple pull requests.

  - Do not commit changes to files that are irrelevant to your feature or bugfix
    (eg: `.gitignore`).

  - Be willing to accept criticism and work on improving your code; we don't want
    to break other users' code, so care must be taken not to introduce bugs. We
    discuss pull requests and keep working on them until we believe we've done a
    good job.

  - Be aware that the pull request review process is not immediate, and is
    generally proportional to the size of the pull request.

## Reporting a bug

The easiest way to get involved is to report issues you encounter when using
ClimaAtmos or to request something you think is missing.

  - Head over to the [issues](https://github.com/CLiMA/ClimaAtmos.jl/issues) page.

  - Search to see if your issue already exists or has even been solved previously.

  - If you indeed have a new issue or request, click the "New Issue" button.

  - Please be as specific as possible. Include the version of the code you were using, as
    well as what operating system you are running. The output of Julia's `versioninfo()`
    and `] status` is helpful to include. Try your best to include a complete, ["minimal working example"](https://en.wikipedia.org/wiki/Minimal_working_example) that reproduces the issue.

## Setting up your development environment

  - Install [Julia](https://julialang.org/) on your system.

  - Install `git` on your system if it is not already there (install XCode command line tools on
    a Mac or `git bash` on Windows).

  - Login to your GitHub account and make a fork of the
    [ClimaAtmos repository](https://github.com/CLiMA/ClimaAtmos.jl) by
    clicking the "Fork" button.

  - Clone your fork of the ClimaAtmos repository (in terminal on Mac/Linux or git shell/
    GUI on Windows) in the location you'd like to keep it.

    ```
    git clone https://github.com/your-user-name/ClimaAtmos.jl.git
    ```

  - Navigate to that folder in the terminal or in Anaconda Prompt if you're on Windows.

  - Connect your repository to the upstream (main project).

    ```
    git remote add ClimaAtmos https://github.com/CLiMA/ClimaAtmos.jl.git
    ```

For instantiating the Julia environments, day-to-day REPL workflow, running
tests, and resolving stuck environments, follow the shared
[onboarding guide](https://github.com/CliMA/ClimaAtmos.jl/blob/main/docs/dev-guides/workflow/onboarding.md);
the layout of this repository's environments is described in the
[dependency management guide](https://github.com/CliMA/ClimaAtmos.jl/blob/main/docs/dev-guides/architecture/dependency_management.md).

## Pull Requests

We follow the [ColPrac guide](https://github.com/SciML/ColPrac) for collaborative practices.
We ask that new contributors read that guide before submitting a pull request.

Changes and contributions should be made via GitHub pull requests against the ``main`` branch.

When you're done making changes, commit the changes you made. Chris Beams has written a
[guide](https://chris.beams.io/posts/git-commit/) on how to write good commit messages.

When you think your changes are ready to be merged into the main repository, push to your fork
and [submit a pull request](https://github.com/CLiMA/ClimaAtmos.jl/compare/).

**Working on your first Pull Request?** You can learn how from this _free_ video series
[How to Contribute to an Open Source Project on GitHub](https://egghead.io/courses/how-to-contribute-to-an-open-source-project-on-github), Aaron Meurer's [tutorial on the git workflow](https://www.asmeurer.com/git-workflow/), or the guide [“How to Contribute to Open Source"](https://opensource.guide/how-to-contribute/).

## Documentation

Now that you've made your awesome contribution, it's time to tell the world how to use it.
Writing documentation strings is important to make sure others use your functionality
properly. Didn't write new functions? That's fine, but be sure that the documentation for
the code you touched is still in great shape. It is not uncommon to find some strange wording
or clarification that you can take care of while you are here.

Docstring anatomy, section headings, units, math, citations, and worked examples
are all specified in the shared
[documentation policy](https://github.com/CliMA/ClimaAtmos.jl/blob/main/docs/dev-guides/code-quality/documentation_policy.md);
follow it for every docstring and documentation page you touch.

You can preview how the documentation will look after merging by building the documentation
locally. From the main directory of your local repository call

```
julia --project=docs -e 'using Pkg; Pkg.develop(Pkg.PackageSpec(path = ".")); Pkg.instantiate()'
JULIA_DEBUG=Documenter julia --project=docs docs/make.jl
```

and then open `docs/build/index.html` in your favorite browser. Setting the environment variable
`JULIA_DEBUG=Documenter` will provide more information on the documentation build process and
thus help figure out a potential bug.

## Formatting

One of the CI checks verifies that the code is uniformly formatted with
[JuliaFormatter.jl](https://github.com/JuliaEditorSupport/JuliaFormatter.jl);
the rules are defined in the root `.JuliaFormatter.toml`. Usage, the
version-consistency requirement, and the recommended `prek` pre-commit hooks
(defined in `.pre-commit-config.yaml`, running the formatter from the
version-pinned `.dev/format/` environment so results match the
[Prek CI check](https://github.com/CliMA/ClimaAtmos.jl/blob/main/.github/workflows/run-prek.yml))
are all documented in the shared
[code style guide, §1](https://github.com/CliMA/ClimaAtmos.jl/blob/main/docs/dev-guides/code-quality/code_style.md).

!!! note

    In the past, `ClimaAtmos` used to have a `.dev/climaformat.jl` script. We moved
    away from it to reduce complexity in our repository and to align with the
    general tools used by the Julia community. If you are still using
    `climaformat.jl`, migrate to `JuliaFormatter` (`climaformat.jl` was just a
    wrapper around `JuliaFormatter`).

## Updating environments

The repository for `ClimaAtmos` includes several checked `Manifests.toml`. This
is to help with reproducing results.
[PkgDevTools](https://juliahub.com/ui/Packages/General/PkgDevTools) provides a
convenient system to quickly update all the `Manifests.toml`: add it to your
base environment with `Pkg.add("PkgDevTools")`, then run
`using PkgDevTools; PkgDevTools.update_deps(".")` from the repository root.

!!! note

    In the past, `ClimaAtmos` used to have a `.dev/up_deps.jl` script. We moved away
    from it because `PkgDevTools` provides a simpler and more efficient way to
    accomplish the same result.

## Credits

This contributor's guide is based on the excellent [Oceananigans.jl contributor's guide](https://github.com/CliMA/Oceananigans.jl/blob/main/CONTRIBUTING.md), which is in turn based on the [MetPy contributor's guide](https://github.com/Unidata/MetPy/blob/main/CONTRIBUTING.md).
