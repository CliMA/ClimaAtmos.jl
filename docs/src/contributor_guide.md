# Contributors Guide

Thank you for considering contributions to ClimaAtmos! We hope this guide
helps you make a contribution.

Feel free to ask us questions and chat with us at any time about any topic at all
by:

* [Opening a GitHub issue](https://github.com/CliMA/ClimaAtmos.jl/issues/new)

## Creating issues

The simplest way to contribute to ClimaAtmos is to create or comment on issues.

The most useful bug reports:

* Provide an explicit code snippet --- not just a link --- that reproduces the bug in the latest tagged version of ClimaAtmos. This is sometimes called the ["minimal working example"](https://en.wikipedia.org/wiki/Minimal_working_example). Reducing bug-producing code to a minimal example can dramatically decrease the time it takes to resolve an issue.

* Paste the _entire_ error received when running the code snippet, even if it's unbelievably long.

* Use triple backticks (e.g., ````` ```some_code; and_some_more_code;``` `````) to enclose code snippets, and other [markdown formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax) to make your issue easy and quick to read.

* Report the ClimaAtmos version, Julia version, machine (especially if using a GPU) and any other possibly useful details of the computational environment in which the bug was created.

Discussions are recommended for asking questions about (for example) the user interface, implementation details, science, and life in general.

## But I want to _code_!

* New users help write ClimaAtmos code and documentation by [forking the ClimaAtmos repository](https://docs.github.com/en/github/collaborating-with-pull-requests/working-with-forks), [using git](https://guides.github.com/introduction/git-handbook/) to edit code and docs, and then creating a [pull request](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork). Pull requests are reviewed by ClimaAtmos collaborators.

* A pull request can be merged once it is reviewed and approved by collaborators. If the pull request author has write access, they have the responsibility of merging their pull request. Otherwise, ClimaAtmos.jl collaborators will execute the merge with permission from the pull request author.

* Note: for small or minor changes (such as fixing a typo in documentation), the [GitHub editor](https://docs.github.com/en/github/managing-files-in-a-repository/managing-files-on-github/editing-files-in-your-repository) is super useful for forking and opening a pull request with a single click.

* Write your code with love and care. In particular, conform to existing ClimaAtmos style and formatting conventions. For example, we love verbose and explicit variable names, use `TitleCase` for types, `snake_case` for objects, and always.put.spaces.after.commas. For formatting decisions we loosely follow the [YASGuide](https://github.com/jrevels/YASGuide). It's worth few extra minutes of our time to leave future generations with well-written, readable code.

### General coding guidelines
1. Keep the number of members of Julia structs small if possible (less than 8 members).
2. Code should reflect "human intuition" if possible. This means abstraction should reflect how humans reason about the problem under consideration.
3. Code with small blast radius. If your code needs to be modified or extended, the resulting required changes should be as small and as localized as possible.
4. When you write code, write it with testing and debugging in mind.
5. Ideally, the lowest level structs have no defaults for their member fields. Nobody can remember all the defaults, so it is better to introduce them at the high-level API only.
6. Make sure that module imports are specific so that it is easy to trace back where functions that are used inside a module are coming from.
7. Consider naming abstract Julia types "AbstractMyType" in order to avoid confusion for the reader of your code.
8. Comments in your code should explain why the code exists and clarify if necessary, not just restate the line of code in words.
9. Be mindful of namespace issues when writing functional code, especially when writing function code that represents mathematical or physical concepts.
10. Consider using keywords in your structs to allow readers to more effectively reason about your code.

## What is a "collaborator" and how can I become one?

* Collaborators have permissions to review pull requests and status allows a contributor to review pull requests in addition to opening them. Collaborators can also create branches in the main ClimaAtmos repository.

* We ask that new contributors try their hand at forking ClimaAtmos, and opening and merging a pull request before requesting collaborator status.

## What's a good way to start developing ClimaAtmos?

* Tackle an existing issue. We keep a list of [good first issues](https://github.com/CLiMA/ClimaAtmos.jl/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
  that are self-contained and suitable for a newcomer to try and work on.

* Try to run ClimaAtmos and play around with it to simulate your favorite
  fluids and atmosphere physics. If you run into any problems or find it difficult
  to use or understand, please open an issue!

* Write up an example or tutorial on how to do something useful with
  ClimaAtmos, like how to set up a new physical configuration.

* Improve documentation or comments if you found something hard to use.

* Implement a new feature if you need it to use ClimaAtmos.

If you're interested in working on something, let us know by commenting on existing issues or 
by opening a new issue. This is to make sure no one else is working on the same issue and so 
we can help and guide you in case there is anything you need to know beforehand.

## Ground Rules

* Each pull request should consist of a logical collection of changes. You can
  include multiple bug fixes in a single pull request, but they should be related.
  For unrelated changes, please submit multiple pull requests.

* Do not commit changes to files that are irrelevant to your feature or bugfix
  (eg: `.gitignore`).

* Be willing to accept criticism and work on improving your code; we don't want
  to break other users' code, so care must be taken not to introduce bugs. We
  discuss pull requests and keep working on them until we believe we've done a
  good job.

* Be aware that the pull request review process is not immediate, and is
  generally proportional to the size of the pull request.

## Reporting a bug

The easiest way to get involved is to report issues you encounter when using
ClimaAtmos or by requesting something you think is missing.

* Head over to the [issues](https://github.com/CLiMA/ClimaAtmos.jl/issues) page.

* Search to see if your issue already exists or has even been solved previously.

* If you indeed have a new issue or request, click the "New Issue" button.

* Please be as specific as possible. Include the version of the code you were using, as
  well as what operating system you are running. The output of Julia's `versioninfo()`
  and `] status` is helpful to include. Try your best to include a complete, ["minimal working example"](https://en.wikipedia.org/wiki/Minimal_working_example) that reproduces the issue.

## Setting up your development environment

* Install [Julia](https://julialang.org/) on your system.

* Install `git` on your system if it is not already there (install XCode command line tools on
  a Mac or `git bash` on Windows).

* Login to your GitHub account and make a fork of the
  [ClimaAtmos repository](https://github.com/CLiMA/ClimaAtmos.jl) by
  clicking the "Fork" button.

* Clone your fork of the ClimaAtmos repository (in terminal on Mac/Linux or git shell/
  GUI on Windows) in the location you'd like to keep it.
  ```
  git clone https://github.com/your-user-name/ClimaAtmos.jl.git
  ```

* Navigate to that folder in the terminal or in Anaconda Prompt if you're on Windows.

* Connect your repository to the upstream (main project).
  ```
  git remote add ClimaAtmos https://github.com/CLiMA/ClimaAtmos.jl.git
  ```

* Create the development environment by opening Julia via `julia --project` then
  typing in `] instantiate`. This will install all the dependencies in the Project.toml
  file.

* You can test to make sure ClimaAtmos works by typing in `] test`. Doing so will run all
  the tests (and this can take a while).

Your development environment is now ready!

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

Generally, we follow the Julia conventions for documentation https://docs.julialang.org/en/v1/manual/documentation/.

Now that you've made your awesome contribution, it's time to tell the world how to use it.
Writing documentation strings is really important to make sure others use your functionality
properly. Didn't write new functions? That's fine, but be sure that the documentation for
the code you touched is still in great shape. It is not uncommon to find some strange wording
or clarification that you can take care of while you are here.

Here is an example of a docstring:

TODO: add example

You can preview how the Documentation will look like after merging by building the documentation 
locally. From the main directory of your local repository call

```
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project=docs/ -e 'using Pkg; Pkg.instantiate(); develop(PackageSpec(path=pwd()))'
JULIA_DEBUG=Documenter julia --project=docs/ docs/make.jl
```

and then open `docs/build/index.html` in your favorite browser. Providing the environment variable 
`JULIA_DEBUG=Documenter` will provide with more information in the documentation build process and
thus help figuring out a potential bug.

## Formatting

One of the tests consists in checking that the code is uniformly formatted. We
use [JuliaFormatter.jl](https://github.com/domluna/JuliaFormatter.jl) to achieve
consistent formatting. Here's how to use it:

You can install in your base environment with
``` sh
julia -e 'using Pkg; Pkg.add("JuliaFormatter")'
```
alongside your other development tools.

Then, you can format the package running:
``` julia
using JuliaFormatter; format(".")
```
or just with `format(".")` if the package is already imported.

The rules for formatting are defined in the `.JuliaFormatter.toml`.

If you are used to formatting from the command line instead of the REPL, you can
install `JuliaFormatter` in your base environment and call
``` sh
julia -e 'using JuliaFormatter; format(".")'
```
You could also define a shell alias
``` sh
alias julia_format_here="julia -e 'using JuliaFormatter; format(\".\")'"
```

!!! note

In the past, `ClimaAtmos` used to have a `.dev/climaformat.jl` script. We moved
away from it to reduce complexity in our repository and to align with the
general tools used by the Julia community. If you are still using
`climaformat.jl`, migrate to `JuliaFormatter` (`climaformat.jl` was just a
wrapper around `JuliaFormatter`).

## Updating environments

The repository for `ClimaAtmos` includes several checked `Manifests.toml`. This
is to help with reproducing results.
[PkgDevTools](https://github.com/CliMA/PkgDevTools.jl) provides a convenient
system to quickly update all the `Manifests.toml`. Please, refer to the
documentation for more information.

!!! note

In the past, `ClimaAtmos` used to have a `.dev/up_deps.jl` script. We moved away
from it because `PkgDevTools` provides a much simpler and more efficient way to
accomplish the same result.


## Credits

This contributor's guide is heavily based on the excellent [Oceananigans.jl contributor's guide](https://clima.github.io/OceananigansDocumentation/stable/contributing/) which is heavily based on the excellent [MetPy contributor's guide](https://github.com/Unidata/MetPy/blob/master/CONTRIBUTING.md).
