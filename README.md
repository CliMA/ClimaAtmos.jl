<!-- Title -->
<h1 align="center">
  ClimaAtmos.jl
</h1>

<!-- description -->
<p align="center">
  <strong>Atmosphere components of the CliMA software stack.</strong>
</p>

[![docsbuild][docs-bld-img]][docs-bld-url]
[![dev][docs-dev-img]][docs-dev-url]
[![ghaci][gha-ci-img]][gha-ci-url]
[![buildkite][bk-ci-img]][bk-ci-url]
[![codecov][codecov-img]][codecov-url]
[![bors][bors-img]][bors-url]
[![discussions][discussions-img]][discussions-url]
[![col-prac][col-prac-img]][col-prac-url]

[docs-bld-img]: https://github.com/CliMA/ClimaAtmos.jl/workflows/Documentation/badge.svg
[docs-bld-url]: https://github.com/CliMA/ClimaAtmos.jl/actions?query=workflow%3ADocumentation

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://CliMA.github.io/ClimaAtmos.jl/dev/

[gha-ci-img]: https://github.com/CliMA/ClimaAtmos.jl/actions/workflows/ci.yml/badge.svg
[gha-ci-url]: https://github.com/CliMA/ClimaAtmos.jl/actions/workflows/ci.yml

[bk-ci-img]: https://badge.buildkite.com/2a31b42d67409c27660a0dcce65b49294cd9c6b9f14c12f21e.svg
[bk-ci-url]: https://buildkite.com/clima/climaatmos-ci

[codecov-img]: https://codecov.io/gh/CliMA/ClimaAtmos.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/CliMA/ClimaAtmos.jl

[bors-img]: https://bors.tech/images/badge_small.svg
[bors-url]: https://app.bors.tech/repositories/35474

[col-prac-img]: https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet?style=flat-square
[col-prac-url]: https://github.com/SciML/ColPrac

[discussions-img]: https://img.shields.io/badge/Ask%20us-anything-1abc9c.svg?style=flat-square
[discussions-url]: https://github.com/CliMA/ClimaAtmos.jl/discussions


ClimaAtmos.jl is the atmosphere components of the CliMA software stack. We strive for a user interface that makes ClimaAtmos.jl as friendly and intuitive to use as possible, allowing users to focus on the science.

## Installation instructions

Recommended Julia: Stable release v1.8.1

Download the `ClimaAtmos`
[source](https://github.com/CliMA/ClimaAtmos.jl) with:

```
$ git clone https://github.com/CliMA/ClimaAtmos.jl.git
```

Now change into the `ClimaAtmos.jl` directory with

```
$ cd ClimaAtmos.jl
```

To use ClimaAtmos, you need to instantiate all dependencies with:

```
$ julia --project
julia>]
(v1.8) pkg> instantiate
```

## Running instructions

Currently, the simulations are stored in the `test` folder. Run all the test cases with:

```
$ julia --project=test test/runtests.jl
```

## Contributing

If you're interested in contributing to the development of ClimaAtmos we want your help no matter how big or small a contribution you make! It's always great to have new people look at the code with fresh eyes: you will see errors that other developers have missed.

Let us know by [opening an issue](https://github.com/CliMA/ClimaAtmos.jl/issues/new) if you'd like to work on a new feature.

Here is the rule of thumb [coding style](https://clima.github.io/ClimateMachine.jl/latest/DevDocs/CodeStyle/) and [unicode usage restrictions](https://clima.github.io/ClimateMachine.jl/latest/DevDocs/AcceptableUnicode/).

For more information, check out our [contributor's guide](https://clima.github.io/ClimaAtmos.jl/dev/contributor_guide/).
