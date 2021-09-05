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
[![codecov][codecov-img]][codecov-url]
[![bors][bors-img]][bors-url]

[docs-bld-img]: https://github.com/CliMA/ClimaAtmos.jl/workflows/Documentation/badge.svg
[docs-bld-url]: https://github.com/CliMA/ClimaAtmos.jl/actions?query=workflow%3ADocumentation

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://CliMA.github.io/ClimaAtmos.jl/dev/

[gha-ci-img]: https://github.com/CliMA/ClimaAtmos.jl/actions/workflows/ci.yml/badge.svg
[gha-ci-url]: https://github.com/CliMA/ClimaAtmos.jl/actions/workflows/ci.yml

[codecov-img]: https://codecov.io/gh/CliMA/ClimaAtmos.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/CliMA/ClimaAtmos.jl

[bors-img]: https://bors.tech/images/badge_small.svg
[bors-url]: https://app.bors.tech/repositories/35474

ClimaAtmos.jl is the atmosphere components of the CliMA software stack. We strive for a user interface that makes ClimaAtmos.jl as friendly and intuitive to use as possible, allowing users to focus on the science.

## Installation instructions

Download the `ClimaAtmos`
[source](https://github.com/CliMA/ClimaAtmos.jl) with:

```
$ git clone https://github.com/CliMA/ClimaAtmos.jl.git
```

Now change into the `ClimaAtmos.jl` directory with 

```
$ cd ClimaAtmos.jl
```

To use ClimaAtmos, you need to add the `ClimaCore` package and instantiate all dependencies with:

```
$ julia --project
julia>]
(v1.6) pkg> add https://github.com/CliMA/ClimaCore.jl.git
(v1.6) pkg> instantiate
```

## Running instructions

Currently, the simulations are stored in the `test` folder. Run all the test cases with:

```
$ julia --project test/runtests.jl
```

## Contributing

If you're interested in contributing to the development of ClimaAtmos we want your help no matter how big or small a contribution you make! It's always great to have new people look at the code with fresh eyes: you will see errors that other developers have missed.

Let us know by [opening an issue](https://github.com/CliMA/ClimaAtmos.jl/issues/new) if you'd like to work on a new feature.

For more information, check out our [contributor's guide](https://clima.github.io/ClimaAtmos.jl/dev/contributor_guide/).
