# API

## Grids

```@docs
ClimaAtmos.ColumnGrid
ClimaAtmos.SphereGrid
ClimaAtmos.PlaneGrid
ClimaAtmos.BoxGrid
```

## Jacobian

```@docs
ClimaAtmos.Jacobian
ClimaAtmos.JacobianAlgorithm
ClimaAtmos.ManualSparseJacobian
ClimaAtmos.AutoDenseJacobian
ClimaAtmos.AutoSparseJacobian
```

## Topography

```@docs
ClimaAtmos.CosineTopography
ClimaAtmos.AgnesiTopography
ClimaAtmos.ScharTopography
ClimaAtmos.EarthTopography
ClimaAtmos.DCMIP200Topography
ClimaAtmos.Hughes2023Topography
ClimaAtmos.SLEVEWarp
ClimaAtmos.LinearWarp
```

## Setups

See the [Setups page](setups.md) for setup types and interface documentation.

## Internals

```@docs
ClimaAtmos.parallel_lu_factorize!
ClimaAtmos.parallel_lu_solve!
```
