# API

## Simulation

```@docs
ClimaAtmos.AtmosSimulation
```

## Presets

```@docs
ClimaAtmos.Presets.aquaplanet
ClimaAtmos.Presets.baroclinic_wave
ClimaAtmos.Presets.bomex
ClimaAtmos.Presets.dry
ClimaAtmos.Presets.equil_moist_0m
ClimaAtmos.Presets.nonequil_moist_1m
ClimaAtmos.Presets.diagnostic_edmf
ClimaAtmos.Presets.prognostic_edmf
ClimaAtmos.Presets.prognostic_edmf_1m
```

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

## Diagnostics

```@docs
ClimaAtmos.DiagnosticsConfig
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

## Internals

```@docs
ClimaAtmos.parallel_lu_factorize!
ClimaAtmos.parallel_lu_solve!
```
