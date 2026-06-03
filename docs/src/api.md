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

## Surface

See the [Surface Conditions](@ref "Surface Conditions") page for a guide to
these types and how to choose among them.

```@docs
ClimaAtmos.AtmosSurface
```

### Flux schemes

```@docs
ClimaAtmos.SurfaceConditions.SurfaceParameterization
ClimaAtmos.SurfaceConditions.MoninObukhov
ClimaAtmos.SurfaceConditions.ExchangeCoefficients
ClimaAtmos.SurfaceConditions.HeatFluxes
ClimaAtmos.SurfaceConditions.θAndQFluxes
ClimaAtmos.SurfaceConditions.DefaultMoninObukhov
ClimaAtmos.SurfaceConditions.DefaultExchangeCoefficients
```

### Surface temperature

```@docs
ClimaAtmos.SurfaceConditions.SurfaceTemperature
ClimaAtmos.SurfaceConditions.AnalyticTemperature
ClimaAtmos.SurfaceConditions.SlabOceanTemperature
ClimaAtmos.SurfaceConditions.ExternalTemperature
ClimaAtmos.SurfaceConditions.CoupledTemperature
```

### Boundary overrides

```@docs
ClimaAtmos.SurfaceConditions.SurfaceBoundaryOverrides
```

### Surface albedo

```@docs
ClimaAtmos.SurfaceAlbedoModel
ClimaAtmos.ConstantAlbedo
ClimaAtmos.RegressionFunctionAlbedo
ClimaAtmos.CouplerAlbedo
```

### Core functions

```@docs
ClimaAtmos.SurfaceConditions.update_surface_conditions!
ClimaAtmos.SurfaceConditions.surface_state_to_conditions
ClimaAtmos.SurfaceConditions.atmos_surface_conditions
```

## Internals

```@docs
ClimaAtmos.parallel_lu_factorize!
ClimaAtmos.parallel_lu_solve!
```
