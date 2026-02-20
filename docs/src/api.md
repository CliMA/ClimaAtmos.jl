# API

## Initial conditions

### General

```@docs
ClimaAtmos.InitialConditions.InitialCondition
ClimaAtmos.InitialConditions.IsothermalProfile
ClimaAtmos.InitialConditions.DecayingProfile
ClimaAtmos.InitialConditions.hydrostatic_pressure_profile
```

### Plane / Box

```@docs
ClimaAtmos.InitialConditions.ConstantBuoyancyFrequencyProfile
ClimaAtmos.InitialConditions.DryDensityCurrentProfile
ClimaAtmos.InitialConditions.RisingThermalBubbleProfile
```

### Sphere

```@docs
ClimaAtmos.InitialConditions.DryBaroclinicWave
ClimaAtmos.InitialConditions.MoistBaroclinicWaveWithEDMF
ClimaAtmos.InitialConditions.MoistAdiabaticProfileEDMFX
```

### Cases from literature

```@docs
ClimaAtmos.InitialConditions.GABLS
ClimaAtmos.InitialConditions.DYCOMS_RF01
ClimaAtmos.InitialConditions.DYCOMS_RF02
ClimaAtmos.InitialConditions.TRMM_LBA
ClimaAtmos.InitialConditions.Soares
ClimaAtmos.InitialConditions.RCEMIPIIProfile
```

## Helper

```@docs
ClimaAtmos.InitialConditions.ColumnInterpolatableField
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

### Internals

```@docs
ClimaAtmos.parallel_lu_factorize!
ClimaAtmos.parallel_lu_solve!
```
