# API

## Simulation

```@docs
ClimaAtmos.AtmosSimulation
```

## AtmosModel

`AtmosModel` is the typed configuration of physics and parameterizations.
Components are organized into subsystem **group structs**; each group's
docstring lists every accepted value with its description, YAML alias, and
default.

```@docs
ClimaAtmos.AtmosModel
```

### Water and microphysics

```@docs
ClimaAtmos.AtmosWater
ClimaAtmos.DryModel
ClimaAtmos.EquilibriumMicrophysics0M
ClimaAtmos.NonEquilibriumMicrophysics1M
ClimaAtmos.NonEquilibriumMicrophysics2M
ClimaAtmos.NonEquilibriumMicrophysics2MP3
ClimaAtmos.GridScaleCloud
ClimaAtmos.QuadratureCloud
ClimaAtmos.MLCloud
ClimaAtmos.SGSQuadrature
ClimaAtmos.FixedTerminalVelocity
ClimaAtmos.DiagnosticTerminalVelocity
ClimaAtmos.TracerNonnegativityElementConstraint
ClimaAtmos.TracerNonnegativityVaporConstraint
ClimaAtmos.TracerNonnegativityVaporTendency
ClimaAtmos.TracerNonnegativityVerticalWaterBorrowing
```

### Radiation

```@docs
ClimaAtmos.AtmosRadiation
ClimaAtmos.HeldSuarezForcing
ClimaAtmos.IdealizedInsolation
ClimaAtmos.TimeVaryingInsolation
ClimaAtmos.RCEMIPIIInsolation
ClimaAtmos.GCMDrivenInsolation
ClimaAtmos.ExternalTVInsolation
ClimaAtmos.RadiationDYCOMS
ClimaAtmos.RadiationISDAC
ClimaAtmos.RadiationTRMM_LBA
```

### Turbulence and convection

```@docs
ClimaAtmos.AtmosTurbconv
ClimaAtmos.EDMFXModel
ClimaAtmos.EDOnlyEDMFX
ClimaAtmos.PrognosticEDMFX
ClimaAtmos.DiagnosticEDMFX
ClimaAtmos.SmagorinskyLilly
ClimaAtmos.AnisotropicMinimumDissipation
ClimaAtmos.ConstantHorizontalDiffusion
```

### Gravity wave drag

```@docs
ClimaAtmos.AtmosGravityWave
ClimaAtmos.NonOrographicGravityWave
ClimaAtmos.LinearOrographicGravityWave
ClimaAtmos.FullOrographicGravityWave
```

### Sponge

```@docs
ClimaAtmos.AtmosSponge
ClimaAtmos.ViscousSponge
ClimaAtmos.RayleighSponge
```

### Surface

```@docs
ClimaAtmos.AtmosSurface
ClimaAtmos.ZonallySymmetricSST
ClimaAtmos.RCEMIPIISST
ClimaAtmos.ExternalTVColumnSST
ClimaAtmos.PrescribedSST
ClimaAtmos.SlabOceanSST
ClimaAtmos.ConstantAlbedo
ClimaAtmos.RegressionFunctionAlbedo
ClimaAtmos.CouplerAlbedo
```

### Numerics

```@docs
ClimaAtmos.AtmosNumerics
ClimaAtmos.Hyperdiffusion
ClimaAtmos.QuasiMonotoneLimiter
ClimaAtmos.TestDycoreConsistency
ClimaAtmos.ReproducibleRestart
ClimaAtmos.Explicit
ClimaAtmos.Implicit
ClimaAtmos.VerticalDiffusion
ClimaAtmos.DecayWithHeightDiffusion
```

### Single-column model

```@docs
ClimaAtmos.SCMSetup
ClimaAtmos.Subsidence
ClimaAtmos.LargeScaleAdvection
ClimaAtmos.GCMForcing
ClimaAtmos.ExternalDrivenTVForcing
ClimaAtmos.ISDACForcing
ClimaAtmos.ShipwayHill2012VelocityProfile
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

## Diagnostics

```@docs
ClimaAtmos.DiagnosticsConfig
```

## Internals

```@docs
ClimaAtmos.parallel_lu_factorize!
ClimaAtmos.parallel_lu_solve!
```
