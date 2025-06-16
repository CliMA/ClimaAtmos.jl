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
ClimaAtmos.InitialConditions.Nieuwstadt
ClimaAtmos.InitialConditions.GABLS
ClimaAtmos.InitialConditions.GATE_III
ClimaAtmos.InitialConditions.ARM_SGP
ClimaAtmos.InitialConditions.DYCOMS_RF01
ClimaAtmos.InitialConditions.DYCOMS_RF02
ClimaAtmos.InitialConditions.Rico
ClimaAtmos.InitialConditions.TRMM_LBA
ClimaAtmos.InitialConditions.LifeCycleTan2018
ClimaAtmos.InitialConditions.Bomex
ClimaAtmos.InitialConditions.Soares
```

### Helper

```@docs
ClimaAtmos.InitialConditions.ColumnInterpolatableField
```

## Jacobian

```@docs
ClimaAtmos.Jacobian
ClimaAtmos.JacobianAlgorithm
ClimaAtmos.ManualSparseJacobian
ClimaAtmos.AutoDenseJacobian
```

### Internals

```@docs
ClimaAtmos.parallel_lu_factorize!
ClimaAtmos.parallel_lu_solve!
```
