# API

This page documents the types and functions a user constructs to define and run
a simulation, organized the way a model is assembled: first the simulation and
grid, then the `AtmosModel` component by component, then the numerics.

Case definitions (initial conditions and forcing) live on the
[Setups](setups.md) page; the YAML equivalents of these options are listed in
[Configuration options](configuration_options.md).

## Simulation

```@docs
ClimaAtmos.AtmosSimulation
ClimaAtmos.AtmosSimulation{FT}()
ClimaAtmos.AtmosSimulation()
ClimaAtmos.AtmosSimulation(::ClimaAtmos.AtmosConfig)
ClimaAtmos.AtmosConfig
ClimaAtmos.AtmosConfig(::String)
ClimaAtmos.AtmosConfig(::AbstractDict)
ClimaAtmos.solve_atmos!
ClimaAtmos.get_simulation
ClimaAtmos.AtmosSolveResults
```

## Presets

```@docs
ClimaAtmos.Presets.dry
ClimaAtmos.Presets.equil_moist_0m
ClimaAtmos.Presets.nonequil_moist_1m
ClimaAtmos.Presets.prognostic_edmf
ClimaAtmos.Presets.prognostic_edmf_1m
ClimaAtmos.Presets.aquaplanet
ClimaAtmos.Presets.baroclinic_wave
ClimaAtmos.Presets.bomex
```

## Grids

```@docs
ClimaAtmos.SphereGrid
ClimaAtmos.ColumnGrid
ClimaAtmos.BoxGrid
ClimaAtmos.PlaneGrid
```

## Topography

```@docs
ClimaAtmos.AbstractTopography
ClimaAtmos.NoTopography
ClimaAtmos.EarthTopography
ClimaAtmos.CosineTopography
ClimaAtmos.AgnesiTopography
ClimaAtmos.ScharTopography
ClimaAtmos.DCMIP200Topography
ClimaAtmos.Hughes2023Topography
```

Mesh warping determines how the vertical coordinate is deformed to follow the
terrain:

```@docs
ClimaAtmos.MeshWarpType
ClimaAtmos.LinearWarp
ClimaAtmos.SLEVEWarp
```

## The atmosphere model

`AtmosModel` holds the physics configuration. Its components are grouped into
the structs below; keyword arguments may be passed either to the group or
directly to `AtmosModel`, which routes them to the right group.

```@docs
ClimaAtmos.AtmosModel
ClimaAtmos.AtmosModel()
ClimaAtmos.AtmosWater
ClimaAtmos.AtmosTurbconv
ClimaAtmos.AtmosRadiation
ClimaAtmos.AtmosSurface
ClimaAtmos.AtmosSponge
ClimaAtmos.AtmosGravityWave
ClimaAtmos.AtmosChem
ClimaAtmos.AtmosNumerics
ClimaAtmos.AtmosNumerics()
```

### Water and microphysics

```@docs
ClimaAtmos.AbstractMicrophysicsModel
ClimaAtmos.DryModel
ClimaAtmos.EquilibriumMicrophysics0M
ClimaAtmos.NonEquilibriumMicrophysics1M
ClimaAtmos.NonEquilibriumMicrophysics2M
ClimaAtmos.NonEquilibriumMicrophysics2MP3
```

Sedimentation and tracer positivity:

```@docs
ClimaAtmos.AbstractTerminalVelocityMode
ClimaAtmos.DiagnosticTerminalVelocity
ClimaAtmos.FixedTerminalVelocity
ClimaAtmos.TracerNonnegativityMethod
ClimaAtmos.TracerNonnegativityElementConstraint
ClimaAtmos.TracerNonnegativityVaporConstraint
ClimaAtmos.TracerNonnegativityVaporTendency
ClimaAtmos.TracerNonnegativityVerticalWaterBorrowing
```

### Cloud fraction

```@docs
ClimaAtmos.AbstractCloudModel
ClimaAtmos.GridScaleCloud
ClimaAtmos.QuadratureCloud
ClimaAtmos.MLCloud
ClimaAtmos.AbstractSGSamplingType
ClimaAtmos.SGSMean
ClimaAtmos.SGSQuadrature
ClimaAtmos.AbstractSGSDistribution
ClimaAtmos.GridMeanSGS
ClimaAtmos.GaussianSGS
ClimaAtmos.LogNormalSGS
ClimaAtmos.AbstractPhysicalPointTransform
ClimaAtmos.GridMeanPhysicalPointTransform
ClimaAtmos.GaussianPhysicalPointTransform
ClimaAtmos.LogNormalPhysicalPointTransform
ClimaAtmos.create_physical_transform
ClimaAtmos.integrate_over_sgs
```

### Turbulence and convection (PROPHET)

The turbulence and convection scheme, called EDMFX in the code; see the
[PROPHET equations](edmf_equations.md).

```@docs
ClimaAtmos.AbstractEDMF
ClimaAtmos.EDOnlyEDMFX
ClimaAtmos.PrognosticEDMFX
ClimaAtmos.PrognosticEDMFX{FT}()
ClimaAtmos.EDMFXModel
ClimaAtmos.EDMFXModel()
```

Entrainment and detrainment closures:

```@docs
ClimaAtmos.AbstractEntrainmentModel
ClimaAtmos.NoEntrainment
ClimaAtmos.PiGroupsEntrainment
ClimaAtmos.InvZEntrainment
ClimaAtmos.AbstractDetrainmentModel
ClimaAtmos.NoDetrainment
ClimaAtmos.BuoyancyVelocityDetrainment
ClimaAtmos.PiGroupsDetrainment
ClimaAtmos.SmoothAreaDetrainment
```

Buoyancy gradients, mixing-length blending, and tendency selection:

```@docs
ClimaAtmos.AbstractEnvBuoyGradClosure
ClimaAtmos.BuoyGradMean
ClimaAtmos.AbstractScaleBlendingMethod
ClimaAtmos.SmoothMinimumBlending
ClimaAtmos.HardMinimumBlending
ClimaAtmos.AbstractTendencyModel
ClimaAtmos.UseAllTendency
ClimaAtmos.NoGridScaleTendency
ClimaAtmos.NoSubgridScaleTendency
```

### Radiation

See the [Radiation](radiation.md) page for an overview of the RRTMGP coupling.

```@docs
ClimaAtmos.AbstractCloudInRadiation
ClimaAtmos.InteractiveCloudInRadiation
ClimaAtmos.PrescribedCloudInRadiation
ClimaAtmos.RadiationDYCOMS
ClimaAtmos.RadiationISDAC
ClimaAtmos.RadiationTRMM_LBA
```

Insolation at the top of the atmosphere:

```@docs
ClimaAtmos.AbstractInsolation
ClimaAtmos.IdealizedInsolation
ClimaAtmos.TimeVaryingInsolation
ClimaAtmos.RCEMIPIIInsolation
ClimaAtmos.GCMDrivenInsolation
ClimaAtmos.ExternalTVInsolation
ClimaAtmos.Larcform1Insolation
```

### Surface

See the [Surface Conditions](surface_conditions.md) page for a guide to
choosing these.

```@docs
ClimaAtmos.SurfaceConditions.SurfaceParameterization
ClimaAtmos.SurfaceConditions.MoninObukhov
ClimaAtmos.SurfaceConditions.ExchangeCoefficients
ClimaAtmos.SurfaceConditions.HeatFluxes
ClimaAtmos.SurfaceConditions.θAndQFluxes
ClimaAtmos.SurfaceConditions.DefaultMoninObukhov
ClimaAtmos.SurfaceConditions.DefaultExchangeCoefficients
ClimaAtmos.SurfaceConditions.SurfaceTemperature
ClimaAtmos.SurfaceConditions.AnalyticTemperature
ClimaAtmos.SurfaceConditions.SlabOceanTemperature
ClimaAtmos.SurfaceConditions.ExternalTemperature
ClimaAtmos.SurfaceConditions.CoupledTemperature
ClimaAtmos.SurfaceConditions.SurfaceBoundaryOverrides
```

Surface albedo:

```@docs
ClimaAtmos.SurfaceAlbedoModel
ClimaAtmos.ConstantAlbedo
ClimaAtmos.RegressionFunctionAlbedo
ClimaAtmos.CouplerAlbedo
```

### Diffusion and sponges

```@docs
ClimaAtmos.AbstractVerticalDiffusion
ClimaAtmos.VerticalDiffusion
ClimaAtmos.DecayWithHeightDiffusion
ClimaAtmos.EddyViscosityModel
ClimaAtmos.SmagorinskyLilly
ClimaAtmos.AnisotropicMinimumDissipation
ClimaAtmos.ConstantHorizontalDiffusion
ClimaAtmos.SpongeModel
ClimaAtmos.RayleighSponge
ClimaAtmos.RayleighSponge(::Any)
ClimaAtmos.ViscousSponge
ClimaAtmos.ViscousSponge(::Any)
```

### Gravity-wave drag

```@docs
ClimaAtmos.AbstractGravityWave
ClimaAtmos.NonOrographicGravityWave
ClimaAtmos.OrographicGravityWave
ClimaAtmos.FullOrographicGravityWave
```

### Forcings

Forcing terms for externally driven single-column cases are documented on the
[Single Column Models](single_column.md) page.

```@docs
ClimaAtmos.AbstractForcing
ClimaAtmos.LargeScaleSubsidence
ClimaAtmos.LargeScaleAdvection
ClimaAtmos.HeldSuarezForcing
ClimaAtmos.GCMForcing
ClimaAtmos.ISDACForcing
ClimaAtmos.PrescribedFlow
ClimaAtmos.ShipwayHill2012VelocityProfile
```

### Chemistry

```@docs
ClimaAtmos.AbstractChemistryModel
ClimaAtmos.GasPhaseChem
```

## Numerics

```@docs
ClimaAtmos.AbstractTimesteppingMode
ClimaAtmos.Explicit
ClimaAtmos.Implicit
ClimaAtmos.Hyperdiffusion
ClimaAtmos.QuasiMonotoneLimiter
```

### Jacobian and the implicit solver

See the [Implicit Solver](implicit_solver.md) page for the algorithms.

```@docs
ClimaAtmos.Jacobian
ClimaAtmos.JacobianAlgorithm
ClimaAtmos.ManualSparseJacobian
ClimaAtmos.AutoDenseJacobian
ClimaAtmos.AutoSparseJacobian
ClimaAtmos.AutoSparseJacobian()
ClimaAtmos.AutoSparseJacobian(::Any)
```

## Diagnostics

```@docs
ClimaAtmos.DiagnosticsConfig
```

## Surface-condition internals

```@docs
ClimaAtmos.SurfaceConditions.update_surface_conditions!
ClimaAtmos.SurfaceConditions.surface_state_to_conditions
ClimaAtmos.SurfaceConditions.atmos_surface_conditions
```

## Column dataset formats

Data access for single-column (SCM) forcing files: the generic
[`ColumnDataset`](@ref ClimaAtmos.ColumnDatasets.ColumnDataset) handle and format
interface, the native `ClimaColumn` reader/writer, and the ARM VARANAL
converter. See the
[Column Datasets](@ref "Column Datasets") page for usage and
[Adding a Column Dataset](@ref) for the extension interface.

### Opening and reading

```@docs
ClimaAtmos.ColumnDatasets.ColumnDataset
ClimaAtmos.ColumnDatasets.open_dataset
ClimaAtmos.ColumnDatasets.has_variable
ClimaAtmos.ColumnDatasets.read_profile
ClimaAtmos.ColumnDatasets.read_series
ClimaAtmos.ColumnDatasets.read_initial_profiles
ClimaAtmos.ColumnDatasets.read_surface_series
ClimaAtmos.ColumnDatasets.height_profile
ClimaAtmos.ColumnDatasets.site_location
```

### Time coordinates and interpolation

```@docs
ClimaAtmos.ColumnDatasets.dates
ClimaAtmos.ColumnDatasets.file_time_span
ClimaAtmos.ColumnDatasets.simulation_times
ClimaAtmos.ColumnDatasets.time_index_closest
ClimaAtmos.ColumnDatasets.wraps_periodically
ClimaAtmos.ColumnDatasets.column_timevaryinginputs
ClimaAtmos.ColumnDatasets.surface_timevaryinginputs
ClimaAtmos.ColumnDatasets.time_interpolation_method
ClimaAtmos.ColumnDatasets.periodic_calendar_method
ClimaAtmos.ColumnDatasets.extrapolation_bc
ClimaAtmos.ColumnDatasets.preprocess
```

### Canonical variables and validation

```@docs
ClimaAtmos.ColumnDatasets.CANONICAL_COLUMN_VARS
ClimaAtmos.ColumnDatasets.CANONICAL_SURFACE_VARS
ClimaAtmos.ColumnDatasets.CANONICAL_IC_VARS
ClimaAtmos.ColumnDatasets.missing_forcing_variables
ClimaAtmos.ColumnDatasets.require_forcing_variables
ClimaAtmos.ColumnDatasets.validate
```

### Format interface

```@docs
ClimaAtmos.ColumnDatasets.AbstractColumnFormat
ClimaAtmos.ColumnDatasets.format_name
ClimaAtmos.ColumnDatasets.format_variable_name
```

### Formats

```@docs
ClimaAtmos.ColumnDatasets.ClimaColumnFiles.ClimaColumnFile
ClimaAtmos.ColumnDatasets.ClimaColumnFiles.CANONICAL_UNITS
ClimaAtmos.ColumnDatasets.ClimaColumnFiles.is_conforming
ClimaAtmos.ColumnDatasets.ClimaColumnFiles.write_column_forcing_file
ClimaAtmos.ColumnDatasets.VaranalFiles.to_climacolumn
```

## Modules

```@docs
ClimaAtmos.ClimaAtmos
ClimaAtmos.Parameters
ClimaAtmos.Diagnostics
ClimaAtmos.RRTMGPInterface
ClimaAtmos.AtmosArtifacts
ClimaAtmos.ColumnDatasets
ClimaAtmos.ColumnDatasets.ClimaColumnFiles
ClimaAtmos.ColumnDatasets.VaranalFiles
```

## Internals

```@docs
ClimaAtmos.parallel_lu_factorize!
ClimaAtmos.parallel_lu_solve!
```
