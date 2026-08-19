# Column Datasets

Adding a new externally-driven column case in a supported format requires no
source code: point the config at the file and the
[`ForcingFromFile`](@ref ClimaAtmos.Setups.ForcingFromFile) setup builds the
case (initial condition, external forcing, surface temperature, and insolation)
from it.

```yaml
initial_condition: "ForcingFromFile"
external_forcing_file: /path/to/my_case_forcing.nc
start_date: "20200101"
config: "column"
```

To use a forcing file with a different (analytic) initial condition, set
`external_forcing: "ForcingFromFile"` instead and keep your `initial_condition`.
When the file supplies the initial condition, it must contain the `ta`, `ua`,
`va`, `hus`, and `rho` profiles in addition to the forcing variables.

The reader uses one format: the native `ClimaColumn` schema (below),
written by the ERA5 generator and the target for hand-made case files. A file
that is not a conforming ClimaColumn file is a loud error at construction. A
stale cached file (e.g. an ERA5 forcing file written by an older version in a
different on-disk layout) is regenerated on demand from the source rather than read.

The forcing is composed from explicit per-process terms
([`HorizontalAdvection`](@ref ClimaAtmos.HorizontalAdvection),
[`VerticalFluctuation`](@ref ClimaAtmos.VerticalFluctuation),
[`Nudging`](@ref ClimaAtmos.Nudging),
[`Subsidence`](@ref ClimaAtmos.Subsidence)). The default composition is all
four. A runscript can narrow or reshape it without any YAML option:

```julia
forcing = ClimaAtmos.ExternalDrivenTVForcing(
    forcing_file;
    forcing = (ClimaAtmos.HorizontalAdvection(),),   # advection only
)
model = ClimaAtmos.AtmosModel(; external_forcing = forcing)
simulation = ClimaAtmos.AtmosSimulation{Float64}(; model, setup, grid)
```

When the same file also supplies the initial condition, pass the terms to the
setup's `forcing` slot: `ForcingFromFile(...; forcing = (...,))`.

Per-variable relaxation timescales and height-dependent masks compose as
multiple `Nudging` terms (`Nudging(:ta; timescale, mask = z -> ...)`).

Surface-temperature and insolation inputs are required only when the model uses
them
([`ExternalTemperature`](@ref ClimaAtmos.SurfaceConditions.ExternalTemperature)
needs `ts`; [`ExternalTVInsolation`](@ref ClimaAtmos.ExternalTVInsolation) needs
`coszen`/`rsdt`), so runscripts need not track those separately.

The built-in file-driven cases wire these defaults (a runscript can override any
slot):

| Case                                                                          | Large-scale forcing (default)                                                                  | Surface / insolation (default)                                                                                                                |
|:----------------------------------------------------------------------------- |:---------------------------------------------------------------------------------------------- |:--------------------------------------------------------------------------------------------------------------------------------------------- |
| `ForcingFromFile`, `ReanalysisTimeVarying` (ERA5 time-varying)                | `default_forcing_terms()`: HAdv + VertFluc + Nudge(`ta`,`hus`) + Nudge(`ua`,`va`) + Subsidence | MO (`z0 = 1e-4`); `ExternalTemperature` (file `ts`); `ExternalTVInsolation` (file `coszen`/`rsdt`)                                            |
| `ReanalysisMonthlyAveragedDiurnal` (ERA5 monthly, set via `external_forcing`) | same terms, but periodic time interpolation (repeats the one-day file)                         | MO (`z0 = 1e-4`); `ExternalTemperature`; `ExternalTVInsolation`                                                                               |
| `ARMVARANAL`                                                                  | HAdv + Nudge(`ta`,`hus`) + Nudge(`ua`,`va`) + Subsidence (no VertFluc)                         | MO (`z0 = 0.05`, `ustar = 0.28`) + `FileHeatFluxes` when `hfls`/`hfss` present; `ExternalTemperature`; `TimeVaryingInsolation` (site lat/lon) |

```@docs
ClimaAtmos.ExternalDrivenTVForcing
ClimaAtmos.AbstractForcingTerm
ClimaAtmos.HorizontalAdvection
ClimaAtmos.VerticalFluctuation
ClimaAtmos.Subsidence
ClimaAtmos.Nudging
```

## The ClimaColumn schema

A ClimaColumn file is self-describing, so the reader needs no per-file
exceptions.

  - Global attributes: `site_latitude` / `site_longitude` in degrees.
  - Dimensions: column variables are pure 1D `(z, time)` and surface variables
    are `(time,)`. `z` is height in meters, strictly ascending, with at least
    two levels. `time` is a CF time coordinate (units plus calendar).
  - Variables use CMIP short names with SI `units` attributes. Column:
    `ta` [K], `hus` [kg kg⁻¹], `ua`/`va`/`wa` [m s⁻¹], `rho` [kg m⁻³],
    `tntha`/`tntva` [K s⁻¹], `tnhusha`/`tnhusva` [kg kg⁻¹ s⁻¹]. Surface:
    `ts` [K], `hfls`/`hfss` [W m⁻², upward positive], `coszen` [1],
    `rsdt` [W m⁻²].

Constructing a [`ColumnDataset`](@ref ClimaAtmos.ColumnDatasets.ColumnDataset)
validates a native file against this schema, including exact canonical SI unit
strings, and reports all violations.
`ColumnDatasets.validate(ColumnDatasets.ClimaColumnFile(), path)` performs the
same check explicitly;
[`ClimaColumnFiles.write_column_forcing_file`](@ref ClimaAtmos.ColumnDatasets.ClimaColumnFiles.write_column_forcing_file)
is the one producer implementation, used by the ERA5 generator.

To extend this machinery (nonstandard forcing from a runscript, generating
ERA5 forcing files, or a reader for a new file format), see
[Adding a Column Dataset](extending_column_datasets.md) in the Developer
Guide.
