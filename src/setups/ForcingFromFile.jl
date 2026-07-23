"""
    ForcingFromFile

Generic file-driven single-column setup: initial condition, external forcing,
surface temperature, and insolation are sourced from one column forcing file,
read through the `ColumnDatasets` interface so any registered dataset format
works.

The initial condition reads vertical profiles (`ta`, `ua`, `va`, `hus`,
`rho`) at the file time closest to `start_date` and builds 1D interpolators
via `ColumnProfiles`. The forcing, surface, and insolation are composition
slots, each defaulting to the ERA5-case behavior:

  - `forcing`: a tuple of [`AbstractForcingTerm`](@ref ClimaAtmos.AbstractForcingTerm)s
    (or a built [`ExternalDrivenTVForcing`](@ref)). Default: `default_forcing_terms()`.
  - `flux_scheme`: the surface flux scheme. Default (`nothing`): interactive
    Monin-Obukhov. For prescribed fluxes pass e.g.
    `MoninObukhov(; z0, ustar, fluxes = SurfaceConditions.FileHeatFluxes(data, start_date))`.
  - `surface_temperature`: default `ExternalTemperature()` (the file's `ts`).
  - `insolation`: default `ExternalTVInsolation()` (the file's `coszen`/`rsdt`).
    Pass `TimeVaryingInsolation(; latitude, longitude, start_date)` for
    astronomically-computed insolation.

## Example

```julia
setup = ForcingFromFile("path/to/era5_forcing.nc", "20070701")

# horizontal advection only
setup = ForcingFromFile(
    "path/to/forcing.nc",
    "20070701";
    forcing = (HorizontalAdvection(),),
)
```
"""
struct ForcingFromFile{
    CD <: ColumnDatasets.ColumnDataset,
    F <: ExternalDrivenTVForcing,
    FS,
    ST,
    I,
    P <: ColumnProfiles,
}
    dataset::CD
    start_date::Dates.DateTime
    forcing::F
    flux_scheme::FS
    surface_temperature::ST
    insolation::I
    profiles::P
end

function ForcingFromFile(
    dataset::ColumnDatasets.ColumnDataset,
    start_date::String;
    forcing = default_forcing_terms(),
    flux_scheme = nothing,
    surface_temperature = ExternalTemperature(),
    insolation = ExternalTVInsolation(),
)
    start_date_dt = parse_date(start_date)
    external_forcing =
        forcing isa ExternalDrivenTVForcing ? forcing :
        ExternalDrivenTVForcing(dataset; forcing)
    profiles = ColumnDatasets.open_dataset(dataset) do ds
        prof = ColumnDatasets.read_initial_profiles(dataset, ds, start_date_dt)
        ColumnProfiles(prof.z, prof.ta, prof.ua, prof.va, prof.hus, prof.rho)
    end
    return ForcingFromFile(
        dataset,
        start_date_dt,
        external_forcing,
        flux_scheme,
        surface_temperature,
        insolation,
        profiles,
    )
end

ForcingFromFile(path::String, start_date::String; kwargs...) =
    ForcingFromFile(ColumnDatasets.ColumnDataset(path), start_date; kwargs...)

center_initial_condition(setup::ForcingFromFile, local_geometry, params) =
    column_profiles_ic(setup.profiles, local_geometry)

function surface_condition(setup::ForcingFromFile, params)
    FT = eltype(params)
    flux_scheme =
        isnothing(setup.flux_scheme) ? MoninObukhov(; z0 = FT(1e-4)) :
        setup.flux_scheme
    return (; flux_scheme, temperature = nothing, overrides = nothing)
end

external_forcing(setup::ForcingFromFile, ::Type) = setup.forcing

insolation_model(setup::ForcingFromFile) = setup.insolation

surface_temperature_model(setup::ForcingFromFile) = setup.surface_temperature
