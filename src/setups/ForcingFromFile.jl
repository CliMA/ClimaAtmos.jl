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

A vector of datasets drives a multi-column grid with one dataset per column, in
column order. The columns are placed at the datasets' `sites` (default: their
recorded site locations), and each column's initial condition is the profile
set of the dataset whose site it sits at, so two datasets at one site must read
the same data.

# Examples

```julia
setup = ForcingFromFile("path/to/era5_forcing.nc", "20070701")

# horizontal advection only
setup = ForcingFromFile(
    "path/to/forcing.nc",
    "20070701";
    forcing = (HorizontalAdvection(),),
)

# one file per column
setup = ForcingFromFile(ColumnDataset.(["site_a.nc", "site_b.nc"]), "20070701")
```
"""
struct ForcingFromFile{
    CD <: ColumnDatasets.ColumnData,
    F <: ExternalDrivenTVForcing,
    FS,
    ST,
    I,
    P,
    S <: NamedTuple,
}
    dataset::CD
    start_date::Dates.DateTime
    forcing::F
    flux_scheme::FS
    surface_temperature::ST
    insolation::I
    profiles::P
    sites::S
end

function ForcingFromFile(
    dataset::ColumnDatasets.ColumnData,
    start_date::String;
    forcing = default_forcing_terms(),
    flux_scheme = nothing,
    surface_temperature = ExternalTemperature(),
    insolation = ExternalTVInsolation(),
    sites = ColumnDatasets.column_sites(dataset),
)
    start_date_dt = parse_date(start_date)
    external_forcing =
        forcing isa ExternalDrivenTVForcing ? forcing :
        ExternalDrivenTVForcing(dataset; forcing)
    check_sites(dataset, sites)
    return ForcingFromFile(
        dataset,
        start_date_dt,
        external_forcing,
        flux_scheme,
        surface_temperature,
        insolation,
        column_profiles(dataset, start_date_dt),
        sites,
    )
end

ForcingFromFile(path::String, start_date::String; kwargs...) =
    ForcingFromFile(ColumnDatasets.ColumnDataset(path), start_date; kwargs...)

function column_profiles(dataset::ColumnDatasets.AbstractColumnData, start_date)
    prof = ColumnDatasets.read_initial_profiles(dataset, start_date)
    return ColumnProfiles(prof.z, prof.ta, prof.ua, prof.va, prof.hus, prof.rho)
end
column_profiles(datasets::AbstractVector, start_date) =
    map(d -> column_profiles(d, start_date), datasets)

check_sites(dataset, sites) = nothing
# The initial condition is matched by site, so one site cannot carry two datasets
function check_sites(datasets::AbstractVector, sites)
    by_site = Dict{Tuple{Float64, Float64}, String}()
    for (d, lat, lon) in zip(datasets, sites.latitude, sites.longitude)
        name = ColumnDatasets.source_name(d)
        isnan(lat) && error("$name records no site; pass `sites` to place its column")
        get!(by_site, (lat, lon), name) == name || error(
            "Columns at site ($lat, $lon) read different data: \
             $(by_site[(lat, lon)]) and $name",
        )
    end
end

"""
    column_sites(setup)

The sites placing the columns of a multi-column grid, as `(; latitude, longitude)`
vectors, or `nothing` when the setup does not place them.
"""
column_sites(setup) = nothing
column_sites(setup::ForcingFromFile) = setup.sites

site_profiles(profiles::ColumnProfiles, sites, coords) = profiles
function site_profiles(profiles::AbstractVector, sites, coords)
    FT = typeof(coords.lat)
    i = findfirst(
        j ->
            FT(sites.latitude[j]) == coords.lat &&
            FT(sites.longitude[j]) == coords.long,
        eachindex(profiles),
    )
    isnothing(i) && error("No forcing dataset at ($(coords.lat), $(coords.long))")
    return profiles[i]
end

center_initial_condition(setup::ForcingFromFile, local_geometry, params) =
    column_profiles_ic(
        site_profiles(setup.profiles, setup.sites, local_geometry.coordinates),
        local_geometry,
    )

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
