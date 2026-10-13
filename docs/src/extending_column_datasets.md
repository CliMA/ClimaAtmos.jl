# Adding a Column Dataset

The [Column Datasets](column_datasets_reference.md) reference page documents
the file-driven column cases, the forcing terms, and the ClimaColumn schema.
This page covers the extension points: forcing behavior the standard terms do
not encode, generating new ERA5 forcing files, and reading a new file format.

## Nonstandard forcing behavior from a runscript

The composed terms cover the standard file-driven processes. They are not
intended to encode every forcing experiment. For a wholly new tendency term,
an in-memory data source, or state-dependent behavior, define a small forcing
type in the runscript and extend the forcing interface. This keeps the
experiment visible next to the simulation construction and does not require a
new YAML option or a change under `src/`.

```julia
import ClimaAtmos as CA
import ClimaAtmos: external_forcing_cache, external_forcing_tendency!

struct MyCaseForcing{D, M}
    data::D
    mask::M
end

function external_forcing_cache(Y, forcing::MyCaseForcing, params, start_date)
    # Allocate model-grid fields and prepare any interpolation objects here.
    # `forcing.data` may be a ColumnDataset, arrays, or another runscript-owned
    # object. The returned value is available as `p.external_forcing` below.
    return (; mask = forcing.mask)
end

function external_forcing_tendency!(Yₜ, Y, p, t, ::MyCaseForcing)
    cache = p.external_forcing
    # Evaluate the data at `t` and add this case's tendencies to Yₜ here.
    # The implementation may use Y, p.precomputed, p.params, and cache.
    return nothing
end

forcing = MyCaseForcing(case_data, case_mask)
model = CA.AtmosModel(grid; setup, external_forcing = forcing)
simulation = CA.AtmosSimulation(model)
```

The two methods are the complete interface when the tendency can evaluate its
data at the current model time. A custom forcing can reuse the standard file
reader by storing `CA.ColumnDatasets.ColumnDataset(path)` and calling the
[`ColumnDatasets.column_timevaryinginputs`](@ref ClimaAtmos.ColumnDatasets.column_timevaryinginputs)
or
[`ColumnDatasets.surface_timevaryinginputs`](@ref ClimaAtmos.ColumnDatasets.surface_timevaryinginputs)
utilities in its cache method. It can instead store arrays or callables for a
fully in-memory experiment.

If forcing data must be refreshed by a scheduled callback, also extend
`CA.default_model_callbacks(::MyCaseForcing; kwargs...)` and return the callback
tuple for that type. It will be composed with the model's other default
callbacks. Supplying `callbacks` directly to
[`AtmosSimulation`](@ref ClimaAtmos.AtmosSimulation) requires
`default_callbacks = false` and replaces the complete default callback set, so
the component extension is normally the safer hook.

## Generating ERA5 forcing data

The reanalysis-driven cases (`ReanalysisTimeVarying`,
`ReanalysisMonthlyAveragedDiurnal`; see
[Single Column Models](single_column.md)) generate their ClimaColumn forcing
files from raw ERA5 downloads with
`src/config/era5_observations_to_forcing_file.jl`. The processing script
expects three files in one directory, following the ERA5 variable naming:

 1. Hourly profiles with `t`, `q`, `u`, `v`, `w`, `z`, `clwc`, and `ciwc`,
    named `"forcing_and_cloud_hourly_profiles_$(start_date).nc"` for
    `ReanalysisTimeVarying` and `"monthly_diurnal_profiles_$(start_date).nc"`
    for `ReanalysisMonthlyAveragedDiurnal`, with `start_date` formatted
    YYYYMMDD. The `clwc` and `ciwc` profiles are calibration targets and are
    not needed to run the simulation itself.
 2. Instantaneous variables, including the surface temperature `ts`, in
    `"hourly_inst_$(start_date).nc"` (or `monthly_diurnal_inst`).
 3. Accumulated variables, including the surface fluxes `hfls` and `hfss`, in
    `"hourly_accum_$(start_date).nc"` (or `monthly_diurnal_accum`).
    Accumulated values are divided by the accumulation period: 3600 for
    hourly data, 86400 for daily and monthly data (see the
    [ERA5 documentation](https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#ERA5:datadocumentation-Monthlymeans)).

On the `clima` and Caltech HPC servers, sites already covered by the
`era5_hourly_atmos_raw` artifact (the tropical Pacific, first 5 days of July
2007) need no further setup. To run elsewhere, download the raw data from
ECMWF, place the three files in one directory, and point the artifact at it
in `~/.julia/artifacts/Overrides.toml`:

```toml
8234def2ead82e385a330a48ed2f0c030e434065 = "/path/to/raw_data_dir" # raw data
a1a465e8d237d78bef1e6d346054da395787a9f9 = "/path/to/processed_files" # processed output
```

## Adding a format module

A new format is one self-contained module under `src/column_datasets/`: define a
singleton subtype of
[`ColumnDatasets.AbstractColumnFormat`](@ref ClimaAtmos.ColumnDatasets.AbstractColumnFormat),
extend the three required methods
([`format_name`](@ref ClimaAtmos.ColumnDatasets.format_name),
[`format_variable_name`](@ref ClimaAtmos.ColumnDatasets.format_variable_name),
[`height_profile`](@ref ClimaAtmos.ColumnDatasets.height_profile)) plus
whichever overrides the format needs
([`preprocess`](@ref ClimaAtmos.ColumnDatasets.preprocess) for unit conversions
and fill values, [`dates`](@ref ClimaAtmos.ColumnDatasets.dates) for nonstandard
time axes,
[`read_profile`](@ref ClimaAtmos.ColumnDatasets.read_profile)/[`read_series`](@ref ClimaAtmos.ColumnDatasets.read_series)
for layout quirks or derived variables,
[`open_dataset`](@ref ClimaAtmos.ColumnDatasets.open_dataset) for grouped
files), and pass the singleton via the `format` keyword of
[`ColumnDatasets.ColumnDataset`](@ref ClimaAtmos.ColumnDatasets.ColumnDataset).
No changes to the forcing, setup, or config machinery are needed.
