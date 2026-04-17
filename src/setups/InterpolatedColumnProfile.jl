"""
    InterpolatedColumnProfile

Reads vertical profiles from a time-varying external forcing NetCDF file at a
specific time index and builds 1D interpolators via `ColumnProfiles`.

This setup is used for single-column simulations initialized from ERA5 reanalysis
data with time-varying external forcing (e.g., "ReanalysisTimeVarying").

## Example
```julia
setup = InterpolatedColumnProfile("path/to/era5_forcing.nc", "20070701")
```
"""
struct InterpolatedColumnProfile{P <: ColumnProfiles}
    external_forcing_file::String
    start_date::String
    profiles::P
end

function InterpolatedColumnProfile(external_forcing_file::String, start_date::String)
    profiles = _read_column_profiles(external_forcing_file, start_date)
    return InterpolatedColumnProfile(external_forcing_file, start_date, profiles)
end

function _read_column_profiles(external_forcing_file, start_date)
    start_date_dt = Dates.DateTime(start_date, "yyyymmdd")
    z, T, u, v, q_tot, ρ = NC.NCDataset(external_forcing_file) do ds
        time_index = argmin(abs.(ds["time"][:] .- start_date_dt))
        (
            ds["z"][:],
            ds["ta"][1, 1, :, time_index],
            ds["ua"][1, 1, :, time_index],
            ds["va"][1, 1, :, time_index],
            ds["hus"][1, 1, :, time_index],
            ds["rho"][1, 1, :, time_index],
        )
    end
    return ColumnProfiles(z, T, u, v, q_tot, ρ)
end

function center_initial_condition(setup::InterpolatedColumnProfile, local_geometry, params)
    return column_profiles_ic(setup.profiles, local_geometry)
end

function surface_condition(::InterpolatedColumnProfile, params)
    FT = eltype(params)
    parameterization = MoninObukhov(; z0 = FT(1e-4))
    return SurfaceState(; parameterization)
end

insolation_model(::InterpolatedColumnProfile) = ExternalTVInsolation()
surface_temperature_model(::InterpolatedColumnProfile) = ExternalColumnInputSST()
