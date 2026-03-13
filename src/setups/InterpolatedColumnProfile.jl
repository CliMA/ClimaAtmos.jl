"""
    InterpolatedColumnProfile{F}

Reads vertical profiles from a time-varying external forcing NetCDF file at a
specific time index and builds 1D interpolators. The `center_initial_condition`
evaluates these profiles at each grid height.

This setup is used for single-column simulations initialized from ERA5 reanalysis
data with time-varying external forcing (e.g., "ReanalysisTimeVarying").

## Fields
- `T`: Temperature interpolator function `z -> T(z)`
- `u`: Zonal wind velocity interpolator function `z -> u(z)`
- `v`: Meridional wind velocity interpolator function `z -> v(z)`
- `q_tot`: Total specific humidity interpolator function `z -> q_tot(z)`
- `ρ`: Air density interpolator function `z -> ρ(z)`

## Example
```julia
setup = interpolated_column_profile("path/to/era5_forcing.nc", "20070701")
```
"""
struct InterpolatedColumnProfile{F}
    "temperature"
    T::F
    "zonal wind velocity"
    u::F
    "meridional wind velocity"
    v::F
    "total specific humidity"
    q_tot::F
    "air density"
    ρ::F
end

"""
    interpolated_column_profile(external_forcing_file, start_date)

Factory function that creates an `InterpolatedColumnProfile` setup by reading
time-varying ERA5 data from a NetCDF file.

## Arguments
- `external_forcing_file`: Path to the NetCDF file containing external forcing data
- `start_date`: Date string in format "yyyymmdd" specifying which time to use

## Returns
An `InterpolatedColumnProfile` instance with interpolators for T, u, v, q_tot, and ρ.

The function reads the file, finds the time index closest to `start_date`, extracts
vertical profiles at that time, and builds 1D interpolators for each variable.
"""
function interpolated_column_profile(external_forcing_file::String, start_date::String)
    start_date_dt = Dates.DateTime(start_date, "yyyymmdd")
    z, T, u, v, q_tot, ρ₀ = NC.NCDataset(external_forcing_file) do ds
        time_index = argmin(abs.(ds["time"][:] .- start_date_dt))
        (
            z = ds["z"][:],
            T = ds["ta"][1, 1, :, time_index],
            u = ds["ua"][1, 1, :, time_index],
            v = ds["va"][1, 1, :, time_index],
            q_tot = ds["hus"][1, 1, :, time_index],
            ρ₀ = ds["rho"][1, 1, :, time_index],
        )
    end
    T_prof, u_prof, v_prof, q_tot_prof, ρ_prof = map((T, u, v, q_tot, ρ₀)) do value
        Intp.extrapolate(
            Intp.interpolate((z,), value, Intp.Gridded(Intp.Linear())),
            Intp.Flat(),
        )
    end
    return InterpolatedColumnProfile(T_prof, u_prof, v_prof, q_tot_prof, ρ_prof)
end

function center_initial_condition(setup::InterpolatedColumnProfile, local_geometry, params)
    (; T, u, v, q_tot, ρ) = setup
    (; z) = local_geometry.coordinates
    FT = typeof(z)

    return physical_state(;
        T = FT(T(z)),
        ρ = FT(ρ(z)),
        q_tot = FT(q_tot(z)),
        u = FT(u(z)),
        v = FT(v(z)),
        tke = FT(0),
    )
end
