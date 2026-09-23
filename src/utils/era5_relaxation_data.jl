# Preprocessing for the global ERA5 relaxation (`ERA5Relaxation`).
#
# The relaxation targets are read at runtime with a `TimeVaryingInput`, which
# expects a single global `lon`-`lat`-`z`-`time` NetCDF file (the same layout as
# the prescribed-aerosol/ozone inputs). The raw ERA5 data, however, is a
# directory of 6-hourly *pressure-level* snapshots. This file bridges the two:
# it interpolates each snapshot from pressure levels to a common set of altitude
# levels (using the geopotential for heights, exactly as the weather-model
# initial-condition path does in `to_z_levels_1d`) and concatenates the snapshots
# spanning the relaxation window along a CF `time` dimension.

import NCDatasets as NC
import Dates
import Thermodynamics as TD

"""
    era5_relaxation_snapshot_dates(start_date, window; snapshot_interval = Dates.Hour(6))

Return the ordered ERA5 snapshot `DateTime`s bracketing the relaxation window
`[start_date, start_date + window]`.

The first snapshot is `start_date` floored to the `snapshot_interval` grid and the
last is `start_date + window` ceiled to it, so time interpolation is always
bracketed on both ends. `window` is a number of seconds.
"""
function era5_relaxation_snapshot_dates(
    start_date,
    window;
    snapshot_interval = Dates.Hour(6),
)
    step_s = Dates.value(Dates.Second(snapshot_interval))
    # Floor start and ceil end onto the snapshot grid, in whole seconds since the
    # Unix epoch, so the window is always bracketed.
    epoch = Dates.DateTime(1970, 1, 1)
    t0 = Dates.value(Dates.Second(start_date - epoch))
    t1 = t0 + round(Int, window)
    first_s = fld(t0, step_s) * step_s
    last_s = cld(t1, step_s) * step_s
    return [
        epoch + Dates.Second(s) for s in first_s:step_s:last_s
    ]
end

"""
    era5_relaxation_snapshot_path(data_dir, date)

Return the path of the raw ERA5 pressure-level snapshot for `date`, following the
`era5_pressure_levels_<yyyymmdd>_<HHMM>.nc` naming convention.
"""
function era5_relaxation_snapshot_path(data_dir, date)
    stamp = Dates.format(date, "yyyymmdd_HHMM")
    return joinpath(data_dir, "era5_pressure_levels_$(stamp).nc")
end

"""
    era5_relaxation_data_path(data_dir, start_date, window, target_levels, FT;
                              context = nothing, snapshot_interval = Dates.Hour(6))

Return the path of the combined `lon`-`lat`-`z`-`time` ERA5 relaxation file for
the window `[start_date, start_date + window]`, generating it from the raw
pressure-level snapshots in `data_dir` if it does not already exist.

The generated file holds `t`, `q`, `u`, and `v` on `target_levels`, with one time
slice per snapshot from `era5_relaxation_snapshot_dates`. Generation runs on the
root rank only (guarded by `context`, a no-op without MPI) and all ranks wait on a
barrier before returning, so every rank reads the same file.
"""
function era5_relaxation_data_path(
    data_dir,
    start_date,
    window,
    target_levels,
    ::Type{FT};
    context = nothing,
    snapshot_interval = Dates.Hour(6),
) where {FT}
    isdir(data_dir) ||
        error("ERA5 relaxation data directory does not exist: $(data_dir)")

    dates = era5_relaxation_snapshot_dates(
        start_date,
        window;
        snapshot_interval,
    )
    window_hours = round(window / 3600; digits = 2)
    nz = length(target_levels)
    z_top = round(Int, maximum(target_levels))
    stamp = Dates.format(first(dates), "yyyymmdd_HHMM")
    out_path = joinpath(
        data_dir,
        "era5_relaxation_combined_$(stamp)_$(window_hours)h_z$(nz)_$(z_top).nc",
    )

    iamroot = isnothing(context) || ClimaComms.iamroot(context)
    if iamroot && !isfile(out_path)
        snapshot_paths = map(d -> era5_relaxation_snapshot_path(data_dir, d), dates)
        missing_paths = filter(!isfile, snapshot_paths)
        isempty(missing_paths) || error(
            "Missing ERA5 relaxation snapshots for window " *
            "[$(first(dates)), $(last(dates))]: $(join(missing_paths, ", ")).",
        )
        @info "Generating combined ERA5 relaxation file" out_path n_snapshots =
            length(dates) window_hours
        generate_era5_relaxation_file(
            out_path,
            snapshot_paths,
            dates,
            target_levels,
            FT,
        )
    end
    isnothing(context) || ClimaComms.barrier(context)
    return out_path
end

"""
    generate_era5_relaxation_file(out_path, snapshot_paths, dates, target_levels, FT)

Write the combined ERA5 relaxation NetCDF `out_path`.

Each raw pressure-level snapshot in `snapshot_paths` (paired with its `DateTime`
in `dates`) is interpolated from pressure levels to `target_levels` in height,
column by column, using the geopotential `z` divided by `g` as the source heights
(the same conversion as the weather-model initial condition). The temperature
`t`, specific humidity `q` (clipped at zero), and horizontal winds `u`, `v` are
written as `(lon, lat, z, time)` fields, with a CF `time` coordinate in seconds
since the Unix epoch so a `TimeVaryingInput` can anchor it to the run's
`start_date`.
"""
function generate_era5_relaxation_file(
    out_path,
    snapshot_paths,
    dates,
    target_levels,
    ::Type{FT},
) where {FT}
    param_set = TD.Parameters.ThermodynamicsParameters(FT)
    grav = TD.Parameters.grav(param_set)
    target_levels = FT.(target_levels)
    nz = length(target_levels)

    relax_vars = ("t", "q", "u", "v")

    # Read coordinates from the first snapshot; all snapshots share the grid.
    lon, lat = NC.NCDataset(first(snapshot_paths), "r") do ds
        (FT.(ds["longitude"][:]), FT.(ds["latitude"][:]))
    end
    nlon = length(lon)
    nlat = length(lat)
    ntime = length(dates)

    # Times as seconds since the Unix epoch (CF convention).
    epoch = Dates.DateTime(1970, 1, 1)
    times = [Float64(Dates.value(Dates.Second(d - epoch))) for d in dates]

    mktemp_out = out_path * ".tmp"
    NC.NCDataset(mktemp_out, "c") do ncout
        NC.defDim(ncout, "lon", nlon)
        NC.defDim(ncout, "lat", nlat)
        NC.defDim(ncout, "z", nz)
        NC.defDim(ncout, "time", ntime)

        lon_var = NC.defVar(
            ncout,
            "lon",
            FT,
            ("lon",);
            attrib = Dict(
                "standard_name" => "longitude",
                "long_name" => "longitude",
                "units" => "degrees_east",
            ),
        )
        lon_var[:] = lon
        lat_var = NC.defVar(
            ncout,
            "lat",
            FT,
            ("lat",);
            attrib = Dict(
                "standard_name" => "latitude",
                "long_name" => "latitude",
                "units" => "degrees_north",
                "stored_direction" => "decreasing",
            ),
        )
        lat_var[:] = lat
        z_var = NC.defVar(
            ncout,
            "z",
            FT,
            ("z",);
            attrib = Dict(
                "standard_name" => "altitude",
                "long_name" => "altitude",
                "units" => "m",
            ),
        )
        z_var[:] = target_levels
        time_var = NC.defVar(
            ncout,
            "time",
            Float64,
            ("time",);
            attrib = Dict(
                "standard_name" => "time",
                "long_name" => "time",
                "units" => "seconds since 1970-01-01",
                "calendar" => "proleptic_gregorian",
            ),
        )
        time_var[:] = times

        out_vars = Dict(
            name => NC.defVar(
                ncout,
                name,
                FT,
                ("lon", "lat", "z", "time");
                attrib = Dict("long_name" => name),
            ) for name in relax_vars
        )

        for (it, path) in enumerate(snapshot_paths)
            NC.NCDataset(path, "r") do ds
                # Source heights [m] from geopotential (lon, lat, plev).
                source_z =
                    FT.(coalesce.(ds["z"][:, :, :, 1], NaN)) ./ FT(grav)
                for name in relax_vars
                    data = interpz_3d(
                        target_levels,
                        source_z,
                        FT.(coalesce.(ds[name][:, :, :, 1], NaN)),
                    )
                    name == "q" && (data = max.(data, FT(0)))
                    out_vars[name][:, :, :, it] = data
                end
            end
        end
    end
    Base.Filesystem.mv(mktemp_out, out_path; force = true)
    return out_path
end
