using NCDatasets
using Dates
import ClimaInterpolations.Interpolation1D: interpolate1d!, Linear, Flat
import ..parse_date

"""
    weather_model_data_path(start_date)

Get the path to the weather model data for a given start date and time.
If the data is not found, will attempt to generate it from raw data. If 
the raw data is not found, throw an error

# Arguments
- `start_date`: Start date as string yyyymmdd or yyyymmdd-HHMM
"""
function weather_model_data_path(start_date)
    # Parse the date using the existing parse_date function
    dt = parse_date(start_date)

    # Extract components for filename generation
    start_date_str = Dates.format(dt, "yyyymmdd")
    start_time = Dates.format(dt, "HHMM")
    ic_data_path = joinpath(
        @clima_artifact("weather_model_ic"),
        "init",
        "era5_init_$(start_date_str)_$(start_time).nc",
    )
    raw_data_path = joinpath(
        @clima_artifact("weather_model_ic"),
        "raw",
        "era5_raw_$(start_date_str)_$(start_time).nc",
    )

    if !isfile(ic_data_path)
        @info "Initial condition file $ic_data_path does not exist. Attempting to generate it now..."
        if !isfile(raw_data_path)
            day = Dates.format(dt, "yyyy-mm-dd")
            time = Dates.format(dt, "HH:MM")
            error(
                "Source file $(raw_data_path) does not exist. Please run `python get_initial_conditions.py --output-dir $(@clima_artifact("weather_model_ic"))/raw --date $(day) --time $(time)` to download the data.",
            )
        end
        # get target levels - TODO: make this more flexible
        target_level_path =
            joinpath(@clima_artifact("weather_model_ic"), "target_levels.txt")
        target_levels = parse.(Float32, readlines(target_level_path))

        @info "Interpolating raw weather model data onto z-levels"
        to_z_levels(raw_data_path, ic_data_path, target_levels, Float32)
    end
    return ic_data_path
end

"""
    to_z_levels(source_file, target_file, target_levels, FT)

Interpolate ERA5 data from native model levels to specified z-levels. Note that 
to use _overwrite_initial_conditions_from_file! we rename surface pressure, sp, 
to p. This allows us to share functionality with the Dyamond setup.

# Arguments
- `source_file` / `target_file`: Path to input / output NetCDF 
- `target_levels`: Vector of target altitude levels (in meters)

"""
function to_z_levels(source_file, target_file, target_levels, FT)

    param_set = TD.Parameters.ThermodynamicsParameters(FT)
    grav = TD.Parameters.grav(param_set)

    ncin = Dataset(source_file)

    # Read and cast coordinates to FT type
    lat = FT.(ncin["latitude"][:])
    lon = FT.(ncin["longitude"][:])

    # Read and cast variables to FT type, replacing missing values with NaN
    z_raw = ncin["z"][:, :, :, 1]
    source_z = FT.(coalesce.(z_raw, NaN)) ./ grav # convert from geopotential height to height

    # Create output file
    ncout = NCDataset(target_file, "c", attrib = copy(ncin.attrib))

    # Define dimensions
    defDim(ncout, "lon", length(lon))
    defDim(ncout, "lat", length(lat))
    defDim(ncout, "z", length(target_levels))

    # Define coordinate variables with clean attributes
    lon_attrib = Dict(
        "standard_name" => "longitude",
        "long_name" => "longitude",
        "units" => "degrees_east",
    )
    lon_var = defVar(ncout, "lon", FT, ("lon",), attrib = lon_attrib)
    lon_var[:] = lon

    lat_attrib = Dict(
        "standard_name" => "latitude",
        "long_name" => "latitude",
        "units" => "degrees_north",
        "stored_direction" => "decreasing",
    )
    lat_var = defVar(ncout, "lat", FT, ("lat",), attrib = lat_attrib)
    lat_var[:] = lat

    z_attrib = Dict(
        "standard_name" => "altitude",
        "long_name" => "altitude",
        "units" => "m",
    )
    z_var = defVar(ncout, "z", FT, ("z",), attrib = z_attrib)
    z_var[:] = target_levels

    # Interpolate and write 3D variables
    u_var =
        defVar(ncout, "u", FT, ("lon", "lat", "z"), attrib = ncin["u"].attrib)
    u_var[:, :, :] =
        interpz_3d(target_levels, source_z, FT.(ncin["u"][:, :, :, 1]))

    v_var =
        defVar(ncout, "v", FT, ("lon", "lat", "z"), attrib = ncin["v"].attrib)
    v_var[:, :, :] =
        interpz_3d(target_levels, source_z, FT.(ncin["v"][:, :, :, 1]))

    # ERA5 w is from a hydrostatic model and so isn't meaningful for ClimaAtmos
    # See https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017MS001059
    w_var =
        defVar(ncout, "w", FT, ("lon", "lat", "z"), attrib = ncin["w"].attrib)
    w_var[:, :, :] = zeros(FT, length(lon), length(lat), length(target_levels))

    t_var =
        defVar(ncout, "t", FT, ("lon", "lat", "z"), attrib = ncin["t"].attrib)
    t_var[:, :, :] =
        interpz_3d(target_levels, source_z, FT.(ncin["t"][:, :, :, 1]))

    q_var =
        defVar(ncout, "q", FT, ("lon", "lat", "z"), attrib = ncin["q"].attrib)
    q_var[:, :, :] =
        max.(
            interpz_3d(target_levels, source_z, FT.(ncin["q"][:, :, :, 1])),
            FT(0),
        )

    # Write 2D surface variables - extend to all levels (TODO: accept 2D variables in atmos)
    # Simply repeat the surface values for all levels
    skt_var = defVar(
        ncout,
        "skt",
        FT,
        ("lon", "lat", "z"),
        attrib = ncin["skt"].attrib,
    )
    for k in 1:length(target_levels)
        skt_var[:, :, k] = FT.(ncin["skt"][:, :, 1])
    end

    # TODO: rename p to sp in MoistFromFile for consistency
    sp_var =
        defVar(ncout, "p", FT, ("lon", "lat", "z"), attrib = ncin["sp"].attrib)
    for k in 1:length(target_levels)
        sp_var[:, :, k] = FT.(ncin["sp"][:, :, 1])
    end

    # Close files
    close(ncin)
    close(ncout)
end

"""
    interpz_3d(ztarget, zsource, fsource)

Interpolate 3D field `fsource` from 3D source levels `zsource` to 1D target levels `ztarget`.
"""
function interpz_3d(ztarget, zsource, fsource)
    nx, ny, nz = size(zsource)
    # permute dimensions from (nx, ny, nz) to (nz, nx, ny) if needed
    ztargetp = ndims(ztarget) == 1 ? ztarget : permutedims(ztarget, (3, 1, 2))
    zsourcep = ndims(zsource) == 1 ? zsource : permutedims(zsource, (3, 1, 2))
    fsourcep = ndims(fsource) == 1 ? fsource : permutedims(fsource, (3, 1, 2))
    ftargetp = similar(fsourcep, size(ztargetp, 1), nx, ny)
    # interpolate
    interpolate1d!(ftargetp, zsourcep, ztargetp, fsourcep, Linear(), Flat())
    # permute interpolated data to initial ordering
    return permutedims(ftargetp, (2, 3, 1))
end
