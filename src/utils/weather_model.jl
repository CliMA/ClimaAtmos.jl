using NCDatasets
using Dates
import ClimaInterpolations.Interpolation1D: interpolate1d!, Linear, Flat
import ..parse_date


"""
    weather_model_data_path(start_date, target_levels, era5_initial_condition_dir])

Get the path to the weather model data for a given start date and time.
If the data is not found, will attempt to generate it from raw data. If 
the raw data is not found, throw an error

# Arguments
- `start_date`: Start date as string yyyymmdd or yyyymmdd-HHMM
- `target_levels`: Vector of target altitude levels (in meters) to interpolate to
- `era5_initial_condition_dir`: Optional directory containing preprocessed ERA5

- If `era5_initial_condition_dir` is provided, use
  `era5_init_processed_internal_YYYYMMDD_0000.nc` from that directory.
- Otherwise, use the `weather_model_ic` artifact.
"""
function weather_model_data_path(
    start_date,
    target_levels,
    era5_initial_condition_dir = nothing,
)
    # Parse the date using the existing parse_date function
    dt = parse_date(start_date)

    # Extract components for filename generation
    start_date_str = Dates.format(dt, "yyyymmdd")
    start_time = Dates.format(dt, "HHMM") # Note: this is not the same as `start_time` in the coupler!

    # If user provided a directory with preprocessed initial conditions, use it
    if !isnothing(era5_initial_condition_dir)
        ic_data_path = joinpath(
            era5_initial_condition_dir,
            "era5_init_processed_internal_$(start_date_str)_0000.nc", # TODO: generalize for all times once Coupler supports HHMM specification
        )
        raw_data_path = joinpath(
            era5_initial_condition_dir,
            "era5_raw_$(start_date_str)_0000.nc",
        )
        if !isfile(ic_data_path)
            if !isfile(raw_data_path)
                error(
                    "Neither preprocessed nor raw initial condition file exist in $(era5_initial_condition_dir).  Please run `python get_initial_conditions.py` in the WeatherQuest repository to download the data.",
                )
            end
            @info "Interpolating raw weather model data onto z-levels from user-provided directory"
            to_z_levels(raw_data_path, ic_data_path, target_levels, Float32)
        else
            @info "Using existing interpolated IC file: $ic_data_path"
        end
        return ic_data_path
    end

    # Otherwise, use artifact-based paths and generate if needed
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

    return ic_data_path
end

"""
    to_z_levels(source_file, target_file, target_levels, FT)

Interpolate ERA5 data from native model levels to specified z-levels. Note that 
to use _overwrite_initial_conditions_from_file! we rename surface pressure, sp, 
to p. This allows us to share functionality with the Dyamond setup. We assert the 
variables are present in the source file, which may be modified if additional
variables are needed, e.g. for land or ocean models. `target_levels` is a vector 
of target altitude levels (in meters) to interpolate to.
"""
function to_z_levels(source_file, target_file, target_levels, FT)

    param_set = TD.Parameters.ThermodynamicsParameters(FT)
    grav = TD.Parameters.grav(param_set)

    ncin = Dataset(source_file)

    # assert ncin has correct input dimensions
    in_dims = ["pressure_level", "latitude", "longitude", "valid_time"]
    @assert all(map(x -> x in (keys(ncin)), in_dims)) "Source file $source_file is missing subset of the required dimensions: $in_dims"

    # assert ncin has required variables
    req_vars = ["u", "v", "w", "t", "q", "skt", "sp"]
    opt_vars = ["crwc", "cswc", "clwc", "ciwc"]
    @assert all(map(x -> x in (keys(ncin)), req_vars)) "Source file $source_file is missing subset of the required variables: $req_vars"

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

    # Interpolate and write required 3D variables via loop
    # ERA5 w is from a hydrostatic model and so isn't meaningful for ClimaAtmos
    # See https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017MS001059
    req3d = ["u", "v", "t", "q", "w"]
    for var_name in req3d
        var_obj =
            defVar(ncout, var_name, FT, ("lon", "lat", "z"), attrib = ncin[var_name].attrib)
        if var_name == "w"
            var_obj[:, :, :] = zeros(FT, length(lon), length(lat), length(target_levels))
        else
            data = interpz_3d(target_levels, source_z, FT.(ncin[var_name][:, :, :, 1]))
            if var_name == "q"
                data = max.(data, FT(0))
            end
            var_obj[:, :, :] = data
        end
    end

    # Write 2D surface variables - extend to all levels (TODO: accept 2D variables in atmos)
    # Duplicate 2D surface field across all target vertical levels
    surf_map = Dict("skt" => "skt", "sp" => "p", "surface_geopotential" => "z_sfc")
    for (src_name, dst_name) in surf_map
        # Choose attributes; for z_sfc, set clean altitude attributes
        var_attrib = if dst_name == "z_sfc"
            Dict(
                "standard_name" => "surface_altitude",
                "long_name" => "surface altitude derived from ERA5",
                "units" => "m",
                "source_variable" => src_name,
            )
        else
            ncin[src_name].attrib
        end
        var_obj = defVar(ncout, dst_name, FT, ("lon", "lat", "z"), attrib = var_attrib)
        # Read first time slice and coalesce; follow same convention as sp (use [:, :, 1])
        data2d = FT.(coalesce.(ncin[src_name][:, :, 1], NaN))
        # Convert geopotential to meters if necessary
        if dst_name == "z_sfc"
            data2d .= data2d ./ grav
        end
        for k in 1:length(target_levels)
            var_obj[:, :, k] = data2d
        end
    end

    # Interpolate optional cloud water content variables if available
    for var_name in opt_vars
        if haskey(ncin, var_name)
            @info "Interpolating optional variable: $var_name"
            var_data = ncin[var_name][:, :, :, 1]
            var_var = defVar(
                ncout,
                var_name,
                FT,
                ("lon", "lat", "z"),
                attrib = ncin[var_name].attrib,
            )
            var_var[:, :, :] = interpz_3d(target_levels, source_z, FT.(var_data))
        end
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

    # check the size of the input arrays
    @assert ndims(ztarget) == 1 && ndims(zsource) == 3 && ndims(fsource) == 3 "Input arrays must have expected dimensions"

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
