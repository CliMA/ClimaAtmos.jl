using NCDatasets


function weather_model_data_path(start_date, start_time)
    # get data path
    rootdir = "/home/jschmitt/experiments/ecmwf/initial_conditions"
    ic_data_path = joinpath(rootdir, "era5_init_$(start_date)_$(start_time).nc")
    source_path = joinpath(rootdir, "era5_raw_$(start_date)_$(start_time).nc")
    if !isfile(ic_data_path)
        # process using utilities
        # TODO: implement
        if !isfile(source_path)
            error("Source file $(source_path) does not exist")
        end
        @info "Interpolating raw weather model data onto z-levels"
        # interpolate raw data onto z-levels
        # save to target path
        # return target path
        error("Interpolation from raw not automated yet - see to_z_levels")
        return target_path
    end
    return ic_data_path
end


"""
    to_z_levels(source_file, target_file, target_levels, FT)

Interpolate ERA5 data from native model levels to specified z-levels. Note that 
to match MoistFromFile, we rename surface pressure, sp, to p. This should change 
for consistency if we write a new initial condition dispatch.

# Arguments
- `source_file` / `target_file`: Path to input / output NetCDF 
- `target_levels`: Vector of target altitude levels (in meters)

"""
function to_z_levels(source_file, target_file, target_levels, FT)

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
    defDim(ncout, "lon", size(lon, 1))
    defDim(ncout, "lat", size(lat, 1))
    defDim(ncout, "z", length(target_levels))
    
    # Define coordinate variables with clean attributes
    lon_attrib = OrderedDict(
        "Datatype" => string(FT),
        "standard_name" => "longitude",
        "long_name" => "longitude",
        "units" => "degrees_east",
    )
    lon_var = defVar(ncout, "lon", FT, ("lon",), attrib = lon_attrib)
    lon_var[:] = lon
    
    lat_attrib = OrderedDict(
        "Datatype" => string(FT),
        "standard_name" => "latitude",
        "long_name" => "latitude",
        "units" => "degrees_north",
        "stored_direction" => "decreasing"
    )
    lat_var = defVar(ncout, "lat", FT, ("lat",), attrib = lat_attrib)
    lat_var[:] = lat
    
    z_attrib = OrderedDict(
        "Datatype" => string(FT),
        "standard_name" => "altitude",
        "long_name" => "altitude",
        "units" => "m",
    )
    z_var = defVar(ncout, "z", FT, ("z",), attrib = z_attrib)
    z_var[:] = target_levels
    
    # Interpolate and write 3D variables
    u_var = defVar(ncout, "u", FT, ("lon", "lat", "z",), attrib = ncin["u"].attrib)
    u_var[:, :, :] = interpz_3d(target_levels, source_z, FT.(ncin["u"][:, :, :, 1]))
    
    v_var = defVar(ncout, "v", FT, ("lon", "lat", "z",), attrib = ncin["v"].attrib)
    v_var[:, :, :] = interpz_3d(target_levels, source_z, FT.(ncin["v"][:, :, :, 1]))
    
    w_var = defVar(ncout, "w", FT, ("lon", "lat", "z",), attrib = ncin["w"].attrib)
    w_var[:, :, :] = interpz_3d(target_levels, source_z, FT.(ncin["w"][:, :, :, 1]))
    
    t_var = defVar(ncout, "t", FT, ("lon", "lat", "z",), attrib = ncin["t"].attrib)
    t_var[:, :, :] = interpz_3d(target_levels, source_z, FT.(ncin["t"][:, :, :, 1]))
    
    q_var = defVar(ncout, "q", FT, ("lon", "lat", "z",), attrib = ncin["q"].attrib)
    q_var[:, :, :] = max.(interpz_3d(target_levels, source_z, FT.(ncin["q"][:, :, :, 1])), FT(0))
    
    # Write 2D surface variables - extend to all levels (TODO: accept 2D variables in atmos)
    # Simply repeat the surface values for all levels
    skt_var = defVar(ncout, "skt", FT, ("lon", "lat", "z",), attrib = ncin["skt"].attrib)
    for k in 1:length(target_levels)
        skt_var[:, :, k] = FT.(ncin["skt"][:, :, 1])
    end
    
    # TODO: rename p to sp in MoistFromFile dispatch
    sp_var = defVar(ncout, "p", FT, ("lon", "lat", "z",), attrib = ncin["sp"].attrib)
    for k in 1:length(target_levels)
        sp_var[:, :, k] = FT.(ncin["sp"][:, :, 1])
    end
    
    # Close files
    close(ncin)
    close(ncout)
end


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







# """
#     WeatherModel(start_date, start_time)

# An `InitialCondition` that initializes the model with an empty state, and then overwrites
#  it with the content of the weather model. We assume that the weather initial condition 
#  is stored in a correctly named and formatted file in some TODO artifact. 
# """
# struct WeatherModel <: InitialCondition
#     start_date::String
#     start_time::String
# end

    # # get data path from start date and start time
    # start_date = initial_conditions.start_date
    # start_time = initial_conditions.start_time
    # # replace with artifact path
    # rootdir = "/home/jschmitt/experiments/ecmwf/initial_conditions"
    # file_path = joinpath(rootdir, "era5_init_$(start_date)_$(start_time).nc")
    # if !isfile(file_path)
    #     # process using utilities
    #     # TODO: implement
    #     error("Add utilities for processing raw weather model data")
    # end