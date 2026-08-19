using NCDatasets
using Dates
import ClimaInterpolations.Interpolation1D: interpolate1d!, Linear, Flat
import ..parse_date


"""
    weather_model_data_path(
        start_date,
        target_levels,
        era5_initial_condition_dir = nothing;
        interp_w = false,
    )

Return the path to the ERA5-derived initial-condition file for `start_date`.

Without `era5_initial_condition_dir`, the path is taken from the
`wxquest_initial_conditions` artifact and returned without checking that the
file exists.

With `era5_initial_condition_dir`, a preprocessed 3D file in that directory is
used if present. Otherwise the raw ERA5 file is interpolated to `target_levels`
by `to_z_levels_1d` and the generated 1D file is returned. If neither
the preprocessed nor the raw file exists, an error is thrown pointing at the
WeatherQuest download script.

# Arguments

  - `start_date`: Start date, as a string `yyyymmdd` or `yyyymmdd-HHMM`, or a
    `DateTime`; parsed by `parse_date`.
  - `target_levels`: Target altitude levels for the 1D fallback [m].
  - `era5_initial_condition_dir = nothing`: Directory holding preprocessed or raw
    ERA5 files; `nothing` selects the artifact path.

# Keyword Arguments

  - `interp_w = false`: On the 1D fallback path, write `w = 0` when `false` and
    interpolate the ERA5 `w` when `true`.

# Notes

Only `HHMM = 0000` is supported on the user-directory path, because the coupler
cannot yet specify a time of day.
"""
function weather_model_data_path(
    start_date,
    target_levels,
    era5_initial_condition_dir = nothing;
    interp_w::Bool = false,
)
    # Parse the date using the existing parse_date function
    dt = parse_date(start_date)

    # Extract components for filename generation
    start_date_str = Dates.format(dt, "yyyymmdd")
    start_time = Dates.format(dt, "HHMM") # Note: this is not the same as `start_time` in the coupler!

    # Determine source/destination and whether generation is needed
    local raw_data_path::String
    local ic_data_path::String
    local generate_needed::Bool

    if !isnothing(era5_initial_condition_dir)
        # User-provided directory
        ic_data_path = joinpath(
            era5_initial_condition_dir,
            "era5_init_processed_internal_$(start_date_str)_0000.nc", # TODO: generalize for all times once Coupler supports HHMM specification
        )
        if isfile(ic_data_path)
            @info "Using existing interpolated IC file: $ic_data_path"
            return ic_data_path
        end
        raw_data_path = joinpath(
            era5_initial_condition_dir,
            "era5_raw_$(start_date_str)_0000.nc",
        )
        if !isfile(raw_data_path)
            error(
                "Neither preprocessed nor raw initial condition file exist in $(era5_initial_condition_dir).  Please run `python get_initial_conditions.py` in the WeatherQuest repository to download the data.",
            )
        end
        generate_needed = true
    else
        # Artifact-based paths
        ic_data_path = joinpath(
            @clima_artifact("wxquest_initial_conditions"),
            "era5_init_processed_internal_$(start_date_str)_$(start_time).nc",
        )
        return ic_data_path
    end

    # Fallback: generate a 1D-interpolated IC file when processed_internal file absent
    ic_data_path_1d = joinpath(
        era5_initial_condition_dir,
        "era5_init_$(start_date_str)_0000.nc",
    )
    @info "Processed 3D IC not found; falling back to 1D interpolation" (
        raw = raw_data_path,
        dest = ic_data_path_1d,
        n_target_levels = length(target_levels),
    )
    to_z_levels_1d(
        raw_data_path,
        ic_data_path_1d,
        target_levels,
        Float32;
        interp_w = interp_w,
    )
    return ic_data_path_1d
end


"""
    to_z_levels_1d(
        source_file,
        target_file,
        target_levels,
        FT;
        interp_w = false,
    )

Interpolate ERA5 pressure-level data onto a common set of altitude levels,
column by column, and write the result to a new NetCDF file.

The source file must provide the dimensions `pressure_level`, `latitude`,
`longitude`, and `valid_time`, and the variables `u`, `v`, `w`, `t`, `q`, `skt`,
and `sp`; the cloud-water variables `crwc`, `cswc`, `clwc`, and `ciwc` are
interpolated if present. Source heights come from the geopotential `z` divided
by `g`. Pressure is interpolated in `log(p)` and written as the 3D field `p_3d`.
Specific humidity is clipped at zero. Surface fields are broadcast over all
levels, because the model reader does not yet accept 2D variables.

# Arguments

  - `source_file`: Path of the input ERA5 NetCDF file.
  - `target_file`: Path of the NetCDF file to create, overwriting any existing
    one.
  - `target_levels`: Target altitude levels [m].
  - `FT`: Floating-point element type of the output.

# Keyword Arguments

  - `interp_w = false`: Write `w = 0` everywhere when `false`; interpolate the
    ERA5 `w` when `true`. ERA5 `w` comes from a hydrostatic model and is not
    meaningful for ClimaAtmos, hence the default.

The return value is unused; the result of the call is the file written to
`target_file`.
"""
function to_z_levels_1d(
    source_file,
    target_file,
    target_levels,
    FT;
    interp_w::Bool = false,
)

    param_set = TD.Parameters.ThermodynamicsParameters(FT)
    grav = TD.Parameters.grav(param_set)
    target_levels = FT.(target_levels)

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
        if var_name == "w" && !interp_w
            var_obj[:, :, :] = zeros(FT, length(lon), length(lat), length(target_levels))
        else
            data = interpz_3d(target_levels, source_z, FT.(ncin[var_name][:, :, :, 1]))
            if var_name == "q"
                data = max.(data, FT(0))
            end
            var_obj[:, :, :] = data
        end
    end

    # Compute 3D pressure on target z-levels (p_3d) by log-pressure interpolation
    # Assume ERA5 pressure levels are in hPa and convert to Pa
    plevs_pa = FT.(ncin["pressure_level"][:]) .* FT(100)
    # Prepare output var and per-column interpolation in log(p)
    p3d_var_attrib = Dict(
        "standard_name" => "air_pressure",
        "long_name" => "air pressure on model z-levels",
        "units" => "Pa",
        "source" => "ERA5 pressure levels interpolated in log(p) vs z",
    )
    p3d_var = defVar(ncout, "p_3d", FT, ("lon", "lat", "z"), attrib = p3d_var_attrib)
    nx, ny, _ = size(source_z)
    p3d = similar(source_z, FT, nx, ny, length(target_levels))
    logp_src = FT.(log.(plevs_pa))
    @inbounds for j in 1:ny, i in 1:nx
        zcol = view(source_z, i, j, :)
        dest = view(p3d, i, j, :)
        # Interpolate log(p) along z, then exponentiate
        interpolate1d!(dest, zcol, target_levels, logp_src, Linear(), Flat())
        dest .= exp.(dest)
    end
    p3d_var[:, :, :] = p3d

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

Interpolate the 3D field `fsource`, given on the 3D source heights `zsource`,
onto the 1D target heights `ztarget`.

Interpolation is linear in `z`, column by column, with flat extrapolation beyond
the source range. All three arrays are indexed `(lon, lat, level)`, and the
result has the same layout with `length(ztarget)` levels.
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
