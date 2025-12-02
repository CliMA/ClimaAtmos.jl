using NCDatasets
using Dates
import ClimaInterpolations.Interpolation1D: interpolate1d!, Linear, Flat
import ..parse_date


"""
    weather_model_data_path(
        start_date,
        target_levels,
        era5_initial_condition_dir=nothing;
        kwargs...
    )

Get the path to the weather model data for a given start date and time.
If the data is not found, will attempt to generate it from raw data. If
the raw data is not found, throw an error.

Args:
- `start_date`: Start date as string yyyymmdd or yyyymmdd-HHMM
- `target_levels`: Vector of target altitude levels (in meters); ignored when `interp3d=true`
- `era5_initial_condition_dir`: Optional directory containing preprocessed ERA5

Additional keywords:
- All keyword arguments are forwarded to `to_z_levels`, e.g. `interp3d`, `grid_lon`,
  `grid_lat`, `use_custom_z`, `z_min`, `z_max`, `z_dz`, `interp_w`, etc.
"""

function weather_model_data_path(
    start_date,
    target_levels,
    era5_initial_condition_dir = nothing;
    kwargs...
)
    # Parse the date using the existing parse_date function
    dt = parse_date(start_date)

    # Extract components for filename generation
    start_date_str = Dates.format(dt, "yyyymmdd")
    start_time = Dates.format(dt, "HHMM") # Note: this is not the same as `start_time` in the coupler!

    # Pull switching flag from kwargs (default: false)
    interp3d = get(kwargs, :interp3d, false)
    grid_file = get(kwargs, :grid_file, nothing)

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
            @clima_artifact("weather_model_ic"),
            "init",
            "era5_init_$(start_date_str)_$(start_time).nc",
        )
        return ic_data_path
    end

    @info "Interpolating raw weather model data using to_z_levels (kwargs forwarded)" (dest = ic_data_path)
    to_z_levels(
        raw_data_path,
        ic_data_path,
        target_levels,
        Float32;
        kwargs...,
    )
    return ic_data_path
end

"""
    to_z_levels(
        source_file,
        target_file,
        target_levels,
        FT;
        interp3d=false,
        grid_file=nothing,
        grid_lon=nothing,
        grid_lat=nothing,
        grid_z_phys=nothing,
        use_custom_z=false, z_min=0.0, z_max=48000.0, z_dz=100.0,
        interp_w=false,
    )

Interpolate ERA5 data from native model levels to specified z-levels or to a
prescribed 3D grid.

By default (`interp3d=false`), this performs column-wise 1D interpolation in z
onto `target_levels`. In 3D mode (`interp3d=true`), fields are interpolated in
3D (lon, lat, p) using `Interpolations.jl`, with a per-column log(p)-vs-z mapping.

Args:
- source_file::String: Input ERA5 NetCDF path
- target_file::String: Output NetCDF path
- target_levels::AbstractVector: Target 1D z-levels (used for 1D mode)
- FT::Type{<:AbstractFloat}: Floating-point element type

Keywords:
- interp3d::Bool=false: Enable 3D interpolation onto a target grid
- grid_file::Union{Nothing,String}=nothing: NetCDF grid file
- grid_lon::Union{Nothing,AbstractVector}=nothing: 1D longitudes (degrees_east)
- grid_lat::Union{Nothing,AbstractVector}=nothing: 1D latitudes (degrees_north)
- grid_z_phys::Union{Nothing,AbstractArray}=nothing: Optional physical altitude array (z, lat, lon)
- use_custom_z::Bool=false: If true, build z from [z_min:z_dz:z_max] and broadcast
- z_min::Real=0.0, z_max::Real=48000.0, z_dz::Real=100.0: Custom z grid bounds/spacing
- interp_w::Bool=false: If false, write w=0 everywhere; if true, interpolate w

3D grid source:
- Option A: `grid_file` pointing to a NetCDF with `lon(lon)`, `lat(lat)`, `z_reference(z)`, and `z_physical(z,lat,lon)`.
- Option B: arrays mode using `grid_lon`, `grid_lat`, and optionally `grid_z_phys` (z, lat, lon).

Requirements:
- When interp3d=true, provide exactly one of: grid_file OR (grid_lon and grid_lat).
- In arrays mode with use_custom_z=false, also provide grid_z_phys shaped (z, lat, lon).
"""
function to_z_levels(
    source_file,
    target_file,
    target_levels,
    FT;
    interp3d::Bool = false,
    grid_file::Union{Nothing, String} = nothing,
    grid_lon::Union{Nothing, AbstractVector} = nothing,
    grid_lat::Union{Nothing, AbstractVector} = nothing,
    grid_z_phys::Union{Nothing, AbstractArray} = nothing,
    use_custom_z::Bool=false, z_min::Real=0.0, z_max::Real=48000.0, z_dz::Real=100.0,
    interp_w::Bool=false,
)

    if interp3d
        arrays_mode = grid_lon !== nothing && grid_lat !== nothing
        file_mode = grid_file !== nothing
        @assert xor(arrays_mode, file_mode) "Provide exactly one of: grid_file OR (grid_lon and grid_lat)."
        return to_z_levels_3d(
            source_file,
            target_file,
            FT,
            grid_file;
            grid_lon=grid_lon,
            grid_lat=grid_lat,
            grid_z_phys=grid_z_phys,
            use_custom_z=use_custom_z, z_min=z_min, z_max=z_max, z_dz=z_dz,
            interp_w=interp_w,
        )
    end

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
    to_z_levels_3d(
        source_file,
        target_file,
        FT,
        grid_file;
        grid_lon=nothing,
        grid_lat=nothing,
        grid_z_phys=nothing,
        use_custom_z=false, z_min=0.0, z_max=48000.0, z_dz=100.0,
        interp_w=false,
    )

Interpolate ERA5 data directly onto a (lon, lat, z) target grid.

Args:
- `source_file::String`: Input ERA5 NetCDF path
- `target_file::String`: Output NetCDF path
- `FT::Type{<:AbstractFloat}`: Floating-point element type
- `grid_file::Union{Nothing,String}`: NetCDF grid file with variables:
  - `lon(lon)`, `lat(lat)` – 1D horizontal grid
  - `z_reference(z)` – vertical index
  - `z_physical(z,lat,lon)` – physical altitude in meters

Keywords:
- `grid_lon::Union{Nothing,AbstractVector}`: 1D longitudes (degrees_east)
- `grid_lat::Union{Nothing,AbstractVector}`: 1D latitudes (degrees_north)
- `grid_z_phys::Union{Nothing,AbstractArray}`: Optional physical altitude with shape `(z, lat, lon)`
- `use_custom_z::Bool=false`: If true, build z from `[z_min:z_dz:z_max]` and broadcast
- `z_min::Real=0.0`, `z_max::Real=48000.0`, `z_dz::Real=100.0`
- `interp_w::Bool=false`: If false, write `w=0`; if true, interpolate `w`

Requirements:
- Provide exactly one of: `grid_file` OR `(grid_lon and grid_lat)`.
- In arrays mode with `use_custom_z=false`, you must also provide `grid_z_phys` with shape `(z, lat, lon)`.

Method:
For each ERA5 column, build a log(p)-vs-z mapping and assign pressure to each `(lon, lat, z_physical)`
location, then perform trilinear interpolation in `(lon, lat, p)` using `Interpolations.jl` for 3D fields.
"""
function to_z_levels_3d(
    source_file,
    target_file,
    FT,
    grid_file::Union{Nothing, String};
    grid_lon::Union{Nothing, AbstractVector} = nothing,
    grid_lat::Union{Nothing, AbstractVector} = nothing,
    grid_z_phys::Union{Nothing, AbstractArray} = nothing,
    use_custom_z::Bool=false, z_min::Real=0.0, z_max::Real=48000.0, z_dz::Real=100.0,
    interp_w::Bool=false,
)

    param_set = TD.Parameters.ThermodynamicsParameters(FT)
    grav = TD.Parameters.grav(param_set)
    Γ = FT(6.5e-3)      # lapse rate [K m^-1] used for simple hypsometric extrapolation
    R_d = FT(287.0)     # dry-air gas constant [J kg^-1 K^-1] for hypsometric extrapolation

    # Open source ERA5 data
    ncin = Dataset(source_file)

    # Basic sanity checks
    in_dims = ["pressure_level", "latitude", "longitude", "valid_time"]
    @assert all(map(x -> x in (keys(ncin)), in_dims)) "Source file $source_file is missing subset of the required dimensions: $in_dims"

    req_vars = ["u", "v", "w", "t", "q", "skt", "sp"]
    opt_vars = ["crwc", "cswc", "clwc", "ciwc"]
    @assert all(map(x -> x in (keys(ncin)), req_vars)) "Source file $source_file is missing subset of the required variables: $req_vars"

    # Source coordinates
    lat_src = FT.(ncin["latitude"][:])
    lon_src = FT.(ncin["longitude"][:])
    plevs_pa = FT.(ncin["pressure_level"][:]) .* FT(100) # [Pa]
    logp_src = FT.(log.(plevs_pa))

    # geopotential -> physical height
    z_raw = ncin["z"][:, :, :, 1]
    source_z = FT.(coalesce.(z_raw, NaN)) ./ grav # (lon, lat, plev)
    nx_src, ny_src, nz_src = size(source_z)

    # Temperature field for near-surface lapse-rate-based extrapolation
    t_raw = ncin["t"][:, :, :, 1]
    t_src = FT.(coalesce.(t_raw, NaN)) # (lon, lat, plev)

    # Determine target grid via either grid_file (existing) or grid arrays
    arrays_mode = grid_lon !== nothing && grid_lat !== nothing
    file_mode = grid_file !== nothing
    @assert xor(arrays_mode, file_mode) "Provide exactly one of: grid_file OR (grid_lon and grid_lat)."

    if file_mode
        # Open target 3D grid from file
        ncgrid = Dataset(grid_file)
        lon_t = FT.(ncgrid["lon"][:])
        lat_t = FT.(ncgrid["lat"][:])
        # Determine target vertical grid
        z_ref = if use_custom_z
            z_start = FT(z_min)
            z_stop = FT(z_max)
            z_step = FT(z_dz)
            collect(z_start:z_step:z_stop)
        else
            FT.(ncgrid["z_reference"][:])  # from grid file
        end
    else
        # Arrays mode
        lon_t = FT.(grid_lon)
        lat_t = FT.(grid_lat)
        z_ref = if use_custom_z
            z_start = FT(z_min)
            z_stop = FT(z_max)
            z_step = FT(z_dz)
            collect(z_start:z_step:z_stop)
        else
            @assert grid_z_phys !== nothing "When not using custom z, provide grid_z_phys in arrays mode."
            size(grid_z_phys, 1) == 0 && error("grid_z_phys has empty z dimension")
            collect(FT(1):FT(size(grid_z_phys, 1))) # model-level index
        end
    end
    nx_tgt = length(lon_t)
    ny_tgt = length(lat_t)
    nz_tgt = length(z_ref)

    # Determine z_physical
    z_phys = if use_custom_z
        # Broadcast the 1D altitude grid across (lat, lon)
        out = Array{FT}(undef, nz_tgt, ny_tgt, nx_tgt)
        @inbounds for j in 1:ny_tgt, i in 1:nx_tgt
            @inbounds @simd for k in 1:nz_tgt
                out[k, j, i] = z_ref[k]
            end
        end
        out
    else
        if file_mode
            # Read from grid file and permute into (z, lat, lon)
            z_phys_raw = ncgrid["z_physical"][:, :, :]
            z_phys_arr = FT.(coalesce.(z_phys_raw, NaN))
            sz = size(z_phys_arr)
            idx_z = findfirst(==(length(z_ref)), sz)
            idx_lat = findfirst(==(length(lat_t)), sz)
            idx_lon = findfirst(==(length(lon_t)), sz)
            @info "to_z_levels_3d: z_physical dims vs coords" (sz = sz, idx_z = idx_z, idx_lat = idx_lat, idx_lon = idx_lon)
            @assert idx_z !== nothing && idx_lat !== nothing && idx_lon !== nothing "Could not match z_physical dimensions to (z, lat, lon) in $grid_file"
            permutedims(z_phys_arr, (idx_z, idx_lat, idx_lon))
        else
            # Arrays mode: use provided grid_z_phys and permute if needed into (z, lat, lon)
            z_phys_arr = FT.(coalesce.(grid_z_phys, NaN))
            sz = size(z_phys_arr)
            idx_z = findfirst(==(nz_tgt), sz)
            idx_lat = findfirst(==(length(lat_t)), sz)
            idx_lon = findfirst(==(length(lon_t)), sz)
            @assert idx_z !== nothing && idx_lat !== nothing && idx_lon !== nothing "grid_z_phys dims must match (z, lat, lon) sizes"
            permutedims(z_phys_arr, (idx_z, idx_lat, idx_lon))
        end
    end # (z, lat, lon)

    # Precompute nearest source column index for each target (lon, lat)
    lon_src_idx = Array{Int}(undef, nx_tgt)
    for i in 1:nx_tgt
        _, idx = findmin(abs.(lon_src .- lon_t[i]))
        lon_src_idx[i] = idx
    end
    lat_src_idx = Array{Int}(undef, ny_tgt)
    for j in 1:ny_tgt
        _, idx = findmin(abs.(lat_src .- lat_t[j]))
        lat_src_idx[j] = idx
    end

    # For each target column, derive pressure at each `z_physical` level by
    # interpolating ERA5 log(p) vs. geometric height from the nearest ERA5 column.
    # This anchors the vertical mapping to physical height and avoids a global
    # vertical shift when the model grid's z-spacing differs from ERA5.
    p_targets = Array{FT}(undef, nz_tgt, ny_tgt, nx_tgt) # (z_ref, lat, lon) in Pa
    @inbounds for j in 1:ny_tgt
        j_src = lat_src_idx[j]
        for i in 1:nx_tgt
            i_src = lon_src_idx[i]
            # ERA5 source column heights [m] at all pressure levels
            zcol = view(source_z, i_src, j_src, :)
            # Target physical heights for this column
            z_phys_col = view(z_phys, :, j, i)
            # Interpolate log(p) along z, then exponentiate to recover p [Pa]
            dest_logp = similar(z_phys_col)
            interpolate1d!(dest_logp, zcol, z_phys_col, logp_src, Linear(), Flat())
            @inbounds for k in 1:nz_tgt
                p_targets[k, j, i] = exp(dest_logp[k])
            end
        end
    end
    # Precompute the ERA bottom-level geometric height for each target column
    idx_bottom = argmax(plevs_pa)
    z_bottom_targets = Array{FT}(undef, ny_tgt, nx_tgt) # (lat, lon)
    @inbounds for j in 1:ny_tgt
        j_src = lat_src_idx[j]
        for i in 1:nx_tgt
            i_src = lon_src_idx[i]
            z_bottom_targets[j, i] = source_z[i_src, j_src, idx_bottom]
        end
    end

    # Create output file on target 3D grid
    ncout = NCDataset(target_file, "c", attrib = copy(ncin.attrib))

    defDim(ncout, "lon", nx_tgt)
    defDim(ncout, "lat", ny_tgt)
    defDim(ncout, "z", nz_tgt)

    lon_attrib = Dict(
        "standard_name" => "longitude",
        "long_name" => "longitude",
        "units" => "degrees_east",
    )
    lon_var = defVar(ncout, "lon", FT, ("lon",), attrib = lon_attrib)
    lon_var[:] = lon_t

    lat_attrib = Dict(
        "standard_name" => "latitude",
        "long_name" => "latitude",
        "units" => "degrees_north",
    )
    lat_var = defVar(ncout, "lat", FT, ("lat",), attrib = lat_attrib)
    lat_var[:] = lat_t

    # Vertical coordinate: keep a 1D "z" that matches z_reference from the
    # grid file (model level index), so downstream tools see a standard 1D
    # vertical axis. The actual geometric height field remains available as
    # a separate 3D variable `z_physical(z,lat,lon)`.
    z_attrib = if use_custom_z
        Dict(
            "standard_name" => "altitude",
            "long_name" => "altitude",
            "units" => "m",
        )
    else
        Dict(
            "standard_name" => "model_level_index",
            "long_name" => "vertical index (matches z_reference in grid file)",
            "units" => "1",
        )
    end
    z_var = defVar(ncout, "z", FT, ("z",), attrib = z_attrib)
    z_var[:] = z_ref

    z_phys_attrib = Dict(
        "standard_name" => "altitude",
        "long_name" => "physical altitude on target 3D grid",
        "units" => "m",
        "source" => "z_physical from $grid_file",
    )
    z_phys_out = defVar(ncout, "z_physical", FT, ("z", "lat", "lon"), attrib = z_phys_attrib)
    # Current in-memory `z_phys` is (z, lat, lon) already matches ordering.
    z_phys_out[:, :, :] = z_phys

    # Helper: 3D trilinear interpolation in (lon, lat, p)
    function interpolate_3d_field!(out, src_array, lon_src, lat_src, plevs_pa, lon_t, lat_t, p_targets)
        # Interpolations.jl requires knot vectors to be unique and sorted in increasing order.
        lon_order = sortperm(lon_src)
        lat_order = sortperm(lat_src)
        p_order = sortperm(plevs_pa)
        lon_knots = lon_src[lon_order]
        lat_knots = lat_src[lat_order]
        p_knots = plevs_pa[p_order]
        src_sorted = src_array[lon_order, lat_order, p_order]
        itp = Interpolations.interpolate(
            (lon_knots, lat_knots, p_knots),
            src_sorted,
            Interpolations.Gridded(Interpolations.Linear()),
        )
        nx_t = length(lon_t)
        ny_t = length(lat_t)
        nz_t = size(p_targets, 1)
        @inbounds for j in 1:ny_t
            latv = lat_t[j]
            for i in 1:nx_t
                lonv = lon_t[i]
                pcol = view(p_targets, :, j, i)
                for k in 1:nz_t
                    out[i, j, k] = itp(lonv, latv, pcol[k])
                end
            end
        end
    end

    # Interpolate required 3D variables
    req3d = ["u", "v", "t", "q", "w"]
    # We'll hold t/q to optionally adjust below the bottom ERA level
    var_handles = Dict{String,Any}()
    t_out = Array{FT}(undef, nx_tgt, ny_tgt, nz_tgt)
    q_out = Array{FT}(undef, nx_tgt, ny_tgt, nz_tgt)
    for var_name in req3d
        var_obj = defVar(ncout, var_name, FT, ("lon", "lat", "z"), attrib = ncin[var_name].attrib)
        var_handles[var_name] = var_obj
        if var_name == "w" && !interp_w
            var_obj[:, :, :] = zeros(FT, nx_tgt, ny_tgt, nz_tgt)
        else
            src_data = FT.(coalesce.(ncin[var_name][:, :, :, 1], NaN)) # (lon, lat, plev)
            out = Array{FT}(undef, nx_tgt, ny_tgt, nz_tgt)
            interpolate_3d_field!(out, src_data, lon_src, lat_src, plevs_pa, lon_t, lat_t, p_targets)
            if var_name == "q"
                out = max.(out, FT(0))
                q_out .= out
            elseif var_name == "t"
                t_out .= out
            else
                var_obj[:, :, :] = out
            end
        end
    end

    # Compute 3D pressure on target 3D grid
    p3d_var_attrib = Dict(
        "standard_name" => "air_pressure",
        "long_name" => "air pressure on target model grid",
        "units" => "Pa",
        "source" => "ERA5 pressure levels interpolated in log(p) vs z_physical",
    )
    p3d_var = defVar(ncout, "p_3d", FT, ("lon", "lat", "z"), attrib = p3d_var_attrib)
    # For consistency with other 3D fields, also obtain pressure on the target
    # grid via trilinear interpolation in (lon, lat, p), using a synthetic
    # source field that is linear in pressure (f(lon, lat, p) = p). Because
    # the vertical interpolation in `Interpolations.jl` is linear in `p`, this
    # preserves the column-wise mapping defined by `p_targets` while applying
    # the same trilinear machinery used for u, v, t, q, etc.
    p_src_3d = repeat(reshape(plevs_pa, 1, 1, nz_src), nx_src, ny_src, 1)
    p3d_out = Array{FT}(undef, nx_tgt, ny_tgt, nz_tgt)
    interpolate_3d_field!(p3d_out, p_src_3d, lon_src, lat_src, plevs_pa, lon_t, lat_t, p_targets)
    # Below-bottom extrapolation for pressure and temperature:
    # Use a standard lapse-rate hypsometric formula for p and linear lapse-rate for T.
    # p(z) = p0 * (1 - Γ * (z - z0) / T0)^(g / (R_m Γ)), with R_m ≈ R_d * (1 + 0.608 q0)
    # Only applied where z_target < z0.
    if haskey(var_handles, "t") && haskey(var_handles, "q")
        p0_bottom = plevs_pa[idx_bottom]
        @inbounds for j in 1:ny_tgt
            z0 = z_bottom_targets[j, 1] # dummy init
            for i in 1:nx_tgt
                z0 = z_bottom_targets[j, i]
                # find if we have any levels below bottom
                # find first k with z_phys < z0
                k_first_below = 0
                for k in 1:nz_tgt
                    if z_phys[k, j, i] < z0
                        k_first_below = k
                        break
                    end
                end
                k_first_below == 0 && continue
                # bottom-level values from current (clamped) interpolation
                T0 = t_out[i, j, k_first_below]
                q0 = q_out[i, j, k_first_below]
                R_m_surf = R_d * (1 + FT(0.608) * q0)
                expo = grav / (R_m_surf * Γ)
                @inbounds for k in k_first_below:nz_tgt
                    zk = z_phys[k, j, i]
                    zk < z0 || break
                    Δz = zk - z0
                    base = FT(1) - Γ * (Δz) / T0
                    base = max(base, FT(1e-6))
                    # Adjust pressure upward as we go down
                    p3d_out[i, j, k] = p0_bottom * base^expo
                    # Lapse-rate adjust temperature linearly
                    t_out[i, j, k] = T0 - Γ * (Δz)
                end
            end
        end
    end
    # Write adjusted pressure and t/q to file
    p3d_var[:, :, :] = p3d_out
    if haskey(var_handles, "t")
        var_handles["t"][:, :, :] = t_out
    end
    if haskey(var_handles, "q")
        var_handles["q"][:, :, :] = q_out
    end

    # 2D surface fields horizontally interpolated, then broadcast in z
    function interpolate_2d_to_grid(var2d_src, lon_src, lat_src, lon_t, lat_t)
        lon_order = sortperm(lon_src)
        lat_order = sortperm(lat_src)
        lon_knots = lon_src[lon_order]
        lat_knots = lat_src[lat_order]
        var_sorted = var2d_src[lon_order, lat_order]
        itp2d = Interpolations.interpolate(
            (lon_knots, lat_knots),
            var_sorted,
            Interpolations.Gridded(Interpolations.Linear()),
        )
        nx_t = length(lon_t)
        ny_t = length(lat_t)
        out = Array{FT}(undef, nx_t, ny_t)
        @inbounds for j in 1:ny_t
            latv = lat_t[j]
            for i in 1:nx_t
                lonv = lon_t[i]
                out[i, j] = itp2d(lonv, latv)
            end
        end
        return out
    end

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
        data2d_src = FT.(coalesce.(ncin[src_name][:, :, 1], NaN))
        if dst_name == "z_sfc"
            data2d_src .= data2d_src ./ grav
        end
        data2d_tgt = interpolate_2d_to_grid(data2d_src, lon_src, lat_src, lon_t, lat_t)
        for k in 1:nz_tgt
            var_obj[:, :, k] = data2d_tgt
        end
    end

    # Optional cloud water content variables if available
    for var_name in opt_vars
        if haskey(ncin, var_name)
            @info "Interpolating optional variable (3D grid): $var_name"
            var_data = FT.(coalesce.(ncin[var_name][:, :, :, 1], NaN))
            var_var = defVar(
                ncout,
                var_name,
                FT,
                ("lon", "lat", "z"),
                attrib = ncin[var_name].attrib,
            )
            out = Array{FT}(undef, nx_tgt, ny_tgt, nz_tgt)
            interpolate_3d_field!(out, var_data, lon_src, lat_src, plevs_pa, lon_t, lat_t, p_targets)
            var_var[:, :, :] = out
        end
    end

    if file_mode
        close(ncgrid)
    end
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