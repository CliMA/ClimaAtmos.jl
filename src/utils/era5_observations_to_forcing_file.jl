# This file converts ERA5 observational data to forcing data for ClimaAtmos
# single column model runs. For some terms we use the ERA5 data directly,
# including air temperature, specific humidity, and wind. For advective tendencies
# we need to compute gradient terms in the vertical and horizontal which is
# computed in separate helper functions. ERA5 data is available from ECMWF
# split over different files for variables available at levels, surface variables
# measured instantaneously (surface temperature), and variables that are reported
#as accumulated (sensible and latent heat fluxes), which must be divided by
# the accumulation period (typically 1 hour or 1 day in seconds).

using NCDatasets
using Statistics
using Dates

# for radiation calculation
import Insolation
import Insolation.Parameters as IP
import ClimaParams as CP

"""
    get_external_daily_forcing_file_path(parsed_args; data_dir)

Get the path to the external forcing file for a given site and start date.
When using the BUILDKITE env, a temporary directory is used for the
external forcing file. Otherwise, the file is expected to stored in the
era5_hourly_atmos_processed artifact directory.
"""
function get_external_daily_forcing_file_path(
    parsed_args;
    data_dir = get(ENV, "BUILDKITE", "") == "true" ? mktempdir() :
               joinpath(
        @clima_artifact("era5_hourly_atmos_processed"),
        "daily",
    ),
)
    start_date = parsed_args["start_date"]
    t_end = get(parsed_args, "t_end", "23hours") # generate a single day file if t_end is not specified
    end_time =
        DateTime(start_date, "yyyymmdd") + Dates.Second(time_to_seconds(t_end))
    end_date = Dates.format(end_time, "yyyymmdd")
    # round to era5 quarter degree resolution for site selection
    site_latitude = round(parsed_args["site_latitude"] * 4) / 4
    site_longitude = round(parsed_args["site_longitude"] * 4) / 4

    if site_latitude != parsed_args["site_latitude"] ||
       site_longitude != parsed_args["site_longitude"]
        @info "Rounded site latitude/longitude from ($(parsed_args["site_latitude"]), $(parsed_args["site_longitude"])) to ($(site_latitude), $(site_longitude)) for ERA5 quarter-degree resolution."
    end
    return joinpath(
        data_dir,
        "tv_forcing_$(site_latitude)_$(site_longitude)_$(start_date)_$(end_date).nc",
    )
end

"""
    get_external_monthly_forcing_file_path(parsed_args; data_dir)

Get the path to the external forcing file for a given site and start date.
When using the BUILDKITE env, a temporary directory is used for the
external forcing file. Otherwise, the file is expected to stored in the
era5_hourly_atmos_processed artifact directory.
"""
function get_external_monthly_forcing_file_path(
    parsed_args;
    data_dir = get(ENV, "BUILDKITE", "") == "true" ? mktempdir() :
               joinpath(
        @clima_artifact("era5_hourly_atmos_processed"),
        "monthly",
    ),
)
    start_date = parsed_args["start_date"]
    # round to era5 quarter degree resolution for site selection
    site_latitude = round(parsed_args["site_latitude"] * 4) / 4
    site_longitude = round(parsed_args["site_longitude"] * 4) / 4

    if site_latitude != parsed_args["site_latitude"] ||
       site_longitude != parsed_args["site_longitude"]
        @info "Rounded site latitude/longitude from ($(parsed_args["site_latitude"]), $(parsed_args["site_longitude"])) to ($(site_latitude), $(site_longitude)) for ERA5 quarter-degree resolution."
    end
    return joinpath(
        data_dir,
        "monthly_diurnal_cycle_forcing_$(site_latitude)_$(site_longitude)_$(start_date).nc",
    )
end

"""
    check_daily_forcing_times(forcing_file_path, parsed_args)

Check that the simulation start and end times are within the range of the external forcing file.
Return true if the forcing file is valid, false otherwise.
"""
function check_daily_forcing_times(forcing_file_path, parsed_args)
    start = Dates.DateTime(parsed_args["start_date"], "yyyymmdd")
    stop = start + Dates.Second(time_to_seconds(parsed_args["t_end"]))
    NCDataset(forcing_file_path) do ds
        if ds["time"][1] > start
            @warn "Start time $start is before the first time step in the forcing file"
            return false
        end
        if ds["time"][end] < stop
            @warn "End time $stop is after the last time step in the forcing file"
            return false
        end
    end
    return true
end

"""
    check_monthly_forcing_times(path, parsed_args)

Check the times for the 1 day monthly-averaged forcing file are correct. As we are using 
ClimaUtilities.TimeVaryingInputs.PeriodicCalendar we require the data to cover one day exactly. 
Return true if the forcing file is valid, false otherwise.
"""
function check_monthly_forcing_times(path, parsed_args)
    start = Dates.DateTime(parsed_args["start_date"], "yyyymmdd")
    stop = start + Dates.Day(1)
    NCDataset(path) do ds
        dt = ds["time"][2] - ds["time"][1]
        if ds["time"][1] > start
            @warn "Start time $start is before the first time step in the forcing file"
            return false
        end
        if ds["time"][end] + dt != stop
            @warn "Forcing should cover one day, following ClimaUtilities.TimeVaryingInputs.PeriodicCalendar indexing"
            return false
        end
    end
    return true
end

"""
    get_horizontal_tendencies(lat, lon_index, lat_index, column_ds, external_tv_params)

Calculate the horizontal advective tendencies for temperature and specific humidity
using a second-order finite difference approximation.
"""
function get_horizontal_tendencies(
    lat,
    lon_index,
    lat_index,
    column_ds,
    external_tv_params,
)
    rearth = external_tv_params.planet_radius
    lat_rad = deg2rad(lat)
    coslat = cos(lat_rad)

    # compute grid resolution in degrees from file for estimation of dx and dy in meters
    longitudinal_resolution = abs(
        column_ds["longitude"][lon_index + 1] -
        column_ds["longitude"][lon_index],
    )
    latitudinal_resolution = abs(
        column_ds["latitude"][lat_index + 1] - column_ds["latitude"][lat_index],
    )

    # check that the resolution is not zero and reasonable resolution (typical atmosphere observations are not coarser than 2°)
    @assert longitudinal_resolution > 0 && latitudinal_resolution > 0 "Horizontal resolution must be greater than zero."
    if longitudinal_resolution > 2 || latitudinal_resolution > 2
        @warn "Observational resolution is longitudinal: $longitudinal_resolution°, latitudinal: $latitudinal_resolution°, which is greater than or equal to 2 degrees."
    end
    # compute horizontal spacing in meters
    dx = 2 * π * rearth * coslat / 360 * longitudinal_resolution
    dy = 2 * π * rearth / 360 * latitudinal_resolution

    # get velocities at site location
    ᶜu = smooth_4D_era5(column_ds, "u", lon_index, lat_index)
    ᶜv = smooth_4D_era5(column_ds, "v", lon_index, lat_index)

    # get temperature at N S E W of center for gradient calculation
    ʷT = smooth_4D_era5(column_ds, "t", lon_index - 1, lat_index)
    ⁿT = smooth_4D_era5(column_ds, "t", lon_index, lat_index + 1)
    ˢT = smooth_4D_era5(column_ds, "t", lon_index, lat_index - 1)
    ᵉT = smooth_4D_era5(column_ds, "t", lon_index + 1, lat_index)

    # get specific humidity at N S E W of center for gradient calculation
    ʷq = smooth_4D_era5(column_ds, "q", lon_index - 1, lat_index)
    ⁿq = smooth_4D_era5(column_ds, "q", lon_index, lat_index + 1)
    ˢq = smooth_4D_era5(column_ds, "q", lon_index, lat_index - 1)
    ᵉq = smooth_4D_era5(column_ds, "q", lon_index + 1, lat_index)

    # temperature and specific humidity advective tendency at center
    tntha = -(ᶜu .* (ᵉT .- ʷT) ./ (2 * dx) .+ ᶜv .* (ⁿT .- ˢT) ./ (2 * dy))
    tnhusha = -(ᶜu .* (ᵉq .- ʷq) ./ (2 * dx) .+ ᶜv .* (ⁿq .- ˢq) ./ (2 * dy))

    return tntha, tnhusha
end

"""
    get_vertical_tendencies(sim_forcing, var)

Calculate the temperature and specific humidity vertical tendencies as a function of levels
using vertical advection using second-order finite difference at interior points and
first-order finite difference at the top and bottom levels.

This function is only used for for steady forcing, which is currently not supported.
"""
function get_vertical_tendencies(sim_forcing, var)

    deriv = zeros(size(sim_forcing["wa"]))
    num_vertical_levels = size(sim_forcing["wa"])[1]
    for i in 1:num_vertical_levels
        if i == 1 # bottom boundary
            deriv[1, :] =
                sim_forcing["wa"][1, :] .*
                (sim_forcing[var][2, :] .- sim_forcing[var][1, :]) ./
                (sim_forcing["z"][2, :] .- sim_forcing["z"][1, :])
        elseif i == size(sim_forcing["wa"])[1] # top boundary
            deriv[end, :] =
                sim_forcing["wa"][end, :] .*
                (sim_forcing[var][end, :] .- sim_forcing[var][end - 1, :]) ./
                (sim_forcing["z"][end, :] .- sim_forcing["z"][end - 1, :])
        else # centered FD
            deriv[i, :] =
                sim_forcing["wa"][i, :] .*
                (sim_forcing[var][i + 1, :] .- sim_forcing[var][i - 1, :]) ./
                (sim_forcing["z"][i + 1, :] .- sim_forcing["z"][i - 1, :])
        end
    end
    # return minus because we move the tendency to the RHS
    return -deriv
end

function get_coszen_inst(
    lat,
    lon,
    date,
    FT,
    param_set = IP.InsolationParameters(FT),
    od = Insolation.OrbitalData(),
)

    date = DateTime(date)

    S, μ =
        Insolation.solar_flux_and_cos_sza(date, od, FT(lon), FT(lat), param_set)

    return μ, S * μ
end

"""
    generate_external_era5_forcing_file(
        lat,
        lon,
        start_date,
        forcing_file_path,
        FT;
        input_data_dir,
        smooth_amount = 4,
        time_resolution = 3600,
        data_strs = [
            "forcing_and_cloud_hourly_profiles",
            "hourly_inst",
            "hourly_accum",
        ],
    )

Generate an external forcing file for the ClimaAtmos single column model.

# Input
The reanalysis, e.g., ERA5, input data is expected to be in the `era5_hourly_atmos_raw` artifact directory
and should contain 3 files:
    - Column profile dataset, named "forcing_and_cloud_hourly_profiles_"start_date".nc"
    - Surface sensible and latent heat fluxes, named "hourly_accum_"start_date".nc"
    - Surface temperature, named "hourly_inst_"start_date".nc"
The default file names can be overwritten by the `data_strs` argument. Parsed args should contain the site_latitude,
site_longitude, and start_date. The variables and specific naming convention for these files is better described 
in the Single Column Model section of the documentation.

# Output
The output file is written to forcing file path, by default stored in the `era5_hourly_atmos_processed` artifact 
directory joined with `daily` or `monthly` depending on the simulation type. It contains all forcings required to
drive the single column model.

Note:
- Single column runs are treated as boxes, so the dimensions of the variables are expanded to
    `2x2x(pressure levels)x(time)` to be able to interpolate to the model grid.
- The end time of the simulation is inferred from the start date and the simulation time, `t_end`.
"""
function generate_external_forcing_file(
    parsed_args,
    forcing_file_path,
    FT;
    input_data_dir,
    smooth_amount = 4,
    time_resolution = FT(3600), # size of accumulated variable period in seconds (3600 for hourly, 86400 for daily and monthly)
    data_strs = [
        "forcing_and_cloud_hourly_profiles",
        "hourly_inst",
        "hourly_accum",
    ],
)
    # unpack parsed args
    lat = parsed_args["site_latitude"]
    lon = parsed_args["site_longitude"]
    start_date = parsed_args["start_date"]

    external_tv_params = CP.get_parameter_values(
        CP.create_toml_dict(FT),
        ["gravitational_acceleration", "planet_radius", "gas_constant_dry_air"],
    )
    # load datasets
    tvforcing =
        NCDataset(joinpath(input_data_dir, "$(data_strs[1])_$(start_date).nc"))
    tv_inst =
        NCDataset(joinpath(input_data_dir, "$(data_strs[2])_$(start_date).nc"))
    tv_accum =
        NCDataset(joinpath(input_data_dir, "$(data_strs[3])_$(start_date).nc"))

    # round to era5 quarter degree resolution for site selection
    lat = round(lat * 4) / 4
    lon = round(lon * 4) / 4

    # find indexes for site location in pressure file
    lon_index = findfirst(tvforcing["longitude"][:] .== lon)
    lat_index = findfirst(tvforcing["latitude"][:] .== lat)
    @assert !isnothing(lon_index) "Longitude $lon not found in forcing_and_cloud_hourly_profiles_$(start_date).nc"
    @assert !isnothing(lat_index) "Latitude $lat not found in forcing_and_cloud_hourly_profiles_$(start_date).nc"
    @assert smooth_amount + 1 <
            lon_index <
            length(tvforcing["longitude"][:]) - smooth_amount "Longitude $lon is not covered by profile forcing file with smoothing amount $smooth_amount"
    @assert smooth_amount + 1 <
            lat_index <
            length(tvforcing["latitude"][:]) - smooth_amount "Latitude $lat is not covered by profile forcing file with smoothing amount $smooth_amount"

    sim_forcing = Dict()
    sim_forcing["time"] = tvforcing["valid_time"][:]
    sim_forcing["pressure_level"] = tvforcing["pressure_level"][:]

    name_map = clima_to_era5_name_dict()
    for clima_name in ["ua", "va", "wap", "hus", "ta", "zg", "clw", "cli"]
        era5_name = name_map[clima_name]
        sim_forcing[clima_name] = smooth_4D_era5(
            tvforcing,
            era5_name,
            lon_index,
            lat_index;
            smooth_amount = smooth_amount,
        )
    end

    sim_forcing["z"] =
        sim_forcing["zg"] / external_tv_params.gravitational_acceleration # height in meters

    # compute subsidence
    pressure = tvforcing["pressure_level"] .* 100 # convert hPa to Pa
    ρ =
        pressure ./
        (external_tv_params.gas_constant_dry_air .* sim_forcing["ta"])
    sim_forcing["rho"] = ρ # air density
    sim_forcing["wa"] =
        .-sim_forcing["wap"] ./
        (ρ .* external_tv_params.gravitational_acceleration)

    # compute vertical advection terms

    # TODO for steady forcing we need to prescribe vertical eddy tendencies (see #3771)
    # sim_forcing["tntva"] = get_vertical_tendencies(sim_forcing, "ta")
    # sim_forcing["tnhusva"] = get_vertical_tendencies(sim_forcing, "hus")

    # for time-varying forcing we set these tendencies to zero
    sim_forcing["tntva"] = zeros(size(sim_forcing["ta"]))
    sim_forcing["tnhusva"] = zeros(size(sim_forcing["hus"]))

    # compute horizontal advection terms - we need the spatial profile to compute horizontal gradients
    sim_forcing["tntha"], sim_forcing["tnhusha"] = get_horizontal_tendencies(
        lat,
        lon_index,
        lat_index,
        tvforcing,
        external_tv_params,
    )

    # create the dataset to store the forcing data - it needs to have expanded dimensions because
    # in ClimaAtmos all column simulations are actually boxes
    ds = NCDataset(forcing_file_path, "c")

    # Define the dimensions
    defDim(ds, "time", length(sim_forcing["time"]))
    defDim(ds, "z", length(sim_forcing["pressure_level"]))

    # expand dimensions for box
    defDim(ds, "x", 2)
    defDim(ds, "y", 2)

    # define variables for x and y at 0.5
    defVar(ds, "x", FT, ("x",))
    ds["x"][:] = [0.0, 1.0]

    defVar(ds, "y", FT, ("y",))
    ds["y"][:] = [0.0, 1.0]

    defVar(ds, "z", FT, ("z",))
    ds["z"][:] = mean(sim_forcing["z"], dims = 2)[:]

    # Define time variable with attributes
    defVar(
        ds,
        "time",
        sim_forcing["time"][:],
        ("time",),
        attrib = tvforcing["valid_time"].attrib,
    )

    # Define other variables
    defVar(ds, "pressure_level", FT, ("x", "y", "z"))
    # again repeating dimensions because single column model is a box
    for z_index in 1:ds.dim["z"]
        ds["pressure_level"][:, :, z_index] .=
            sim_forcing["pressure_level"][z_index]
    end

    # Define the variables and add them to the ds
    for (name, data) in sim_forcing
        if name in [
            "tnhusha",
            "tntha",
            "hus",
            "tntva",
            "zg",
            "wa",
            "ua",
            "va",
            "ta",
            "tnhusva",
            "wap",
            "rho",
            "clw",
            "cli",
        ]
            defVar(ds, name, FT, ("x", "y", "z", "time"))
            for x_ind in 1:ds.dim["x"], y_ind in 1:ds.dim["y"]
                ds[name][x_ind, y_ind, :, :] .= data
            end
        elseif name ∉ ["time", "pressure_level", "z"]
            # surface vars
            defVar(ds, name, FT, ("x", "y", "time"))
            for x_ind in 1:ds.dim["x"], y_ind in 1:ds.dim["y"]
                ds[name][x_ind, y_ind, :] .= data
            end
        end
    end

    # add coszen
    coszen_list = get_coszen_inst.(lat, lon, tvforcing["valid_time"][:], FT)
    defVar(ds, "coszen", FT, ("x", "y", "z", "time"))
    defVar(ds, "rsdt", FT, ("x", "y", "z", "time"))

    # add latent and sensitble heat fluxes (currently we just set surface conditions based on temperature)
    lon_index_surf = findfirst(tv_accum["longitude"][:] .== lon)
    lat_index_surf = findfirst(tv_accum["latitude"][:] .== lat)
    @assert !isnothing(lon_index_surf) "Longitude $lon not found in hourly_accum_$(start_date).nc"
    @assert !isnothing(lat_index_surf) "Latitude $lat not found in hourly_accum_$(start_date).nc"
    @assert smooth_amount + 1 <
            lon_index_surf <
            length(tv_accum["longitude"][:]) - smooth_amount "Longitude $lon is not covered by accumulated forcing file with smoothing amount $smooth_amount"
    @assert smooth_amount + 1 <
            lat_index_surf <
            length(tv_accum["latitude"][:]) - smooth_amount "Latitude $lat is not covered by accumulated forcing file with smoothing amount $smooth_amount"

    defVar(ds, "hfls", FT, ("x", "y", "z", "time"))
    defVar(ds, "hfss", FT, ("x", "y", "z", "time"))
    # sensible and latent heat fluxes are defined upwards in CliMA, also need to divide by the aggregation
    slhf =
        -smooth_3D_era5(
            tv_accum,
            "slhf",
            lon_index_surf,
            lat_index_surf;
            smooth_amount = smooth_amount,
        ) / time_resolution

    sshf =
        -smooth_3D_era5(
            tv_accum,
            "sshf",
            lon_index_surf,
            lat_index_surf;
            smooth_amount = smooth_amount,
        ) / time_resolution

    # surface temperature
    lon_index_surf2 = findfirst(tv_inst["longitude"][:] .== lon)
    lat_index_surf2 = findfirst(tv_inst["latitude"][:] .== lat)
    @assert !isnothing(lon_index_surf2) "Longitude $lon not found in hourly_inst_$(start_date).nc"
    @assert !isnothing(lat_index_surf2) "Latitude $lat not found in hourly_inst_$(start_date).nc"
    @assert smooth_amount + 1 <
            lon_index_surf2 <
            length(tv_inst["longitude"][:]) - smooth_amount "Longitude $lon is not covered by accumulated forcing file with smoothing amount $smooth_amount"
    @assert smooth_amount + 1 <
            lat_index_surf2 <
            length(tv_inst["latitude"][:]) - smooth_amount "Latitude $lat is not covered by accumulated forcing file with smoothing amount $smooth_amount"

    defVar(ds, "ts", FT, ("x", "y", "z", "time"))
    skt = smooth_3D_era5(
        tv_inst,
        "skt",
        lon_index_surf2,
        lat_index_surf2;
        smooth_amount = smooth_amount,
    )

    for time_ind in 1:ds.dim["time"]
        ds["coszen"][:, :, :, time_ind] .= coszen_list[time_ind][1]
        ds["rsdt"][:, :, :, time_ind] .= coszen_list[time_ind][2]
        ds["ts"][:, :, :, time_ind] .= skt[time_ind]
        ds["hfls"][:, :, :, time_ind] .= slhf[time_ind]
        ds["hfss"][:, :, :, time_ind] .= sshf[time_ind]
    end

    # Close the datasets
    close(ds)
    close(tvforcing)
    close(tv_inst)
    close(tv_accum)
end

"""
    generate_multiday_era5_external_forcing_file(parsed_args, forcing_file_path, FT; smooth_amount = 4, time_resolution = FT(3600), input_data_dir = @clima_artifact("era5_hourly_atmos_raw"), output_data_dir = @clima_artifact("era5_hourly_atmos_processed"))

Generate an external forcing file for multi-day single column model runs, reusing daily forcing files if they already exist.

# Arguments
- `parsed_args`: Dictionary containing simulation parameters including start_date, t_end, site_latitude, and site_longitude
- `forcing_file_path`: Path where the concatenated forcing file will be saved
- `FT`: Floating point type for the simulation

# Keyword Arguments
- `smooth_amount`: Amount of temporal smoothing to apply (default: 4 - 1° on each side)
- `time_resolution`: Time resolution in seconds for accumulated variables (defined in ERA5 docs; 3600 for hourly data; 86400 for daily and monthly data)
- `input_data_dir`: Directory containing raw ERA5 data files, artifact directory by default
- `output_data_dir`: Directory where individual daily forcing files are stored
"""
function generate_multiday_era5_external_forcing_file(
    parsed_args,
    forcing_file_path,
    FT;
    smooth_amount = 4,
    time_resolution = FT(3600), # size of accumulated variable period in seconds (3600 for hourly, 86400 for daily and monthly)
    input_data_dir = @clima_artifact("era5_hourly_atmos_raw"),
    output_data_dir = get(ENV, "BUILDKITE", "") == "true" ? mktempdir() :
                      @clima_artifact("era5_hourly_atmos_processed"),
)
    # run generate_external_era5_forcing_file for each day if its processed data file not found 
    # get range of starttimes and endtimes
    start_date = DateTime(parsed_args["start_date"], "yyyymmdd")
    end_time = start_date + Dates.Second(time_to_seconds(parsed_args["t_end"]))
    end_date = Dates.format(end_time, "yyyymmdd")

    start_dates = start_date:Day(1):end_time

    file_list = String[]
    for dd in start_dates
        # get forcing file path
        single_parsed_args = Dict(
            "start_date" => Dates.format(dd, "yyyymmdd"),
            "site_latitude" => parsed_args["site_latitude"],
            "site_longitude" => parsed_args["site_longitude"],
        )
        single_file_path = get_external_daily_forcing_file_path(
            single_parsed_args;
            data_dir = output_data_dir,
        )
        push!(file_list, single_file_path)
        # generate the external forcing file for this day
        if !isfile(single_file_path)
            generate_external_forcing_file(
                single_parsed_args,
                single_file_path,
                FT;
                time_resolution = time_resolution,
                input_data_dir = input_data_dir,
                smooth_amount = smooth_amount,
            )
        end
    end
    # concatenate data and save 
    concat_ds = Dataset(file_list; aggdim = "time")
    NCDatasets.write(forcing_file_path, concat_ds)
end

"""
    smooth_4D_era5(data, variable, lon_index, lat_index; smooth_amount = 4)

data is an array from ERA5 data, which has dimension order longitude, latitude,
pressure_level, and time. We want to return smoothed data by a certain amount. 
Here we choose 4 points on either side which corresponds to a 2° box total. we
just average the points here, but something more creative could be done.
"""
function smooth_4D_era5(data, variable, lon_index, lat_index; smooth_amount = 4)
    # extract data in box around the center point
    data_slice = data[variable][
        (lon_index - smooth_amount):(lon_index + smooth_amount),
        (lat_index - smooth_amount):(lat_index + smooth_amount),
        :,
        :,
    ]
    # compute mean over lat/lon dimensions and return slice
    return mean(data_slice, dims = (1, 2))[1, 1, :, :]
end

"""
    smooth_3D_era5(data, variable, lon_index, lat_index; smooth_amount = 4)

data is an array from ERA5, which has dimension order longitude, latitude, and time. 
This function returns data smoothed by a certain amount. Here, we choose 4 points on 
either side which corresponds to a 2° box total. wejust average the points here, but 
something more creative could be done.
"""
function smooth_3D_era5(data, variable, lon_index, lat_index; smooth_amount = 4)
    # extract data in box around the center point
    data_slice = data[variable][
        (lon_index - smooth_amount):(lon_index + smooth_amount),
        (lat_index - smooth_amount):(lat_index + smooth_amount),
        :,
    ]
    # compute mean over lat/lon dimensions and return slice
    return mean(data_slice, dims = (1, 2))[1, 1, :]
end
