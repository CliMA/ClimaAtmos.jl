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

Return the path of the daily ERA5 forcing file for the site and time range described by
`parsed_args`.

The name encodes the site latitude and longitude (rounded to the ERA5 quarter-degree
grid, with a message when rounding changes them) and the start and end dates, the latter
inferred from `start_date` and `t_end` (`"23hours"`, i.e. a single day, when `t_end` is
absent). Under Buildkite `data_dir` defaults to a temporary directory, and otherwise to
the `daily` subdirectory of the `era5_hourly_atmos_processed` artifact.
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

Return the path of the monthly-averaged diurnal ERA5 forcing file for the site and
month described by `parsed_args`.

The name encodes the site latitude and longitude (rounded to the ERA5 quarter-degree
grid, with a message when rounding changes them), the start date, and the
`era5_diurnal_warming` offset [K] when one is set. Under Buildkite `data_dir` defaults
to a temporary directory, and otherwise to the `monthly` subdirectory of the
`era5_hourly_atmos_processed` artifact.
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
    warming_amount = parsed_args["era5_diurnal_warming"]
    warming_amount_str =
        (warming_amount isa Number) ? "_plus_$(float(warming_amount))K" : ""

    # round to era5 quarter degree resolution for site selection
    site_latitude = round(parsed_args["site_latitude"] * 4) / 4
    site_longitude = round(parsed_args["site_longitude"] * 4) / 4

    if site_latitude != parsed_args["site_latitude"] ||
       site_longitude != parsed_args["site_longitude"]
        @info "Rounded site latitude/longitude from ($(parsed_args["site_latitude"]), $(parsed_args["site_longitude"])) to ($(site_latitude), $(site_longitude)) for ERA5 quarter-degree resolution."
    end
    return joinpath(
        data_dir,
        "monthly_diurnal_cycle_forcing_$(site_latitude)_$(site_longitude)_$(start_date)$(warming_amount_str).nc",
    )
end

"""
    check_daily_forcing_times(forcing_file_path, parsed_args)

Check that the run's start and end times, derived from `start_date` and `t_end`, lie
within the time range of the forcing file, warning when they do not.

# Notes

The `return false` statements sit inside the `NCDataset` `do` block, so they exit only
that closure: the function itself currently always returns `true`, and the warnings are
the effective signal.
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

Check that the monthly-averaged diurnal forcing file covers exactly one day starting at
`start_date`, warning when it does not.

Exactly one day is required because the file is read with
`ClimaUtilities.TimeVaryingInputs.PeriodicCalendar`, which wraps it in time.

# Notes

As in `check_daily_forcing_times`, the `return false` statements exit only the
`NCDataset` `do` block, so the function itself currently always returns `true`.
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

Compute the horizontal advective tendencies of temperature and specific humidity at the
site, using centered differences of the smoothed fields at the four neighboring grid
points.

Grid spacings are derived from the longitude and latitude increments of `column_ds` and
the planet radius in `external_tv_params`. The routine warns when the data are coarser
than 2°.

# Returns

`(tntha, tnhusha)`: temperature [K/s] and specific humidity [kg/kg/s] tendencies, sign
convention as right-hand-side forcing terms.
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

Compute the vertical advective tendency of the field `var` in `sim_forcing`, using
centered differences at interior levels and one-sided differences at the top and bottom.

The result is negated, so that it can be added to the right-hand side of the prognostic
equation. Currently unused: it applies to steady forcing, which is not supported (the
time-varying path sets these tendencies to zero).
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

"""
    generate_external_forcing_file(parsed_args, forcing_file_path, FT; input_data_dir,
                                   smooth_amount = 4, time_resolution = FT(3600),
                                   data_strs = ["forcing_and_cloud_hourly_profiles",
                                                "hourly_inst", "hourly_accum"])

Write a single-column forcing file for one site and one raw-data period.

The site is taken from `site_latitude` and `site_longitude` in `parsed_args` (rounded to
the ERA5 quarter-degree grid) and the period from `start_date`. Profiles are averaged
over a box of `smooth_amount` points on each side, horizontal advective tendencies come
from `get_horizontal_tendencies`, subsidence is derived from the pressure velocity, and
insolation (`coszen`, `rsdt`) is computed with Insolation.jl. When
`era5_diurnal_warming` is a number, air and surface temperatures are warmed by it and
specific humidity is rescaled at fixed relative humidity.

# Arguments

  - `parsed_args`: Configuration entries; `site_latitude` [°], `site_longitude` [°],
    `start_date` (`yyyymmdd`), and `era5_diurnal_warming` [K] are read.
  - `forcing_file_path`: Path of the NetCDF file to write.
  - `FT`: Floating-point type of the output.

# Keyword Arguments

  - `input_data_dir`: Directory holding the three raw reanalysis files for `start_date`.
  - `smooth_amount = 4`: Half-width, in grid points, of the box average around the site
    (4 points is 1° per side at ERA5 quarter-degree resolution).
  - `time_resolution = FT(3600)`: Accumulation period of the accumulated surface fields
    [s]; 3600 for hourly data, 86400 for daily and monthly data.
  - `data_strs`: Prefixes of the three input files, each completed as
    `<prefix>_<start_date>.nc`: column profiles, instantaneous surface fields (surface
    temperature), and accumulated surface fields (sensible and latent heat fluxes).

# Notes

  - Surface fluxes are negated, because CliMA defines them positive upwards, and divided
    by `time_resolution` to convert accumulations to rates [W/m²].
  - The output follows the ClimaColumn schema: 1D `(z, time)` column variables and
    `(time,)` surface variables with CMIP names and SI units, sorted by ascending height
    and made self-describing by the `site_latitude` and `site_longitude` global
    attributes (written through `ClimaColumnFiles.write_column_forcing_file`).
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

    warming_amount = parsed_args["era5_diurnal_warming"]
    if warming_amount isa Number
        # Get the relative humidity before warming; this will be used to scale the specific humidity after warming
        param_set = TD.Parameters.ThermodynamicsParameters(FT)
        # expand dimension and convert pressure levels from hPa (ERA5 raw) to Pa (needed for thermodynamics)
        p_expanded =
            FT.(
                repeat(
                    sim_forcing["pressure_level"] .* 100,
                    1,
                    size(sim_forcing["hus"], 2),
                ),
            )
        relative_humidity =
            TD.relative_humidity.(
                param_set,
                sim_forcing["ta"],
                p_expanded,
                sim_forcing["hus"],
                sim_forcing["clw"],
                sim_forcing["cli"],
            )

        # Warming the temperature
        sim_forcing["ta"] .+= warming_amount
        ρ =
            TD.air_density.(
                param_set,
                sim_forcing["ta"],
                p_expanded,
                sim_forcing["hus"],
                sim_forcing["clw"],
                sim_forcing["cli"],
            )

        # Get the saturation specific humidity at the warmed temperature
        saturation_humidity_at_warming =
            TD.q_vap_saturation.(
                param_set,
                sim_forcing["ta"],
                ρ,
                sim_forcing["clw"],
                sim_forcing["cli"],
            )
        sim_forcing["hus"] = saturation_humidity_at_warming .* relative_humidity
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

    # Cosine of the solar zenith angle (`coszen`) and TOA incoming shortwave
    # (`rsdt`), from Insolation
    times = tvforcing["valid_time"][:]
    coszen = Vector{FT}(undef, length(times))
    rsdt = Vector{FT}(undef, length(times))
    for (i, date) in enumerate(times)
        F, _, μ, _ = Insolation.insolation(
            DateTime(date),
            FT(lat),
            FT(lon),
            IP.InsolationParameters(FT),
        )
        coszen[i] = μ
        rsdt[i] = F
    end

    # latent and sensible heat fluxes
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

    skt = smooth_3D_era5(
        tv_inst,
        "skt",
        lon_index_surf2,
        lat_index_surf2;
        smooth_amount = smooth_amount,
    )

    # warm the surface temperature if warming amount is specified
    if warming_amount isa Number
        skt .+= warming_amount
    end

    # sort levels by ascending height, as required by the ClimaColumn schema
    z_column = vec(mean(sim_forcing["z"], dims = 2))
    z_order = sortperm(z_column)

    column_var_names = [
        "ta",
        "hus",
        "ua",
        "va",
        "wa",
        "rho",
        "tntha",
        "tnhusha",
        "tntva",
        "tnhusva",
        "clw",
        "cli",
    ]
    ClimaColumnFiles.write_column_forcing_file(
        forcing_file_path,
        FT;
        z = z_column[z_order],
        time = sim_forcing["time"][:],
        time_attrib = tvforcing["valid_time"].attrib,
        column_vars = Dict(
            name => sim_forcing[name][z_order, :] for name in column_var_names
        ),
        surface_vars = Dict(
            "coszen" => coszen,
            "rsdt" => rsdt,
            "ts" => skt,
            "hfls" => slhf,
            "hfss" => sshf,
        ),
        site_latitude = lat,
        site_longitude = lon,
    )

    close(tvforcing)
    close(tv_inst)
    close(tv_accum)
end

"""
    generate_multiday_era5_external_forcing_file(parsed_args, forcing_file_path, FT;
                                                 smooth_amount = 4,
                                                 time_resolution = FT(3600),
                                                 input_data_dir, output_data_dir)

Write a multi-day single-column forcing file by generating one daily file per day of the
run with `generate_external_forcing_file` and concatenating them along time.

A daily file is reused when it already exists and conforms to the ClimaColumn schema,
and regenerated otherwise.

# Arguments

  - `parsed_args`: Configuration entries; `start_date`, `t_end`, `site_latitude` [°],
    `site_longitude` [°], and `era5_diurnal_warming` [K] are read.
  - `forcing_file_path`: Path of the concatenated NetCDF file to write.
  - `FT`: Floating-point type of the output.

# Keyword Arguments

  - `smooth_amount = 4`: Half-width, in grid points, of the box average around the site
    (4 points is 1° per side at ERA5 quarter-degree resolution).
  - `time_resolution = FT(3600)`: Accumulation period of the accumulated surface fields
    [s]; 3600 for hourly data, 86400 for daily and monthly data.
  - `input_data_dir`: Directory with the raw ERA5 files; the `era5_hourly_atmos_raw`
    artifact by default.
  - `output_data_dir`: Directory holding the individual daily forcing files; a temporary
    directory under Buildkite, and the `era5_hourly_atmos_processed` artifact otherwise.
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

    daily_specs = map(start_dates) do dd
        single_parsed_args = Dict(
            "start_date" => Dates.format(dd, "yyyymmdd"),
            "site_latitude" => parsed_args["site_latitude"],
            "site_longitude" => parsed_args["site_longitude"],
            "era5_diurnal_warming" => parsed_args["era5_diurnal_warming"],
        )
        single_file_path = get_external_daily_forcing_file_path(
            single_parsed_args;
            data_dir = output_data_dir,
        )
        (; parsed_args = single_parsed_args, path = single_file_path)
    end
    file_list = [spec.path for spec in daily_specs]

    # A daily file is (re)generated when it is missing, or when an older run
    # wrote it in the previous ERA5 box layout rather than the current
    # ClimaColumn `(z, time)` schema, so the concatenation below sees one
    # layout.
    for spec in daily_specs
        isfile(spec.path) &&
            ClimaColumnFiles.is_conforming(spec.path) &&
            continue
        isfile(spec.path) &&
            @info "Regenerating $(spec.path): not a conforming ClimaColumn file (stale layout or failed schema validation)"
        generate_external_forcing_file(
            spec.parsed_args,
            spec.path,
            FT;
            time_resolution = time_resolution,
            input_data_dir = input_data_dir,
            smooth_amount = smooth_amount,
        )
    end
    # concatenate data and save
    concat_ds = Dataset(file_list; aggdim = "time")
    NCDatasets.write(forcing_file_path, concat_ds)
end

"""
    smooth_4D_era5(data, variable, lon_index, lat_index; smooth_amount = 4)

Average a 4D ERA5 variable (longitude, latitude, pressure level, time) over a box of
`smooth_amount` grid points on each side of the given horizontal index, returning a
`(pressure_level, time)` array.

The default half-width of 4 points spans a 2° box at ERA5 quarter-degree resolution.
The box is averaged with equal weights; a more elaborate filter could be used instead.
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

Average a 3D ERA5 surface variable (longitude, latitude, time) over a box of
`smooth_amount` grid points on each side of the given horizontal index, returning a
`(time,)` vector.

The default half-width of 4 points spans a 2° box at ERA5 quarter-degree resolution.
The box is averaged with equal weights; a more elaborate filter could be used instead.
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
