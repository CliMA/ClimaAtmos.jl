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
using Insolation
import Insolation.Parameters as IP
import ClimaParams as CP

time_resolution = 3600 # switch to 86400 for monthly data

external_tv_params = CP.get_parameter_values(
    CP.create_toml_dict(Float64),
    [
        "gravitational_acceleration",
        "planet_radius",
        "gas_constant",
        "molar_mass_dry_air",
    ],
)

function get_external_forcing_file_path(
    parsed_args;
    data_dir = get(ENV, "BUILDKITE", "") == "true" ?
               joinpath(tempdir(), "era5_hourly_atmos_processed") :
               @clima_artifact("era5_hourly_atmos_processed"),
)
    start_date = parsed_args["start_date"]
    # round to era5 quarter degree resolution for site selection
    site_latitude = round(parsed_args["site_latitude"] * 4) / 4
    site_longitude = round(parsed_args["site_longitude"] * 4) / 4

    return joinpath(
        data_dir,
        "tv_forcing_$(site_latitude)_$(site_longitude)_$(start_date).nc",
    )
end


"""
    get_horizontal_tendencies(lat, lon_index, lat_index, column_ds)

Calculate the horizontal advective tendencies for temperature and specific humidity
using a second-order finite difference approximation.
"""
function get_horizontal_tendencies(lat, lon_index, lat_index, column_ds)
    rearth = external_tv_params.planet_radius
    lat_rad = deg2rad(lat)
    coslat = cos(lat_rad)
    dx = 2 * π * rearth * coslat / 360 * 0.25
    dy = 2 * π * rearth / 360 * 0.25

    # get velocities at site location
    ᶜu = column_ds["u"][lon_index, lat_index, :, :]
    ᶜv = column_ds["v"][lon_index, lat_index, :, :]

    # get temperature at N S E W of center for gradient calculation
    ʷT = column_ds["t"][lon_index - 1, lat_index, :, :]
    ⁿT = column_ds["t"][lon_index, lat_index + 1, :, :]
    ˢT = column_ds["t"][lon_index, lat_index - 1, :, :]
    ᵉT = column_ds["t"][lon_index + 1, lat_index, :, :]

    # get specific humidity at N S E W of center for gradient calculation
    ʷq = column_ds["q"][lon_index - 1, lat_index, :, :]
    ⁿq = column_ds["q"][lon_index, lat_index + 1, :, :]
    ˢq = column_ds["q"][lon_index, lat_index - 1, :, :]
    ᵉq = column_ds["q"][lon_index + 1, lat_index, :, :]

    # temperature and specific humidity advective tendency at center
    tntha = -(ᶜu .* (ᵉT .- ʷT) ./ (2dx) .+ ᶜv .* (ⁿT .- ˢT) ./ (2dy))
    tnhusha = -(ᶜu .* (ᵉq .- ʷq) ./ (2dx) .+ ᶜv .* (ⁿq .- ˢq) ./ (2dy))

    return tntha, tnhusha
end

"""
    get_vertical_tendencies(sim_forcing, var)

Calculate the temperature and specific humidity vertical tendencies as a function of levels
using vertical advection using second-order finite difference at interior points and
first-order finite difference at the top and bottom levels.
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

    date0 = DateTime("2000-01-01T11:58:56.816")

    S, μ = solar_flux_and_cos_sza(date, date0, od, FT(lon), FT(lat), param_set)

    return μ, S * μ
end


function generate_external_era5_forcing_file(
    lat,
    lon,
    start_date,
    forcing_file_path,
    FT,
)

    # load datasets
    artifact_data_directory = @clima_artifact("era5_hourly_atmos_raw")
    tvforcing = NCDataset(
        joinpath(
            artifact_data_directory,
            "forcing_and_cloud_hourly_profiles_$(start_date).nc",
        ),
    )
    tv_inst = NCDataset(
        joinpath(artifact_data_directory, "hourly_inst_$(start_date).nc"),
    )
    tv_accum = NCDataset(
        joinpath(artifact_data_directory, "hourly_accum_$(start_date).nc"),
    )


    sim_forcing = Dict()
    sim_forcing["time"] = tvforcing["valid_time"][:]

    # find indexes for site location in pressure file
    lon_index = findfirst(tvforcing["longitude"][:] .== lon)
    lat_index = findfirst(tvforcing["latitude"][:] .== lat)

    sim_forcing = Dict()
    sim_forcing["time"] = tvforcing["valid_time"][:]
    sim_forcing["pressure_level"] = tvforcing["pressure_level"][:]
    sim_forcing["ua"] = tvforcing["u"][lon_index, lat_index, :, :]
    sim_forcing["va"] = tvforcing["v"][lon_index, lat_index, :, :]
    sim_forcing["wap"] = tvforcing["w"][lon_index, lat_index, :, :] # era5 w is in Pa/s, this is confusing notation
    sim_forcing["hus"] = tvforcing["q"][lon_index, lat_index, :, :]
    sim_forcing["ta"] = tvforcing["t"][lon_index, lat_index, :, :]
    sim_forcing["zg"] = tvforcing["z"][lon_index, lat_index, :, :]
    sim_forcing["z"] =
        tvforcing["z"][lon_index, lat_index, :, :] /
        external_tv_params.gravitational_acceleration # height in meters

    # add cloud information - this is used for profile calibration and not for the forcing but saves multiple files for profile calibration
    sim_forcing["clw"] = tvforcing["clwc"][lon_index, lat_index, :, :]
    sim_forcing["cli"] = tvforcing["ciwc"][lon_index, lat_index, :, :]

    # compute subsidence
    pressure = tvforcing["pressure_level"] .* 100 # convert hPa to Pa
    R_d =
        external_tv_params.gas_constant / external_tv_params.molar_mass_dry_air # J/(kg*K)
    ρ = reshape(pressure, 37, 1) ./ (R_d .* sim_forcing["ta"])
    sim_forcing["rho"] = ρ # pressure
    sim_forcing["wa"] =
        .-sim_forcing["wap"] ./
        (ρ .* external_tv_params.gravitational_acceleration)


    # compute vertical advection terms - zero for time varying forcing, nonzero for steady state
    sim_forcing["tntva"] =
        zeros(size(get_vertical_tendencies(sim_forcing, "ta")))
    sim_forcing["tnhusva"] =
        zeros(size(get_vertical_tendencies(sim_forcing, "hus")))

    # compute horizontal advection terms - we need the spatial profile to compute horizontal gradients
    sim_forcing["tntha"], sim_forcing["tnhusha"] =
        get_horizontal_tendencies(lat, lon_index, lat_index, tvforcing)

    # create the dataset to store the forcing data - it needs to have expanded dimensions because
    # in ClimaAtmos all column simulations are actually boxes
    ds = Dataset(forcing_file_path, "c")

    # Define the dimensions
    defDim(ds, "time", length(sim_forcing["time"]))
    defDim(ds, "z", length(sim_forcing["pressure_level"]))

    # expand dimensions for box
    defDim(ds, "x", 2)
    defDim(ds, "y", 2)

    # define variables for x and y at 0.5
    defVar(ds, "x", Float64, ("x",))
    ds["x"][:] = [0.0, 1.0]

    defVar(ds, "y", Float64, ("y",))
    ds["y"][:] = [0.0, 1.0]

    defVar(ds, "z", Float64, ("z",))
    ds["z"][:] = mean(sim_forcing["z"], dims = 2)[:]

    # Convert DateTime to numeric values
    time_ref = DateTime(1970, 1, 1)  # Reference time (Unix epoch)
    time_values =
        Float64.(
            Dates.value.(sim_forcing["time"] .- time_ref) ./ (1e3 * 60 * 60)
        ) # hours since 1970-01-01 00:00:00

    # Define time variable with attributes
    defVar(
        ds,
        "time",
        Float64,
        ("time",),
        attrib = [
            "units" => "hours since 1970-01-01 00:00:00",
            "calendar" => "proleptic_gregorian",
        ],
    )
    ds["time"][:] = time_values

    # Define other variables
    defVar(ds, "pressure_level", Float64, ("x", "y", "z"))
    # again repeating dimensions because single column model is a box
    ds["pressure_level"][:] =
        repeat(reshape(sim_forcing["pressure_level"], 1, 1, :), 2, 2, 1)

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
            defVar(ds, name, Float64, ("x", "y", "z", "time"))
            ds[name][:] = repeat(reshape(data, 1, 1, size(data)...), 2, 2, 1, 1)
        elseif name ∉ ["time", "pressure_level", "z"]
            defVar(ds, name, Float64, ("x", "y", "time"))
            ds[name][:] = repeat(reshape(data, 1, 1, size(data)...), 2, 2, 1)
        end
    end

    # add coszen
    coszen_list = get_coszen_inst.(lat, lon, tvforcing["valid_time"][:], FT)
    defVar(ds, "coszen", Float64, ("x", "y", "z", "time"))
    ds["coszen"][:] = repeat(
        reshape([c[1] for c in coszen_list], 1, 1, 1, :),
        2,
        2,
        length(ds["z"]),
        1,
    )

    defVar(ds, "rsdt", Float64, ("x", "y", "z", "time"))
    ds["rsdt"][:] = repeat(
        reshape([c[2] for c in coszen_list], 1, 1, 1, :),
        2,
        2,
        length(ds["z"]),
        1,
    )

    # add latent and sensitble heat fluxes (currently we just set surface conditions based on temperature)
    lon_index_surf = findfirst(tv_accum["longitude"][:] .== lon)
    lat_index_surf = findfirst(tv_accum["latitude"][:] .== lat)
    matching_time_indices =
        findall(in(tvforcing["valid_time"][:]), tv_accum["valid_time"][:])

    defVar(ds, "hfls", Float64, ("x", "y", "z", "time"))
    defVar(ds, "hfss", Float64, ("x", "y", "z", "time"))
    # sensible and latent heat fluxes are opposite
    slhf =
        -tv_accum["slhf"][
            lon_index_surf,
            lat_index_surf,
            matching_time_indices,
        ] / time_resolution
    sshf =
        -tv_accum["sshf"][
            lon_index_surf,
            lat_index_surf,
            matching_time_indices,
        ] / time_resolution
    ds["hfls"][:] = repeat(reshape(slhf, 1, 1, 1, :), 2, 2, length(ds["z"]), 1)
    ds["hfss"][:] = repeat(reshape(sshf, 1, 1, 1, :), 2, 2, length(ds["z"]), 1)

    # surface temperature
    lon_index_surf2 = findfirst(tv_inst["longitude"][:] .== lon)
    lat_index_surf2 = findfirst(tv_inst["latitude"][:] .== lat)
    matching_time_indices =
        findall(in(tvforcing["valid_time"][:]), tv_inst["valid_time"][:])

    defVar(ds, "ts", Float64, ("x", "y", "z", "time"))
    skt =
        tv_inst["skt"][lon_index_surf2, lat_index_surf2, matching_time_indices]
    ds["ts"][:] = repeat(reshape(skt, 1, 1, 1, :), 2, 2, length(ds["z"]), 1)

    # Close the datasets
    close(ds)
    close(tvforcing)
    close(tv_inst)
    close(tv_accum)
end
