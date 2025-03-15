using NCDatasets
using Statistics
using Dates
using DataFrames
using Printf

# for radiation calculation 
using Insolation
import Insolation.Parameters as IP 
import ClimaParams as CP

time_resolution = 3600 # switch to 86400 for monthly data 
FT = Float64

# parameters 
R_d = 287.05
g = -9.81
rootdir = "/scratch/julian/ERA5/data_download/"
rootdir = "/Users/julianschmitt/Downloads/era5/"
tvforcing = NCDataset(rootdir * "july2007_forcing_and_cloud_hourly_profiles.nc") # profile data
tv_inst = NCDataset(rootdir * "july2007_hourly_inst.nc") # skt
tv_accum = NCDataset(rootdir * "july2007_hourly_accum.nc") # slhf, sshf

function get_horizontal_tendencies(lat, lon_index, lat_index, column_ds)
    """
    Computes horizontal tendencies for temperature and specific humidity at a given location.
    """
    rearth = 6378e3
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

# # compute vertical advection terms
# function get_vertical_tendencies(sim_forcing, var, vertvar = "wap")
#     """
#     Calculate the temperature and specific humidity vertical tendencies as a function of levels
#     using vertical advection. Here we take the tendency over the geopotential height not the height in meters
#     """
#     deriv = zeros(size(sim_forcing["wap"]))

#     for i in 1:size(sim_forcing["wap"])[1]
#         if i == 1
#             deriv[1, :] = sim_forcing["wap"][1, :] .* (sim_forcing[var][2, :] .- sim_forcing[var][1, :]) ./ (sim_forcing["zg"][2, :] .- sim_forcing["zg"][1, :])
#         elseif i == size(sim_forcing["wap"])[1]
#             deriv[end, :] = sim_forcing["wap"][end, :] .* (sim_forcing[var][end, :] .- sim_forcing[var][end-1, :]) ./ (sim_forcing["zg"][end, :] .- sim_forcing["zg"][end-1, :])
#         else # centered FD 
#             deriv[i, :] = sim_forcing["wap"][i, :] .* (sim_forcing[var][i+1, :] .- sim_forcing[var][i-1, :]) ./ (sim_forcing["zg"][i+1, :] .- sim_forcing["zg"][i-1, :])
#         end
#     end
    
#     deriv 
# end

# compute vertical advection terms
function get_vertical_tendencies(sim_forcing, var)
    """
    Calculate the temperature and specific humidity vertical tendencies as a function of levels
    using vertical advection. Here we take the tendency over the geopotential height not the height in meters
    """
    deriv = zeros(size(sim_forcing["wa"]))

    for i in 1:size(sim_forcing["wa"])[1]
        if i == 1
            deriv[1, :] = sim_forcing["wa"][1, :] .* (sim_forcing[var][2, :] .- sim_forcing[var][1, :]) ./ (sim_forcing["z"][2, :] .- sim_forcing["z"][1, :])
        elseif i == size(sim_forcing["wa"])[1]
            deriv[end, :] = sim_forcing["wa"][end, :] .* (sim_forcing[var][end, :] .- sim_forcing[var][end-1, :]) ./ (sim_forcing["z"][end, :] .- sim_forcing["z"][end-1, :])
        else # centered FD 
            deriv[i, :] = sim_forcing["wa"][i, :] .* (sim_forcing[var][i+1, :] .- sim_forcing[var][i-1, :]) ./ (sim_forcing["z"][i+1, :] .- sim_forcing["z"][i-1, :])
        end
    end
    # return minus version because we move the tendency to the RHS 
    return -deriv 
end

function get_coszen_inst(lat, lon, date,
    param_set = IP.InsolationParameters(FT),
    od = Insolation.OrbitalData())

    date = DateTime(date)

    date0 = DateTime("2000-01-01T11:58:56.816")

    S, μ = solar_flux_and_cos_sza(date, date0, od, FT(lon), FT(lat), param_set)

    return μ, S * μ
end


function compute_forcing(lat, lon, tvforcing, tv_inst, tv_accum)

    sim_forcing = Dict()
    sim_forcing["time"] = tvforcing["valid_time"][:]

    # find indexes for site location in pressure file 
    lon_index = findfirst(tvforcing["longitude"][:] .== lon)
    lat_index = findfirst(tvforcing["latitude"][:] .== lat)

    sim_forcing["tntha"], sim_forcing["tnhusha"] = get_horizontal_tendencies(lat, lon_index, lat_index, tvforcing)

    sim_forcing = Dict()
    sim_forcing["time"] = tvforcing["valid_time"][:]
    sim_forcing["pressure_level"] = tvforcing["pressure_level"][:]

    lon_index = findfirst(tvforcing["longitude"][:] .== lon)
    lat_index = findfirst(tvforcing["latitude"][:] .== lat)

    sim_forcing["ua"] = tvforcing["u"][lon_index, lat_index, :, :]
    sim_forcing["va"] = tvforcing["v"][lon_index, lat_index, :, :]
    sim_forcing["wap"] = tvforcing["w"][lon_index, lat_index, :, :]
    sim_forcing["hus"] = tvforcing["q"][lon_index, lat_index, :, :]
    sim_forcing["ta"] = tvforcing["t"][lon_index, lat_index, :, :]
    sim_forcing["zg"] = tvforcing["z"][lon_index, lat_index, :, :]
    sim_forcing["z"] = tvforcing["z"][lon_index, lat_index, :, :] / (-g) # height in meters

    # add cloud information - this is used for profile calibration and not for the forcing but saves multiple files for profile calibration
    sim_forcing["clw"] = tvforcing["clwc"][lon_index, lat_index, :, :]
    sim_forcing["cli"] = tvforcing["ciwc"][lon_index, lat_index, :, :]


    # compute subsidence
    pressure = tvforcing["pressure_level"] .* 100 # convert hPa to Pa
    ρ = reshape(pressure, 37, 1) ./ (R_d .* sim_forcing["ta"])
    sim_forcing["rho"] = ρ # pressure 
    println("rho: ", size(ρ))
    println("ta: ", size(sim_forcing["ta"]))
    println("wap", size(sim_forcing["wap"]))
    #sim_forcing["wap"] = sim_forcing["wa"] .* ρ
    sim_forcing["wa"] = sim_forcing["wap"] ./ (ρ .* g) # g is already negative


    # compute vertical advection terms - for these terms we don't need horizontal gradients so can pass sim_forcing directly
    sim_forcing["tntva"] = get_vertical_tendencies(sim_forcing, "ta")
    sim_forcing["tnhusva"] = get_vertical_tendencies(sim_forcing, "hus")
    sim_forcing["tntha"], sim_forcing["tnhusha"] = get_horizontal_tendencies(lat, lon_index, lat_index, tvforcing)

    ds = Dataset("sim_forcing_loc_$(lat)_$(lon).nc", "c")


    # Define the dimensions
    defDim(ds, "time", length(sim_forcing["time"]))
    defDim(ds, "z", length(sim_forcing["pressure_level"]))

    # add ghost dims for box
    defDim(ds, "x", 2)
    defDim(ds, "y", 2)

    # define variables for x and y at 0.5
    defVar(ds, "x", Float64, ("x",))
    ds["x"][:] = [0., 1.]

    defVar(ds, "y", Float64, ("y",))
    ds["y"][:] = [0., 1.]

    defVar(ds, "z", Float64, ("z",))
    ds["z"][:] = mean(sim_forcing["z"], dims = 2)[:]

    # Convert DateTime to numeric values
    time_ref = DateTime(1970, 1, 1)  # Reference time (Unix epoch)
    time_values = Float64.(Dates.value.(sim_forcing["time"] .- time_ref) ./ (1e3 * 60 * 60)) # hours since 1970-01-01 00:00:00

    # Define time variable with attributes
    defVar(ds, "time", Float64, ("time",),
        attrib = [
            "units" => "hours since 1970-01-01 00:00:00",
            "calendar" => "proleptic_gregorian"
        ]
    )
    ds["time"][:] = time_values

    # Define other variables
    defVar(ds, "pressure_level", Float64, ("x", "y", "z"))
    # note the repeat is for downstream where we have to interpolate back 
    #the 2 by 2 grid allows that to happen until those issues are fixed
    ds["pressure_level"][:] = repeat(reshape(sim_forcing["pressure_level"], 1, 1, :), 2, 2, 1)

    # Define the variables and add them to the ds
    for (name, data) in sim_forcing
        println(name)
        if name in ["tnhusha", "tntha", "hus", "tntva", "zg", "wa", "ua", "va", "ta", "tnhusva", "wap", "rho", "clw", "cli"]
            defVar(ds, name, Float64, ("x", "y", "z", "time"))
            ds[name][:] = repeat(reshape(data, 1, 1, size(data)...), 2, 2, 1, 1)
        elseif name ∉ ["time", "pressure_level", "z"]
            defVar(ds, name, Float64, ("x", "y", "time"))
            ds[name][:] = repeat(reshape(data, 1, 1, size(data)...), 2, 2, 1)
        end
    end

    # add coszen 
    coszen_list = get_coszen_inst.(lat, lon, tvforcing["valid_time"][:])
    defVar(ds, "coszen", Float64, ("x", "y", "z", "time"))
    ds["coszen"][:] = repeat(reshape([c[1] for c in coszen_list], 1, 1, 1, :), 2, 2, length(ds["z"]), 1)

    defVar(ds, "rsdt", Float64, ("x", "y", "z", "time"))
    ds["rsdt"][:] = repeat(reshape([c[2] for c in coszen_list], 1, 1, 1, :), 2, 2, length(ds["z"]), 1)


    # add latent and sensitble heat fluxes (currently we just set surface conditions based on temperature)
    lon_index_surf = findfirst(tv_accum["longitude"][:] .== lon)
    lat_index_surf = findfirst(tv_accum["latitude"][:] .== lat)
    matching_time_indices = findall(in(tvforcing["valid_time"][:]), tv_accum["valid_time"][:])

    defVar(ds, "hfls", Float64, ("x", "y", "z", "time"))
    defVar(ds, "hfss", Float64, ("x", "y", "z", "time"))
    slhf = -tv_accum["slhf"][lon_index_surf, lat_index_surf, matching_time_indices] / time_resolution
    sshf = -tv_accum["sshf"][lon_index_surf, lat_index_surf, matching_time_indices] / time_resolution
    ds["hfls"][:] = repeat(reshape(slhf, 1, 1, 1, :), 2, 2, length(ds["z"]), 1)
    ds["hfss"][:] = repeat(reshape(sshf, 1, 1, 1, :), 2, 2, length(ds["z"]), 1)

    # add temperature data
    lon_index_surf2 = findfirst(tv_inst["longitude"][:] .== lon)
    lat_index_surf2 = findfirst(tv_inst["latitude"][:] .== lat)
    matching_time_indices = findall(in(tvforcing["valid_time"][:]), tv_inst["valid_time"][:])

    defVar(ds, "ts", Float64, ("x", "y","z", "time"))
    skt = tv_inst["skt"][lon_index_surf2, lat_index_surf2, matching_time_indices]
    ds["ts"][:] = repeat(reshape(skt, 1, 1, 1, :), 2, 2, length(ds["z"]),  1)

    # Close the dataset
    close(ds)
end

# site 2, 23, and deep convection case
for (lat, lon) in [(-20, 285 - 360), (17, 211 - 360), (10, -135)]
    compute_forcing(lat, lon, tvforcing, tv_inst, tv_accum)
    println("sim_forcing_loc_$(lat)_$(lon).nc")
end
