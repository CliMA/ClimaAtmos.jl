using NCDatasets
using Statistics
using Dates
using DataFrames
using Printf

# for radiation calculation 
using Insolation
import Insolation.Parameters as IP 
import ClimaParams as CP

# ta, hus, ua, va, zg, z, wap, rsdt, tntha, tnhusha, tntva, tnhusva

time_resolution = 3600 # switch to 86400 for monthly data 
# pick location, site 17 to start with 
lat = 17.
lon = 211. - 360. # convert to -180, 180

# parameters 
R_d = 287.05
g = -9.81

tvforcing = NCDataset("/scratch/julian/ERA5/tv/time_varying_site23.nc")

# copy values we don't directly need to compute
sim_forcing = Dict()
sim_forcing["time"] = tvforcing["valid_time"][:]


function get_horizontal_tendencies(lon_index, lat_index, column_ds)
    """
    Computes horizontal tendencies for temperature and specific humidity at a given location.
    """
    rearth = 6378e3
    lat_rad = deg2rad(lat)
    lon_rad = deg2rad(lon - 360)
    coslat = cos(lat_rad)
    dx = 2 * π * rearth * coslat / 360 * 0.25
    dy = 2 * π * rearth / 360 * 0.25

    # lon_index = findfirst(column_ds["longitude"][:] .== lon)
    # lat_index = findfirst(column_ds["latitude"][:] .== lat)
    # println(lon_index, lat_index)
    # get velocities at site location
    ᶜu = column_ds["u"][lat_index, lon_index, :, :]
    ᶜv = column_ds["v"][lat_index, lon_index, :, :]

    # get temperature at N S E W of center for gradient calculation
    ʷT = column_ds["t"][lat_index, lon_index - 1, :, :]
    ⁿT = column_ds["t"][lat_index + 1, lon_index, :, :]
    ˢT = column_ds["t"][lat_index - 1, lon_index, :, :]
    ᵉT = column_ds["t"][lat_index, lon_index + 1, :, :]

    # get specific humidity at N S E W of center for gradient calculation
    ʷq = column_ds["q"][lat_index, lon_index - 1, :, :]
    ⁿq = column_ds["q"][lat_index + 1, lon_index, :, :]
    ˢq = column_ds["q"][lat_index - 1, lon_index, :, :]
    ᵉq = column_ds["q"][lat_index, lon_index + 1, :, :]

    # temperature and specific humidity advective tendency at center 
    tntha = -(ᶜu .* (ᵉT .- ʷT) ./ (2dx) .+ ᶜv .* (ⁿT .- ˢT) ./ (2dy))
    tnhusha = -(ᶜu .* (ᵉq .- ʷq) ./ (2dx) .+ ᶜv .* (ⁿq .- ˢq) ./ (2dy))

    return tntha, tnhusha
end

# compute vertical advection terms
function get_vertical_tendencies(column_ds, var, vertvar = "wap")
    """
    Calculate the temperature and specific humidity vertical tendencies as a function of levels
    using vertical advection. Here we take the tendency over the geopotential height not the height in meters
    """
    deriv = zeros(size(sim_forcing["wap"]))

    for i in 1:size(sim_forcing["wap"])[1]
        if i == 1
            deriv[1, :] = sim_forcing["wap"][1, :] .* (sim_forcing[var][2, :] .- sim_forcing[var][1, :]) ./ (sim_forcing["zg"][2, :] .- sim_forcing["zg"][1, :])
        elseif i == size(sim_forcing["wap"])[1]
            deriv[end, :] = sim_forcing["wap"][end, :] .* (sim_forcing[var][end, :] .- sim_forcing[var][end-1, :]) ./ (sim_forcing["zg"][end, :] .- sim_forcing["zg"][end-1, :])
        else # centered FD 
            deriv[i, :] = sim_forcing["wap"][i, :] .* (sim_forcing[var][i+1, :] .- sim_forcing[var][i-1, :]) ./ (sim_forcing["zg"][i+1, :] .- sim_forcing["zg"][i-1, :])
        end
    end
    
    deriv 
end

FT = Float64

function get_coszen_inst(lat, lon, date,
    param_set = IP.InsolationParameters(FT),
    od = Insolation.OrbitalData())

    date = DateTime(date)

    date0 = DateTime("2000-01-01T11:58:56.816")

    S, μ = solar_flux_and_cos_sza(date, date0, od, FT(lon), FT(lat), param_set)

    return μ, S * μ
end



# find indexes for site location in pressure file 
lon_index = findfirst(tvforcing["longitude"][:] .== lon)
lat_index = findfirst(tvforcing["latitude"][:] .== lat)

sim_forcing["tntha"], sim_forcing["tnhusha"] = get_horizontal_tendencies(lon_index, lat_index, tvforcing)


# temperature = tvforcing["t"]
# pressure = tvforcing["pressure_level"] .* 100
# ρ = reshape(pressure, 1, 1, 37, 1) ./ (R_d .* temperature)
# wap = tvforcing["w"] .* ρ



sim_forcing = Dict()
sim_forcing["time"] = tvforcing["valid_time"][:]
sim_forcing["pressure_level"] = tvforcing["pressure_level"][:]

lat = 17.
lon = 211. - 360. # convert to -180, 180
lon_index = findfirst(tvforcing["longitude"][:] .== lon)
lat_index = findfirst(tvforcing["latitude"][:] .== lat)

sim_forcing["ua"] = tvforcing["u"][lon_index, lat_index, :, :]
sim_forcing["va"] = tvforcing["v"][lon_index, lat_index, :, :]
sim_forcing["wa"] = tvforcing["w"][lon_index, lat_index, :, :]
sim_forcing["hus"] = tvforcing["q"][lon_index, lat_index, :, :]
sim_forcing["ta"] = tvforcing["t"][lon_index, lat_index, :, :]
sim_forcing["zg"] = tvforcing["z"][lon_index, lat_index, :, :]


# compute subsidence
pressure = tvforcing["pressure_level"] .* 100
ρ = reshape(pressure, 37, 1) ./ (R_d .* sim_forcing["ta"])
sim_forcing["wap"] = sim_forcing["wa"] .* ρ


# compute vertical advection terms - for these terms we don't need horizontal gradients so can pass sim_forcing directly
sim_forcing["tntva"] = get_vertical_tendencies(sim_forcing, "ta", "wap")
sim_forcing["tnhusva"] = get_vertical_tendencies(sim_forcing, "hus", "wap")
sim_forcing["tntha"], sim_forcing["tnhusha"] = get_horizontal_tendencies(lon_index, lat_index, tvforcing)

sim_forcing["rho"] = ρ # pressure 
sim_forcing["z"] = tvforcing["z"][lon_index, lat_index, :, :] / (-g) # height in meters


# save sim_forcing to file within group entitled site23 
ds = Dataset("sim_forcing_site23.nc", "c")
group = defGroup(ds, "site23")


# Define the dimensions
defDim(group, "time", length(sim_forcing["time"]))
defDim(group, "pressure_level", length(sim_forcing["pressure_level"]))

# Convert DateTime to numeric values
time_ref = DateTime(1970, 1, 1)  # Reference time (Unix epoch)
time_values = Float64.(Dates.value.(sim_forcing["time"] .- sim_forcing["time"][1]) ./ 1000)

# Define time variable with attributes
defVar(group, "time", Float64, ("time",),
    attrib = [
        "units" => "hours since 1970-01-01 00:00:00",
        "calendar" => "proleptic_gregorian"
    ]
)
group["time"][:] = time_values

# Define other variables
defVar(group, "pressure_level", Float64, ("pressure_level",))
group["pressure_level"][:] = sim_forcing["pressure_level"]

# Define the variables and add them to the group
for (name, data) in sim_forcing
    if name in ["tnhusha", "tntha", "hus", "tntva", "zg", "wa", "ua", "va", "ta", "tnhusva", "wap", "z", "rho"]
        defVar(group, name, Float64, ("pressure_level", "time"))
        group[name][:, :] = data
    elseif name ∉ ["time", "pressure_level"]
        defVar(group, name, Float64, ("time", "pressure_level"))
        group[name][:, :] = data
    end
end

# add coszen 
coszen_list = get_coszen_inst.(lat, lon, tvforcing["valid_time"][:])
defVar(group, "coszen", Float64, ("time",))
group["coszen"][:] = [c[1] for c in coszen_list]
defVar(group, "rsdt", Float64, ("time",))
group["rsdt"][:] = [c[2] for c in coszen_list]

# add latent and sensitble heat fluxes (I think we'll probably just use ts and prescribed monin obukhov length)
tv_site23_surface = NCDataset("/scratch/julian/ERA5/tv/site23_surface_forcing1.nc")
lon_index_surf = findfirst(tv_site23_surface["longitude"][:] .== lon)
lat_index_surf = findfirst(tv_site23_surface["latitude"][:] .== lat)
matching_time_indices = findall(in(tvforcing["valid_time"][:]), tv_site23_surface["valid_time"][:])

defVar(group, "hfls", Float64, ("time",))
defVar(group, "hfss", Float64, ("time",))
group["hfls"][:] = - tv_site23_surface["slhf"][lon_index_surf, lat_index_surf, matching_time_indices] / time_resolution
group["hfss"][:] = - tv_site23_surface["sshf"][lon_index_surf, lat_index_surf, matching_time_indices] / time_resolution

# add temperature data - annoying a different file 
tv_site23_surface2 = NCDataset("/scratch/julian/ERA5/tv/site23_surface_forcing2.nc")
defVar(group, "ts", Float64, ("time",))
group["ts"][:] = tv_site23_surface2["skt"][lon_index_surf, lat_index_surf, matching_time_indices]


# Close the dataset
close(ds)