using Dates
using Statistics

using Insolation
import Insolation.Parameters as IP 
import ClimaParams as CP

using NCDatasets

FT = Float64

geo = NCDataset("data_processing/geolocation.nc")


function get_weighted_coszen(lat, lon, month, timezone=-10., year = 2008, ndays = 31;
    param_set = IP.InsolationParameters(FT),
    od = Insolation.OrbitalData())

    date = DateTime(year, month, 01)

    ntimes = ndays * 24 * 60 # sample every minute
    hours = collect(range(0, 24 * ndays, length = ntimes))
    insol = zeros(ntimes) 
    sza = zeros(ntimes)
    date0 = DateTime("2000-01-01T11:58:56.816")

    for (i, hr) in enumerate(hours)
        h = Int(round(hr + timezone))
        m = Int(round((hr + timezone - h) * 60))

        datetime = date + Dates.Hour(h) + Dates.Minute(m)
        S, mu = solar_flux_and_cos_sza(datetime, date0, od, FT(lon), FT(lat), param_set)
        insol[i] = S * mu 
        sza[i] = rad2deg(acos(mu))
    end

    v = cos.(deg2rad.(sza)) .* insol/ sum(insol)

    return sum(v)
end

# jan = get_weighted_coszen.(geo["lat"][:], geo["lon"][:], 1)
# april = get_weighted_coszen.(geo["lat"][:], geo["lon"][:], 4)
# july = get_weighted_coszen.(geo["lat"][:], geo["lon"][:], 7)
# oct = get_weighted_coszen.(geo["lat"][:], geo["lon"][:], 10)

coszen_sites = map(x -> get_weighted_coszen.(geo["lat"][:], geo["lon"][:], x), 1:12)

# make a NCdataset indexed by geo site lat and lon with cozsen data for each of the 4 selected months
using NCDatasets

# Create a new NetCDF file
ds = NCDataset("coszen_data.nc", "c")

# Define dimensions
site_dim = defDim(ds, "site", length(geo["site"][:]))
time_dim = defDim(ds, "date", 12)

# Define variables
site_var = defVar(ds, "site", Int, ("site",))
lat_var = defVar(ds, "lat", Float64, ("site",))
lon_var = defVar(ds, "lon", Float64, ("site",))
time_var = defVar(ds, "date", Int, ("date",))
coszen_var = defVar(ds, "coszen", Float64, ("site", "date"))

# Assign data to variables
site_var[:] = geo["site"][:]
lat_var[:] = geo["lat"][:]
lon_var[:] = geo["lon"][:]
time_var[:] = 1:12

# Add coszen data to dataset
coszen_var[:, :] = hcat(coszen_sites...)

# Close the dataset
close(ds)