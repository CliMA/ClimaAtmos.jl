# Create ERA5 forcing file for CFSite SCM from ERA5 data downloaded from ECMWF CDS API.
# Need to change file pathing/ define coszen data (see other scripts for computation)
# also need to change short wave period to 1 hour for hourly data and 1 day for monthly data (ugh)

time_resolution = 86400 # for monthly data; change to 3600 for hourly data

import xarray as xr 
import numpy as np
import pandas as pd

months = [1, 4, 7, 10]
lat_min = -45
lat_max = 45
lon_min = 100
lon_max = 290

# extract locations and pick sites (lets just look between midlatitudes for now)
geo = xr.open_dataset("coszen_data.nc")
pressure_avg = xr.open_dataset("~/Downloads/era5/monthly/pressure_avg_monthly_45.nc") # already preprocessed since so big
surface = xr.open_mfdataset("/Users/julianschmitt/Downloads/era5/monthly/global_surface_20yr.nc")
# pressure = xr.open_mfdataset("/Users/julianschmitt/Downloads/era5/monthly/global_pressure_20yr.nc")


def get_horizontal_tendencies(lon, lat, column_ds):
    """Computes horizontal tendencies for temperature and specific humidity at a given location.
    Args:
    lon: site longitude
    lat: site latitude
    column_ds: xarray dataset containing ERA5 data at pressure levels 
    """
    west = column_ds.where((column_ds.latitude == lat) & (column_ds.longitude == lon - .25), drop = True).isel(latitude=0, longitude=0).squeeze()
    east = column_ds.where((column_ds.latitude == lat) & (column_ds.longitude == lon + .25), drop = True).isel(latitude=0, longitude=0).squeeze()
    north = column_ds.where((column_ds.latitude == lat + .25) & (column_ds.longitude == lon), drop = True).isel(latitude=0, longitude=0).squeeze()
    south = column_ds.where((column_ds.latitude == lat - .25) & (column_ds.longitude == lon), drop = True).isel(latitude=0, longitude=0).squeeze()
    center = column_ds.where((column_ds.latitude == lat) & (column_ds.longitude == lon), drop = True).isel(latitude=0, longitude=0).squeeze()
    
    # convert to radians for gradient calculation 
    rearth = 6378e3
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon - 360)
    coslat = np.cos(lat)
    dx = 2 * np.pi * rearth * coslat / 360 * .25# per .25 degree longitude 
    dy = 2 * np.pi * rearth / 360 * .25# per .25 degree latitude

    # compute advective tendencies
    # Temperature
    tntha = center.u * (east.t - west.t) / (2 * dx) + center.v * (north.t - south.t) / (2 * dy)
    # specific humidity
    tnhusha = center.u * (east.q - west.q) / (2 * dx) + center.v * (north.q - south.q) / (2 * dy)

    return tntha, tnhusha

#get_tendencies(lon, lat, col)[0]

def get_vertical_tendencies(column_ds, var, vertvar = "wa"):
    """
    Calculate the temperature and specific humidity vertical tendencies as a function of levels
    using vertical advection. Here we take the tendency over the geopotential height not the height in meters
    """
    tntva_trend = []
    # Loop through each pressure level
    num_levels = column_ds[vertvar].shape[0]  # Number of vertical levels

    for i in range(num_levels):
        if i == 0:  # Bottom level (forward difference)
            tntva = column_ds[vertvar][i] * (column_ds[var][i+1] - column_ds[var][i]) / (column_ds.zg[i+1] - column_ds.zg[i])
        
        elif i == num_levels - 1:  # Top level (backward difference)
            tntva = column_ds[vertvar][i] * (column_ds[var][i] - column_ds[var][i-1]) / (column_ds.zg[i] - column_ds.zg[i-1])
        
        else:  # Middle levels (surface_dsed difference)
            tntva = column_ds[vertvar][i] * (column_ds[var][i+1] - column_ds[var][i-1]) / ((column_ds.zg[i+1] - column_ds.zg[i-1]))

        tntva = tntva.assign_coords(pressure_level=column_ds.pressure_level[i])

        # Append the result to the trend list
        tntva_trend.append(tntva)
    
    # Convert the trend list to an xarray object, correctly indexed by pressure levels
    tntva_trend = xr.concat(tntva_trend, dim="pressure_level")
    tntva_trend = tntva_trend.assign_coords(pressure_level=column_ds.pressure_level)

    return tntva_trend

def get_forcing_data(cfsite, column_ds, surface_ds, geo = geo):
    loc = geo.where(geo["site"] == cfsite, drop = True)
    lat = np.round(loc.lat.values[0] / .25) * .25
    lon = np.round(loc.lon.values[0] / .25) * .25

    sitesf = surface_ds.where((surface_ds.latitude == lat) & (surface_ds.longitude == lon), drop = True)
    sitecol = column_ds.where((column_ds.latitude == lat) & (column_ds.longitude == lon), drop = True)

    ##### get column data #####
    # compute temperature
    R_d = 287.05  # Specific gas constant for dry air (J/(kg·K))
    g = 9.807  # Gravitational acceleration (m/s²)
    pressure = sitecol.pressure_level * 100  # Convert to Pa
    pressure_broadcasted = pressure.broadcast_like(column_ds.t)
    # Compute air density using the ideal gas law: rho = P / (R_d * T)
    rho = pressure_broadcasted / (R_d * column_ds.t)
    sitecol["rho"] = rho


    ##### get surface data #####
    #coszen = xr.open_dataset("/Users/julianschmitt/Downloads/HadGEM2-A_amip.2004-2008.07.nc", group = f"site{group}").coszen
    # sitecol["coszen"] = coszen
    coszen = geo.where(geo["site"] == cfsite, drop = True)

    # rescale TOA incident radiation to w/m2 by dividing by the time step of ERA5 (1 hour)
    sitesf["tisr"] = sitesf["tisr"] / time_resolution

    #### Combine data ####
    sitedata = xr.merge([sitecol[["z", "t", "rho", "u", "v", "w", "q"]], 
                         sitesf[["slhf", "sshf", "tisr", "skt"]],
                         coszen[["coszen"]]])
    
    # convert sensible and latent heat fluxes to W/m2; flip sign to match climaatmos practices 
    sitedata["slhf"] = - sitedata["slhf"] / time_resolution
    sitedata["sshf"] = - sitedata["sshf"] / time_resolution

    sitedata = sitedata.rename({"t": "ta", "u": "ua", "v": "va", "w": "wa", "q": "hus", "slhf": "hfls", "sshf": "hfss", "skt": "ts", "tisr": "rsdt", "z": "zg"})
    sitedata["z"] = sitedata["zg"] / g # convert geopotential (zg) to height in meters (z)

    # remove latitude/longitude dependence
    sitedata = sitedata.squeeze()

    # calculate tendency terms
    sitedata["wap"] = sitedata["wa"] * sitedata["rho"]
    # temperature vertical tendency due to vertical advection
    sitedata["tntva"] = get_vertical_tendencies(sitedata, "ta")
    # specific humidity vertical tendency due to vertical advection
    sitedata["tnhusva"] = get_vertical_tendencies(sitedata, "hus")


    # compute horizontal tendencies
    tntha, tnhusha = get_horizontal_tendencies(lon, lat, column_ds)

    sitedata["tntha"] = tntha
    sitedata["tnhusha"] = tnhusha
    
    return sitedata

# process data to region of interest for some computational efficiency (for now)
# pressure = pressure.where((pressure.latitude > lat_min) & (pressure.latitude < lat_max) & (pressure.longitude > lon_min) & (pressure.longitude < lon_max), drop =True)
surface = surface.where((surface.latitude > lat_min) & (surface.latitude < lat_max) & (surface.longitude > lon_min) & (surface.longitude < lon_max), drop =True)
geo = geo.where((geo.lat > lat_min) & (geo.lat < lat_max) & (geo.lon > lon_min) & (geo.lon < lon_max), drop =True)

# # monthly time mean
# pressure["date"] = pd.to_datetime(pressure["date"].astype(str), format="%Y%m%d")
surface["date"] = pd.to_datetime(surface["date"].astype(str), format="%Y%m%d")
# pressure_avg = pressure.groupby("date.month").mean()
surface_avg = surface.groupby("date.month").mean()

# # compute
surface_avg = surface_avg.load()
# pressure_avg = pressure_avg.load()

# loop through months
for month in range(1, 2):
    output_file = f'era5_monthly_forcing_{month}.nc'

    pressure_ds = pressure_avg.sel(month = month, drop = True)
    surface_ds = surface_avg.sel(month = month, drop = True)
    geo_ds = geo.sel(date = month, drop=True)

    for site_id in geo_ds.site.values:
        print("Running site: ", site_id)
        try:
            site_data = get_forcing_data(site_id, pressure_ds, surface_ds, geo_ds)
            site_data.to_netcdf(output_file, mode='a', group=f'site{site_id}')
        except:
            print("Error processing site: ", site_id)











# output_file = 'era5_monthly_forcing.nc'

# # for southern hemisphere sites 
# for site_id in range(2, 16): # e.g. 2-15
#     print("Running site: ", site_id)
#     site_data = get_forcing_data(site_id, sh_column_data, sh_surface_data)
#     site_data.to_netcdf(output_file, mode='a', group=f'site{site_id}')

# # for northern hemisphere sites
# for site_id in range(16, 24):
#     print("Running site: ", site_id)
#     site_data = get_forcing_data(site_id, nh_column_data, nh_surface_data)
#     site_data.to_netcdf(output_file, mode='a', group=f'site{site_id}')