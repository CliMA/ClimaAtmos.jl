import xarray as xr 
import numpy as np
import matplotlib.pyplot as plt

# extract locations and pick sites
geo = xr.open_dataset("coszen_data.nc")
minsite = 2
maxsite = 23
included_sites = geo.where((geo["site"] <=maxsite) & (geo["site"] >= minsite) & (geo["site"] != 16), drop = True)
lats = included_sites.lat
lons = included_sites.lon
sites = included_sites.site
coszen = included_sites.coszen

# northern hemisphere era5 data 
nh_column_data = xr.open_dataset("~/Downloads/era5/monthly/NH_monthly_PL.nc")
nh_surface_data = xr.open_dataset("~/Downloads/era5/monthly/NH_monthly_surface.nc")


# southern hemisphere era5 data
sh_column_data = xr.open_dataset("~/Downloads/era5/monthly/SH_monthly_PL.nc")
sh_surface_data = xr.open_dataset("~/Downloads/era5/monthly/SH_monthly_surface.nc")



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
    dx = 2 * np.pi * rearth * coslat / 360 # per degree longitude
    dy = 2 * np.pi * rearth / 360 # per degree latitude

    # compute advective tendencies
    # Temperature
    tntha = center.u * (west.t - 2 * center.t + east.t) / (2 * dx / 4) + center.v * (south.t - 2 * center.t + north.t) / (2 * dy / 4)
    # specific humidity
    tnhusha = center.u * (west.q - 2 * center.q + east.q) / (2 * dx / 4) + center.v * (south.q - 2 * center.q + north.q) / (2 * dy / 4)

    return tntha, tnhusha

#get_tendencies(lon, lat, col)[0]

def get_vertical_tendencies(column_ds, var, vertvar = "wa"):
    """
    Calculate the temperature and specific humidity vertical tendencies as a function of levels
    using vertical advection.
    """
    tntva_trend = []
    
    # Loop through each pressure level
    num_levels = column_ds[vertvar].shape[1]  # Number of vertical levels

    for i in range(num_levels):
        if i == 0:  # Bottom level (forward difference)
            tntva = column_ds[vertvar][:, i] * (column_ds[var][:, i+1] - column_ds[var][:, i]) / (column_ds.z[:, i+1] - column_ds.z[:, i])
        
        elif i == num_levels - 1:  # Top level (backward difference)
            tntva = column_ds[vertvar][:, i] * (column_ds[var][:, i] - column_ds[var][:, i-1]) / (column_ds.z[:, i] - column_ds.z[:, i-1])
        
        else:  # Middle levels (surface_dsed difference)
            tntva = column_ds[vertvar][:, i] * (column_ds[var][:, i+1] - 2 * column_ds[var][:, i] +  column_ds[var][:, i-1]) / ((column_ds.z[:, i+1] - column_ds.z[:, i-1]))

        tntva = tntva.assign_coords(pressure_level=column_ds.pressure_level[i])

        # Append the result to the trend list
        tntva_trend.append(tntva)
    
    # Convert the trend list to an xarray object, correctly indexed by pressure levels
    tntva_trend = xr.concat(tntva_trend, dim="pressure_level")
    tntva_trend = tntva_trend.assign_coords(pressure_level=column_ds.pressure_level)

    # transpose so time is the first dimension
    tntva_trend = tntva_trend.transpose("valid_time", "pressure_level")

    return tntva_trend

def get_forcing_data(cfsite, column_ds, surface_ds, geo = geo):
    loc = geo.where(geo["site"] == cfsite, drop = True)
    lat = np.round(loc.lat.values[0] / .25) * .25
    lon = np.round(loc.lon.values[0] / .25) * .25

    sitesf = surface_ds.where((surface_ds.latitude == lat) & (surface_ds.longitude == lon - 360), drop = True)
    sitecol = column_ds.where((column_ds.latitude == lat) & (column_ds.longitude == lon - 360), drop = True)

    ##### get column data #####
    # compute temperature
    R_d = 287.05  # Specific gas constant for dry air (J/(kg·K))
    g = 9.81  # Gravitational acceleration (m/s²)
    pressure = sitecol.pressure_level * 100  # Convert to Pa
    pressure_broadcasted = pressure.broadcast_like(column_ds.t)
    # Compute air density using the ideal gas law: rho = P / (R_d * T)
    rho = pressure_broadcasted / (R_d * column_ds.t)
    sitecol["rho"] = rho


    ##### get surface data #####
    #coszen = xr.open_dataset("/Users/julianschmitt/Downloads/HadGEM2-A_amip.2004-2008.07.nc", group = f"site{group}").coszen
    # sitecol["coszen"] = coszen

    # rescale TOA incident radiation to w/m2 by dividing by the time step of ERA5 (1 hour)
    sitesf["tisr"] = sitesf["tisr"] / 3600

    #### Combine data ####
    sitedata = xr.merge([sitecol[["z", "t", "rho", "u", "v", "w", "q", "coszen"]], sitesf[["slhf", "sshf", "tisr", "skt"]]])

    sitedata = sitedata.rename({"t": "ta", "u": "ua", "v": "va", "w": "wa", "q": "hus", "slhf": "hfls", "sshf": "hfss", "skt": "ts", "tisr": "rsdt"})

    # remove latitude/longitude dependence - not actually selecting a value on the meridian, just the first value of the array
    sitedata = sitedata.isel(latitude=0, longitude=0).squeeze()

    # calculate tendency terms
    sitedata["wap"] = sitedata["wa"] * sitedata["rho"]
    # temperature vertical tendency due to vertical advection
    sitedata["tntva"] = get_vertical_tendencies(sitedata, "ta")
    # specific humidity vertical tendency due to vertical advection
    sitedata["tnhusva"] = get_vertical_tendencies(sitedata, "hus")


    # compute horizontal tendencies
    tntha, tnhusha = get_horizontal_tendencies(lon - 360, lat, column_ds)

    sitedata["tntha"] = tntha
    sitedata["tnhusha"] = tnhusha

    # approximate geopotential 
    sitedata["zg"] = sitedata["z"] / 9.81
    
    return sitedata

output_file = 'era5_forcing.nc'

# for southern hemisphere sites 
# for site_id in range(2, 16): # e.g. 2-15
#     print("Running site: ", site_id)
#     site_data = get_forcing_data(site_id, sh_column_data, sh_surface_data)
#     site_data.to_netcdf(output_file, mode='a', group=f'site{site_id}')

# for northern hemisphere sites
for site_id in range(16, 24):
    site_data = get_forcing_data(site_id, nh_column_data, nh_surface_data)
    site_data["coszen"] = geo2.where((geo2.site ==23) & (geo2.time ==7), drop = True).coszen.values[0][0]
    site_data.to_netcdf(output_file, mode='a', group=f'site{site_id}')