# Extracts integrated observations at cfsites specified by geo file 
import xarray as xr 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

vars = ["mtnswrf", "mtnlwrf", "mtnswrfcs", "mtnlwrfcs", "tclw", "tciw", "mtpr"]

name_mapper = {
    "mtnswrf": "rsut",
    "mtnlwrf": "rlut",
    "mtnswrfcs": "rsutcs",
    "mtnlwrfcs": "rlutcs",
    "tclw": "clwvi",
    "tciw": "clivi",
    "mtpr": "pr"
}

# flip sign of these variables to be consistent with clima 
flip_sign = ["mtnlwrf", "mtnlwrfcs"]


geo = xr.open_dataset("/scratch/julian/ERA5/coszen_data.nc")

integrated = xr.open_mfdataset("/scratch/julian/ERA5/integrated_*.nc")

# time average last 20 years to be comparable to forcing data 
integrated = integrated.sel(valid_time = slice("2000-01-01", "2019-12-31")).groupby("valid_time.month").mean()

output_file = 'integrated_cfsite_obs_data.nc'

for site in geo.site.values:
    loc = geo.where(geo.site ==site, drop = True)
    lat = loc.lat.values[0]
    lon = loc.lon.values[0]
    #for month in range(1, 13):
    integrated_ds = integrated.sel(latitude = lat, longitude = lon, method = "nearest", drop = True)[vars]
    # flip sign of variables to be consistent with clima values
    for var in flip_sign:
        integrated_ds[var] = -1 * integrated_ds[var]
    # rename to clima variables based on dictionary 
    integrated_ds = integrated_ds.rename(name_mapper)

    integrated_ds.to_netcdf(output_file, mode='a', group=f'site{site}')
