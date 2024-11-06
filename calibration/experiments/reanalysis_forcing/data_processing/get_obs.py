import xarray as xr 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# calibration vars
vars = ["t", "q", "clwc", "z"]

geo = xr.open_dataset("data_processing/coszen_data.nc")
geo_45 = geo.where((geo.lat > -45) & (geo.lat < 45) & (geo.lon > 180) & (geo.lon < 290), drop=True)

#surface = xr.open_mfdataset("/Users/julianschmitt/Downloads/era5/monthly/global_surface_20yr.nc")

pressure_pt1 = xr.open_mfdataset("/Users/julianschmitt/Downloads/era5/monthly/global_pressure_20yr_pt1.nc")
pressure_pt2 = xr.open_mfdataset("/Users/julianschmitt/Downloads/era5/monthly/global_pressure_20yr_pt2.nc")


pressure_pt1 = pressure_pt1.where((pressure_pt1.latitude > -45) & (pressure_pt1.latitude < 45) & (pressure_pt1.longitude > 180) & (pressure_pt1.longitude < 290), drop =True)
pressure_pt2 = pressure_pt2.where((pressure_pt2.latitude > -45) & (pressure_pt2.latitude < 45) & (pressure_pt2.longitude > 180) & (pressure_pt2.longitude < 290), drop =True)
#surface = surface.where((surface.latitude > -45) & (surface.latitude < 45) & (surface.longitude > 180) & (surface.longitude < 290), drop =True)
# monthly time mean
pressure_pt1["date"] = pd.to_datetime(pressure_pt1["date"].astype(str), format="%Y%m%d")
pressure_pt2["date"] = pd.to_datetime(pressure_pt2["date"].astype(str), format="%Y%m%d")
#surface["date"] = pd.to_datetime(surface["date"].astype(str), format="%Y%m%d")
#surface_avg = surface.groupby("date.month").mean()

# combine datasets
pressure_pt1_avg = pressure_pt1.groupby("date.month").mean()
pressure_pt2_avg = pressure_pt2.groupby("date.month").mean()
pressure_avg = xr.concat([pressure_pt1_avg, pressure_pt2_avg], dim="month").sortby("month")

# surface_avg = surface_avg.load()
pressure_avg = pressure_avg.load()

pressure = xr.open_dataset("/Users/julianschmitt/Downloads/era5/monthly/pressure_avg_monthly_45.nc")

# create a dataset to store groups for eac cfsite 
output_file = 'era5_cfsite_obs_data.nc'
# loop through each cfsite
for site in geo_45.site.values:
    loc = geo_45.where(geo_45.site ==site, drop = True)
    lat = loc.lat.values[0]
    lon = loc.lon.values[0]
    print("Running site: ", lat, lon)
    for month in range(1, 13):
        pressure_avg_ds = pressure_avg.sel(latitude = lat, longitude = lon, method = "nearest", drop = True)[vars]
        pressure_avg_ds = pressure_avg_ds.rename({"clwc": "clw", "t": "ta", "q": "hus"}) # rename to clima variable names
        pressure_avg_ds.to_netcdf(output_file, mode='a', group=f'site{site}')

