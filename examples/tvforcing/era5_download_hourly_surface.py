import cdsapi
import sys

dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "2m_temperature",
        "sea_surface_temperature",
        "total_precipitation",
        "clear_sky_direct_solar_radiation_at_surface",
        "instantaneous_surface_sensible_heat_flux",
        "surface_latent_heat_flux",
        "surface_net_solar_radiation",
        "surface_net_solar_radiation_clear_sky",
        "surface_net_thermal_radiation",
        "surface_net_thermal_radiation_clear_sky",
        "surface_sensible_heat_flux",
        "surface_solar_radiation_downward_clear_sky",
        "surface_solar_radiation_downwards",
        "surface_thermal_radiation_downward_clear_sky",
        "surface_thermal_radiation_downwards",
        "toa_incident_solar_radiation",
        "top_net_solar_radiation",
        "top_net_solar_radiation_clear_sky",
        "top_net_thermal_radiation",
        "top_net_thermal_radiation_clear_sky",
        "total_sky_direct_solar_radiation_at_surface",
        "total_column_cloud_ice_water",
        "total_column_cloud_liquid_water",
        "skin_temperature"
    ],
    "year": ["2007"],
    "month": ["07"],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "time": [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00"
    ],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [17.5, -151, 16.5, -147]
}

# if len(sys.argv) < 2:
#     print("Usage: python script.py <target_file>")
#     sys.exit(1)

# target = sys.argv[1]

client = cdsapi.Client()
client.retrieve(dataset, request).download()
