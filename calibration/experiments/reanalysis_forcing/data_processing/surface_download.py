import cdsapi

dataset = "reanalysis-era5-single-levels-monthly-means"
request = {
    "product_type": ["monthly_averaged_reanalysis"],
    "variable": [
        "2m_temperature",
        "sea_surface_temperature",
        "total_precipitation",
        "skin_temperature",
        "surface_latent_heat_flux",
        "surface_sensible_heat_flux",
        "toa_incident_solar_radiation"
    ],
    "year": ["2008"],
    "month": [
        "01", "04", "07",
        "10"
    ],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [39, -150, 16, -122]# [-7, -127, -21, -71]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
