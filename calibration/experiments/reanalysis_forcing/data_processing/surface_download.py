import cdsapi

dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
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
    "month": ["07"],
    "day": [
        "01", "02", "03",
        "04", "05"
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
    "area": [-7, -127, -21, -71]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
