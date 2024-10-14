import cdsapi

dataset = "reanalysis-era5-pressure-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "fraction_of_cloud_cover",
        "geopotential",
        "relative_humidity",
        "specific_cloud_ice_water_content",
        "specific_cloud_liquid_water_content",
        "specific_humidity",
        "specific_rain_water_content",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "vertical_velocity"
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
    "pressure_level": [
        "1", "2", "3",
        "5", "7", "10",
        "20", "30", "50",
        "70", "100", "125",
        "150", "175", "200",
        "225", "250", "300",
        "350", "400", "450",
        "500", "550", "600",
        "650", "700", "750",
        "775", "800", "825",
        "850", "875", "900",
        "925", "950", "975",
        "1000"
    ],
    "data_format": "netcdf",
    "download_format": "zip",
    "area": [39, -150, 16, -122]#[-7, -127, -21, -71]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
