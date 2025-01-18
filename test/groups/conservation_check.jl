import ClimaAtmos as CA

jobs = (
    ("config/model_config/sphere_baroclinic_wave_conservation.yml",
    "sphere_baroclinic_wave_conservation"),
    ("config/model_config/sphere_baroclinic_wave_equilmoist_conservation.yml",
    "sphere_baroclinic_wave_equilmoist_conservation"),
    ("config/model_config/sphere_baroclinic_wave_equilmoist_conservation_ft64.yml",
    "sphere_baroclinic_wave_equilmoist_conservation_ft64"),
    ("config/model_config/sphere_baroclinic_wave_equilmoist_conservation_source.yml",
    "sphere_baroclinic_wave_equilmoist_conservation_source"),
)

include(joinpath(pkgdir(CA), "test", "groups", "group_driver.jl"))
