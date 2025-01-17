import ClimaAtmos as CA

jobs = (
    "config/model_configs/plane_agnesi_mountain_test_uniform.yml",
    "plane_agnesi_mountain_test_uniform",
    "config/model_configs/plane_agnesi_mountain_test_stretched.yml",
    "plane_agnesi_mountain_test_stretched",
    "config/model_configs/plane_density_current_test.yml",
    "plane_density_current_test",
)

include(joinpath(pkgdir(CA), "test", "groups", "group_driver.jl"))
