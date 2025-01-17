jobs = (
 ("config/model_configs/single_column_radiative_equilibrium_gray.yml","single_column_radiative_equilibrium_gray")
 ("config/model_configs/single_column_radiative_equilibrium_clearsky.yml","single_column_radiative_equilibrium_clearsky")
 ("config/model_configs/single_column_radiative_equilibrium_clearsky_prognostic_surface_temp.yml","single_column_radiative_equilibrium_clearsky_prognostic_surface_temp")
 ("config/model_configs/single_column_radiative_equilibrium_allsky_idealized_clouds.yml","single_column_radiative_equilibrium_allsky_idealized_clouds")
)

include(joinpath("test", "groups", "group_driver.jl"))
