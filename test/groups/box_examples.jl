import ClimaAtmos as CA
jobs = (
    (
        "config/model_configs/box_hydrostatic_balance_rhoe.yml",
        "box_hydrostatic_balance_rhoe",
    ),
    (
        "config/model_configs/box_density_current_test.yml",
        "box_density_current_test",
    ),
    (
        "config/model_configs/rcemipii_box_diagnostic_edmfx.yml",
        "rcemipii_box_diagnostic_edmfx",
    ),
)

include(joinpath(pkgdir(CA), "test", "groups", "group_driver.jl"))
