default_diag_1M = joinpath(
    pkgdir(CA),
    "LWP_N_references",
    "diagnostic_edmfx_dycoms_rf02_column_1M.yml",
)
default_diag_2M = joinpath(
    pkgdir(CA),
    "LWP_N_references",
    "diagnostic_edmfx_dycoms_rf02_column_2M.yml",
)
default_prog_1M = joinpath(
    pkgdir(CA),
    "LWP_N_references",
    "prognostic_edmfx_dycoms_rf02_column_1M.yml",
)
default_prog_2M = joinpath(
    pkgdir(CA),
    "LWP_N_references",
    "prognostic_edmfx_dycoms_rf02_column_2M.yml",
)

diagnostic_1M_config = joinpath(pkgdir(CA), "LWP_N_config_diagnostic_1M")
diagnostic_2M_config = joinpath(pkgdir(CA), "LWP_N_config_diagnostic_2M")
prognostic_1M_config = joinpath(pkgdir(CA), "LWP_N_config_prognostic_1M")
prognostic_2M_config = joinpath(pkgdir(CA), "LWP_N_config_prognostic_2M")

output_1M =  joinpath(pkgdir(CA), "LWP_N_output_1M")
output_2M = joinpath(pkgdir(CA), "LWP_N_output_2M")