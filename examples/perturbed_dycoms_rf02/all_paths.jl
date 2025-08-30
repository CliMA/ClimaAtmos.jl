# Dependencies 
import ClimaAtmos as CA

base_path = joinpath(pkgdir(CA), "LWP_N")
mkpath(base_path)

base_reference_path = joinpath(base_path, "references")
mkpath(base_reference_path)

default_diag_1M = joinpath(
    pkgdir(CA),
    base_reference_path,
    "diagnostic_edmfx_dycoms_rf02_column_1M.yml",
)
default_diag_2M = joinpath(
    pkgdir(CA),
    base_reference_path,
    "diagnostic_edmfx_dycoms_rf02_column_2M.yml",
)
default_prog_1M = joinpath(
    pkgdir(CA),
    base_reference_path,
    "prognostic_edmfx_dycoms_rf02_column_1M.yml",
)
default_prog_2M = joinpath(
    pkgdir(CA),
    base_reference_path,
    "prognostic_edmfx_dycoms_rf02_column_2M.yml",
)

diagnostic_1M_config = joinpath(pkgdir(CA), base_path, "config_diagnostic_1M")
diagnostic_2M_config = joinpath(pkgdir(CA), base_path, "config_diagnostic_2M")
prognostic_1M_config = joinpath(pkgdir(CA), base_path, "config_prognostic_1M")
prognostic_2M_config = joinpath(pkgdir(CA), base_path, "config_prognostic_2M")

mkpath(diagnostic_1M_config)
mkpath(diagnostic_2M_config)
mkpath(prognostic_1M_config)
mkpath(prognostic_2M_config)

output_1M = joinpath(pkgdir(CA), base_path, "output_1M")
output_2M = joinpath(pkgdir(CA), base_path, "output_2M")
