# Forward-only grid: cases × quadrature_order × subcell geometric variance flag.
#
#   julia --project=.. scripts/sweep_forward_runs.jl
#
# Set `CASES`, `N_LIST`, `VARFIX` in the script or via ENV if you extend this file.
#
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

import ClimaAtmos as CA
import ClimaComms
import YAML

const EXPERIMENT_DIR = dirname(@__DIR__) |> abspath

function run_one(;
    case_yaml::AbstractString,
    scm_toml::AbstractString,
    n_quad::Int,
    varfix::Bool,
    out_subdir::AbstractString,
)
    cfg = YAML.load_file(joinpath(EXPERIMENT_DIR, case_yaml))
    cfg["output_dir"] = joinpath(EXPERIMENT_DIR, "simulation_output", out_subdir)
    mkpath(cfg["output_dir"])
    cfg["quadrature_order"] = n_quad
    cfg["sgs_quadrature_subcell_geometric_variance"] = varfix
    cfg["toml"] = [joinpath(pkgdir(CA), "toml", scm_toml)]
    cfg["output_default_diagnostics"] = get(cfg, "output_default_diagnostics", false)
    atmos_config = CA.AtmosConfig(
        cfg;
        comms_ctx = ClimaComms.SingletonCommsContext(),
    )
    sim = CA.get_simulation(atmos_config)
    CA.solve_atmos!(sim)
    @info "Finished run" out = cfg["output_dir"]
    return sim
end

function main()
    # TRMM + DYCOMS RF01 (extend as needed)
    grid = [
        (
            "model_configs/trmm_column_varquad.yml",
            "prognostic_edmfx_implicit_scm_calibrated_5_cases_shallow_deep_v1.toml",
            "TRMM_LBA",
        ),
        (
            "model_configs/dycoms_rf01_column_varquad.yml",
            "prognostic_edmfx.toml",
            "DYCOMS_RF01",
        ),
    ]
    n_list = 1:10
    for (yml, toml, cas) in grid, n in n_list, vf in (false, true)
        tag = vf ? "varfix_on" : "varfix_off"
        sub = joinpath(cas, "N_$(n)", tag, "forward_only")
        @info "Running" cas n vf
        run_one(;
            case_yaml = yml,
            scm_toml = toml,
            n_quad = n,
            varfix = vf,
            out_subdir = sub,
        )
    end
end

main()
