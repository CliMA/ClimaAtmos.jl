# Build `observations_reference.jld2` from a single deterministic ClimaAtmos run (truth y).
#
#   julia --project=. generate_observations_reference.jl
#
import Pkg
Pkg.activate(@__DIR__)

import ClimaAtmos as CA
import ClimaCalibrate as CAL # load before observation_map.jl (extends CAL.observation_map)
import ClimaComms
import YAML
import JLD2

function reference_config_dict(experiment_dir::AbstractString)
    expc = YAML.load_file(joinpath(experiment_dir, "experiment_config.yml"))
    cfg = YAML.load_file(joinpath(experiment_dir, expc["model_config_path"]))
    ref_out = joinpath(experiment_dir, "simulation_output", "_reference_truth")
    mkpath(ref_out)
    cfg["output_dir"] = ref_out
    cfg["quadrature_order"] = expc["quadrature_order"]
    cfg["sgs_quadrature_subcell_geometric_variance"] =
        expc["sgs_quadrature_subcell_geometric_variance"]
    cfg["toml"] = [joinpath(pkgdir(CA), "toml", expc["scm_toml"])]
    cfg["output_default_diagnostics"] = get(cfg, "output_default_diagnostics", false)
    return cfg, expc
end

function main()
    experiment_dir = @__DIR__
    cfg, expc = reference_config_dict(experiment_dir)
    atmos_config = CA.AtmosConfig(
        cfg;
        comms_ctx = ClimaComms.SingletonCommsContext(),
    )
    sim = CA.get_simulation(atmos_config)
    CA.solve_atmos!(sim)
    include(joinpath(experiment_dir, "observation_map.jl"))
    z_elem = Int(cfg["z_elem"])
    y_template = ones(Float64, z_elem)
    active = joinpath(cfg["output_dir"], "output_active")
    y = process_member_column(active, y_template)
    out_jld = joinpath(experiment_dir, expc["observations_path"])
    JLD2.save_object(out_jld, y)
    @info "Wrote observations" out_jld length(y)
    return y
end

main()
