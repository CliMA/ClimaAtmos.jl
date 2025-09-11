import ClimaAtmos as CA
import YAML
import ClimaComms
@static pkgversion(ClimaComms) >= v"0.6" && ClimaComms.@import_required_backends
using ClimaUtilities.ClimaArtifacts
import ClimaCalibrate: path_to_ensemble_member

import ClimaCalibrate as CAL
import EnsembleKalmanProcesses as EKP
using JLD2

using TOML 
using Glob
using NCDatasets
import YAML


#model_config = "diagnostic_edmfx_diurnal_scm_exp_noneq_1M.yml"
model_config = "diagnostic_edmfx_diurnal_scm_imp_noneq_1M_mixed_phase_site.yml"
output_dir = "/home/oalcabes/SO_singlecolumn"

# CHOOSE ALL SITES IN THE SOUTHERN OCEAN between -30 and -60
ds = NCDataset("/home/oalcabes/coszen_data.nc")

deep_sites = (collect(30:33)..., collect(66:70)..., 82, 92, 94, 96, 99, 100)
shallow_sites = setdiff(collect(1:119), deep_sites)

lats, lons = [], []
for site in 1:119
    if (ds["lat"][site] < -30 && ds["lat"][site] > -60)
        push!(lats, ds["lat"][site])
        push!(lons, (ds["lon"][site] + 180.0) % 360.0 - 180.0)
    end
end

start_date = "20071001"

fast_timescale = false

for (site_index, (lat, lon)) in enumerate(zip(lats, lons))

    config_dict = YAML.load_file(model_config)

    config_dict["site_latitude"] = lat
    config_dict["site_longitude"] = lon
    config_dict["start_date"] = start_date

    member_path = joinpath(output_dir, "location_$(lat)_$(lon)_$(start_date)")
    @info("saving output to ", member_path)
    config_dict["output_dir"] = member_path

    if fast_timescale
        truth_toml = "toml/truth_fast.toml"
    else
        truth_toml = "toml/truth.toml"
    end

    # load configs and directories -- running truth!
    push!(config_dict["toml"], truth_toml)
    @show config_dict["toml"]
    atmos_config = CA.AtmosConfig(config_dict) # ADD PARAM DICT HERE W TRUTH VALS
    diag_sim = CA.AtmosSimulation(atmos_config)
    CA.solve_atmos!(diag_sim)
    truth_out_dir = diag_sim.output_dir
end
