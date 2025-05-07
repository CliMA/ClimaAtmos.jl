using CairoMakie
using ClimaCalibrate
import ClimaCalibrate as CAL
import ClimaAtmos as CA
import EnsembleKalmanProcesses as EKP
import YAML
import TOML

import JLD2
using LinearAlgebra


include("helper_funcs.jl")
include("observation_map.jl")
include("model_interface.jl")
experiment_dir = dirname(Base.active_project())
const experiment_config = YAML.load_file(joinpath(experiment_dir, "experiment_config.yml"))
const model_interface = joinpath(experiment_dir, "model_interface.jl")


for (key, value) in experiment_config
    @eval const $(Symbol(key)) = $value
end
# load configs and directories
model_config_dict = YAML.load_file(model_config)
atmos_config = CA.AtmosConfig(model_config_dict)

start_dates, lats, lons, convection_type, num_sites = get_era5_calibration_library()
ref_paths = []
obs_vec = []

zc_model = get_z_grid(atmos_config; z_max)

fig = Figure(size = (1600, 1200))  

for i in 1:20
    row = fld(i - 1, 5) + 1
    col = mod(i - 1, 5) + 1

    lat = lats[i]
    lon = lons[i]

    forcing_file_path = CA.get_external_forcing_file_path(
        Dict(
            "start_date" => start_dates[i],
            "site_latitude" => lat,
            "site_longitude" => lon,
        ),
    )
    obs_start = Dates.DateTime(start_dates[i], "yyyymmdd") +
        Dates.Second(g_t_start_sec)
    obs_end = Dates.DateTime(start_dates[i], "yyyymmdd") +
        Dates.Second(g_t_end_sec)

    tt = get_Σ_obs(
        forcing_file_path,
        experiment_config["y_var_names"],
        obs_start,
        obs_end;
        norm_factors_dict = norm_factors_by_var,
    )

    m = mean(tt, dims = 2)[:]
    s = std(tt, dims = 2)[:]

    ax = Axis(
        fig[row, col],
        title = "Lat = $(round(lat, digits=2)), Lon = $(round(lon, digits=2))"
    )
    lines!(ax, m)
    band!(ax, 1:length(m), m .- s, m .+ s, alpha = 0.5)
end

save("covariance_width.png", fig)


