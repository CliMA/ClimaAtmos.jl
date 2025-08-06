cd("calibration/experiments/tv_reanalysis/diurnal")
using Pkg; Pkg.activate(".")
using ClimaAnalysis
import ClimaAtmos as CA
import YAML
import ClimaCalibrate as CAL
using Plots
using Dates
using NCDatasets
using NaNStatistics
using ClimaUtilities.ClimaArtifacts

include("helper_funcs.jl")
include("observation_map.jl")

iteration = "iteration_001"
config = "config_20"

# load experiment config
experiment_config = YAML.load_file("experiment_config.yml")

# unpack all experiment_config vars into scope
for (key, value) in experiment_config
    @eval $(Symbol(key)) = $value
end

# Now you can use the variables directly:
# g_start, g_end, ensemble_size, output_dir, y_var_names, etc. are all available

# Get site info from the first available member
member = "member_001"
sim_path = joinpath(output_dir, iteration, member, config, "output_active", ".yml")
sim_dict = YAML.load_file(sim_path)
lat = sim_dict["site_latitude"]
lon = sim_dict["site_longitude"]
start_date = sim_dict["start_date"]
@info "Site latitude: $lat, longitude: $lon, start_date: $start_date"

# Function to determine season from start date
function get_season(start_date_str)
    season_map = Dict(
        "20070101" => "Winter",
        "20070401" => "Spring", 
        "20070701" => "Summer",
        "20071001" => "Fall"
    )
    return get(season_map, start_date_str, "Unknown")
end

# These variables are now directly available from the config unpacking:
# y_var_names, norm_factors_by_var, log_vars, g_t_start_sec, g_t_end_sec, etc.

all_sim_data = Dict(var_name => [] for var_name in y_var_names)

for m in 1:ensemble_size
    member_name = "member_$(lpad(m, 3, '0'))"
    member_path = joinpath(output_dir, iteration, member_name)
    
    # Check if this member/config combination exists
    sim_dir = joinpath(member_path, config, "output_0000")
    if !isdir(sim_dir)
        @info "Skipping member $m - directory not found: $sim_dir"
        continue
    end
    
    try
        for var_name in y_var_names
            simdir = SimDir(sim_dir)
            var_data = process_profile_variable(
                simdir,
                var_name;
                reduction = "inst",
                t_start = g_t_start_sec,
                t_end = g_t_end_sec,
                z_max = nothing,
                norm_factors_dict = nothing,  # Don't normalize for plotting physical space
                log_vars = [],
            )
            push!(all_sim_data[var_name], var_data)
        end
        @info "Loaded data for member $m"
    catch e
        @warn "Failed to load data for member $m: $e"
    end
end

@info "Successfully loaded data for $(length(all_sim_data["lwp"])) ensemble members"

obs_start = Dates.DateTime(start_date, "yyyymmdd") 
obs_end = Dates.DateTime(start_date, "yyyymmdd") + Dates.Month(1)

function get_modis_obs(
    var_name,
    lat, 
    lon,
    obs_start,
    obs_end;
    normalize = false,
    norm_factors_dict = nothing,
)
    # load modis data from artifact
    modis_data = NCDataset(joinpath(@clima_artifact("modis_lwp_iwp"), "modis_lwp_iwp.nc"))

    # get the index of the closest lat and lon
    lat_idx = findmin(abs.(modis_data["latitude"][:] .- lat))[2]
    lon_idx = findmin(abs.(modis_data["longitude"][:] .- lon))[2]

    # find all time indices within the obs_start and obs_end
    time_idx = findall(modis_data["time"][:] .>= obs_start .&& modis_data["time"][:] .< obs_end)
    @info "Found $(length(time_idx)) time indices at $(lat), $(lon)"
    # get the data
    obs_data = modis_data[var_name][time_idx, lat_idx, lon_idx]

    # time mean obs data 
    obs_data = nanmean(obs_data, dims = 1)[1] # its a single value for scm model 

    return obs_data
end
all_sim_data
get_modis_obs("lwp", lat, lon, obs_start, obs_end)
get_modis_obs("iwp", lat, lon, obs_start, obs_end)

obs_data = Dict()
for var_name in y_var_names
    obs_var = get_modis_obs(
        var_name,
        lat,
        lon,
        obs_start,
        obs_end;
        normalize = false,  # Don't normalize for plotting physical space
        norm_factors_dict = norm_factors_dict,
    )
    obs_data[var_name] = obs_var
end

modis_data = NCDataset(joinpath(@clima_artifact("modis_lwp_iwp"), "modis_lwp_iwp.nc"))
nanmean(modis_data["lwp"][:, :, :], dims = 1)

# Define the clipping range
z_min, z_max = 0, .3

# Get the data and clip it to the desired range
lwp_data = nanmean(modis_data["lwp"][:, :, :], dims = 1)[1, :, :] 
iwp_data = nanmean(modis_data["iwp"][:, :, :], dims = 1)[1, :, :] 
# lwp_data = modis_data["lwp"][1, :, :] 
iwp_data = modis_data["iwp"][1, :, :] 

p = Plots.heatmap(
    modis_data["longitude"][:],
    modis_data["latitude"][:],
    lwp_data,
    xlabel = "Longitude",
    ylabel = "Latitude",
    title = "LWP (kg/m^2)",
    clims = (z_min, z_max),  # Use clims instead of zlims for colorbar limits
)

p = Plots.heatmap(
    modis_data["longitude"][:],
    modis_data["latitude"][:],
    iwp_data,
    xlabel = "Longitude",
    ylabel = "Latitude",
    title = "IWP (kg/m^2)",
    clims = (z_min, z_max),  # Use clims instead of zlims for colorbar limits
)

Plots.savefig("lwp_comparison.png")
