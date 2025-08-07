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

# Site selection - use iteration 001 as specified
iteration = "iteration_004"

# load experiment config
experiment_config = YAML.load_file("experiment_config.yml")

# unpack all experiment_config vars into scope
for (key, value) in experiment_config
    @eval $(Symbol(key)) = $value
end

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

# Function to get MODIS observations for a given site
function get_modis_obs(
    var_name,
    lat, 
    lon,
    start_date;
    time_window_days = 30  # Use monthly average
)
    # Load MODIS data from artifact
    modis_data = NCDataset(joinpath(@clima_artifact("modis_lwp_iwp"), "modis_lwp_iwp.nc"))
    
    @assert lat >= -90 && lat <= 90 "Latitude out of bounds: $lat"
    @assert lon >= -180 && lon <= 180 "Please convert longitude to -180 to 180 / out of bounds: $lon"
    
    # Get the index of the closest lat and lon
    lat_idx = findmin(abs.(modis_data["latitude"][:] .- lat))[2]
    lon_idx = findmin(abs.(modis_data["longitude"][:] .- lon))[2]
    
    # Create observation time window around the start date
    obs_start = Dates.DateTime(start_date, "yyyymmdd")
    obs_end = obs_start + Dates.Day(time_window_days)
    
    # Find all time indices within the observation window
    #time_idx = findall(modis_data["time"][:] .>= obs_start .&& modis_data["time"][:] .< obs_end)
    time_idx = findall(Dates.month.(modis_data["time"][:]) .==Dates.month(obs_start)) # select all data for the month
    
    if length(time_idx) == 0
        @warn "No MODIS data found for site ($(lat), $(lon)) in time window"
        close(modis_data)
        return NaN
    end
    
    # Get the data and compute time mean
    obs_data = modis_data[var_name][:, lat_idx, lon_idx]
    @info "obs_data size: $(size(obs_data))"
    obs_mean = nanmean(obs_data)
    
    close(modis_data)
    return obs_mean
end

# Function to get simulation data for a given config/site
function get_sim_data(config, var_name)
    all_member_data = []
    
    for m in 1:ensemble_size
        member_name = "member_$(lpad(m, 3, '0'))"
        member_path = joinpath(output_dir, iteration, member_name)
        
        # Check if this member/config combination exists
        sim_dir = joinpath(member_path, config, "output_0000")
        if !isdir(sim_dir)
            continue
        end
        
        try
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
            # Convert to scalar if it's a 0-dimensional array
            scalar_data = var_data isa AbstractArray && ndims(var_data) == 0 ? var_data[] : var_data
            push!(all_member_data, scalar_data)
        catch e
            @warn "Failed to load data for member $m, config $config: $e"
        end
    end
    
    return all_member_data
end

# Get available configs and their site information
function get_site_info(config)
    member = "member_001"
    sim_path = joinpath(output_dir, iteration, member, config, "output_active", ".yml")
    
    if !isfile(sim_path)
        return nothing
    end
    
    try
        sim_dict = YAML.load_file(sim_path)
        return Dict(
            "lat" => sim_dict["site_latitude"],
            "lon" => sim_dict["site_longitude"],
            "start_date" => sim_dict["start_date"],
            "config" => config
        )
    catch e
        @warn "Failed to read site info for config $config: $e"
        return nothing
    end
end

# Discover all available configs
@info "Discovering available configs in $iteration..."
available_configs = []
config_dirs = readdir(joinpath(output_dir, iteration, "member_001"))
for dir in config_dirs
    if startswith(dir, "config_")
        site_info = get_site_info(dir)
        if site_info !== nothing
            push!(available_configs, site_info)
            @info "Found config $(dir): Lat=$(site_info["lat"]), Lon=$(site_info["lon"]), Season=$(get_season(site_info["start_date"]))"
        end
    end
end

@info "Found $(length(available_configs)) available configs"

# Collect data for all sites and variables
site_data = Dict()
variables_to_plot = ["lwp", "clivi"]  # LWP and IWP (clivi represents IWP)

for var_name in variables_to_plot
    site_data[var_name] = Dict(
        "sites" => [],  # Site information for each available site
        "sim_data_by_site" => [],  # Ensemble data for each site
        "obs_data_by_site" => []   # MODIS observation for each site
    )
    
    @info "Processing variable: $var_name"
    
    for (site_idx, site_config) in enumerate(available_configs)
        config = site_config["config"]
        lat = site_config["lat"]
        lon = site_config["lon"]
        start_date = site_config["start_date"]
        season = get_season(start_date)
        
        # Get simulation data for all ensemble members
        sim_data = get_sim_data(config, var_name)
        if length(sim_data) > 0
            # Get MODIS observation - map clivi to iwp
            modis_var_name = var_name == "clivi" ? "iwp" : var_name
            obs_value = get_modis_obs(modis_var_name, lat, lon, start_date)
            
            if !isnan(obs_value)
                site_label = "$(lat)°N,$(lon)°E\n$season"
                push!(site_data[var_name]["sites"], site_label)
                push!(site_data[var_name]["sim_data_by_site"], sim_data)
                push!(site_data[var_name]["obs_data_by_site"], obs_value)
                
                @info "Site ($(lat), $(lon)) - $season: $(length(sim_data)) ensemble members, Obs=$obs_value"
            end
        end
    end
end

# Create comparison plots
plots_list = []

# Variable names and units for plotting
var_info = Dict(
    "lwp" => ("Liquid Water Path", "kg/m²"),
    "clivi" => ("Ice Water Path", "kg/m²")
)

for var_name in variables_to_plot
    sites = site_data[var_name]["sites"]
    sim_data_by_site = site_data[var_name]["sim_data_by_site"]
    obs_data_by_site = site_data[var_name]["obs_data_by_site"]
    
    if length(sites) == 0
        @warn "No data available for $var_name"
        continue
    end
    
    # Create the plot
    p = Plots.plot(
        xlabel = "Sites",
        ylabel = "$(var_info[var_name][1]) ($(var_info[var_name][2]))",
        title = var_info[var_name][1],
        legend = :topright,
        xticks = (1:length(sites), sites),
        xrotation = 45,
        margin = 15Plots.mm
    )
    
    # Plot ensemble members for each site
    for (site_idx, sim_data) in enumerate(sim_data_by_site)
        # Add some random jitter to x-position for better visibility
        x_positions = site_idx .+ 0.3 * (rand(length(sim_data)) .- 0.5)
        
        Plots.scatter!(p, x_positions, sim_data,
            alpha = 0.6,
            markersize = 4,
            color = :lightblue,
            markerstrokecolor = :blue,
            markerstrokewidth = 0.5,
            label = site_idx == 1 ? "Ensemble Members" : ""
        )
    end
    
    # Plot MODIS observations on top
    Plots.scatter!(p, 1:length(sites), obs_data_by_site,
        markersize = 8,
        color = :red,
        markerstrokecolor = :darkred,
        markerstrokewidth = 2,
        markershape = :diamond,
        label = "MODIS Observations"
    )
    
    push!(plots_list, p)
end

# Create combined plot
if length(plots_list) > 0
    combined_plot = plot(
        plots_list..., 
        layout = (1, length(plots_list)), 
        size = (600 * length(plots_list), 700),  # Increased height for rotated labels
        plot_title = "Ensemble vs MODIS Observations by Site | Iteration: $iteration",
        plot_titlefontsize = 14,
        left_margin = 15Plots.mm,
        right_margin = 15Plots.mm,
        top_margin = 20Plots.mm,
        bottom_margin = 20Plots.mm
    )
    
    # Save the plot
    plot_dir = joinpath(output_dir, "plots", "obs_comparison")
    if !isdir(plot_dir)
        mkpath(plot_dir)
    end
    
    # Create filename
    filename = "ensemble_vs_modis_month_avg_$(iteration)_$(length(available_configs))sites.png"
    savefig(combined_plot, joinpath(plot_dir, filename))
    
    # Display the plot
    display(combined_plot)
    
    @info "Observation comparison plot saved to: $(joinpath(plot_dir, filename))"
    
    # Calculate and display summary statistics
    total_ensemble_points = 0
    for var_name in variables_to_plot
        sites = site_data[var_name]["sites"]
        sim_data_by_site = site_data[var_name]["sim_data_by_site"]
        if length(sites) > 0
            site_ensemble_counts = [length(sim_data) for sim_data in sim_data_by_site]
            total_for_var = sum(site_ensemble_counts)
            total_ensemble_points += total_for_var
            @info "$var_name: $(length(sites)) sites, $total_for_var total ensemble members"
        end
    end
    @info "Total: $(length(available_configs)) sites, $total_ensemble_points ensemble points plotted"
else
    @warn "No plots could be created - no data available"
end

# Σ_obs = get_modis_Σ_obs(["lwp", "iwp"], 43, -72, "20070701")

########################################################
# Debug below
########################################################

modis_data = NCDataset(joinpath(@clima_artifact("modis_lwp_iwp"), "modis_lwp_iwp.nc"))
obs_start = Dates.DateTime(start_date, "yyyymmdd")
obs_end = obs_start + Dates.Day(30)

# Find all time indices within the observation window
time_indices = findall(modis_data["time"][:] .>= obs_start .&& modis_data["time"][:] .< obs_end)
@info "Found $(length(time_indices)) time indices"

if length(time_indices) > 0
    time_idx = time_indices[1]  # Take the first available time index
    @info "Using time index: $time_idx, corresponding to: $(modis_data["time"][time_idx])"
else
    @warn "No MODIS data found for the specified time period"
    time_idx = 1  # Fallback to first time index
end

# Test the data access
lwp_slice = modis_data["lwp"][time_idx, :, :]
@info "LWP data slice size: $(size(lwp_slice))"

z_min = 0
z_max = 0.3

p = Plots.heatmap(
    modis_data["longitude"][:],
    modis_data["latitude"][:],
    modis_data["lwp"][time_idx, :, :],
    xlabel = "Longitude",
    ylabel = "Latitude",
    title = "LWP (kg/m^2)",
    clims = (z_min, z_max),  # Use clims instead of zlims for colorbar limits
)

p = Plots.heatmap(
    modis_data["longitude"][:],
    modis_data["latitude"][:],
    modis_data["iwp"][time_idx, :, :],
    xlabel = "Longitude",
    ylabel = "Latitude",
    title = "IWP (kg/m^2)",
    clims = (z_min, z_max),  # Use clims instead of zlims for colorbar limits
)

get_modis_obs("lwp", 43, -72, "20070701")



# Plots.savefig("lwp_comparison.png")


test_lats = -90:2.:90
test_lons = -180:2.:180

test_lwp = zeros(length(test_lats), length(test_lons))
test_iwp = zeros(length(test_lats), length(test_lons))

for (lat_idx, lat) in enumerate(test_lats)
    for (lon_idx, lon) in enumerate(test_lons)
        lwp_value = get_modis_obs("lwp", lat, lon, "20070701")
        iwp_value = get_modis_obs("iwp", lat, lon, "20070701")
        test_lwp[lat_idx, lon_idx] = lwp_value
        test_iwp[lat_idx, lon_idx] = iwp_value
    end
end

p = Plots.heatmap(test_lons, test_lats, test_iwp, clims = (0, 0.3))
