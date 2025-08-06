using ClimaAnalysis
import ClimaAtmos as CA
import YAML
import ClimaCalibrate as CAL
using Plots
using Dates

include("helper_funcs.jl")
include("observation_map.jl")

# site metadata
iteration = "iteration_009"
config = "config_64"

experiment_config_path = "experiment_config.yml"

# load experiment config
experiment_config = YAML.load_file(experiment_config_path)

# unpack all experiment_config vars into scope
for (key, value) in experiment_config
    @eval $(Symbol(key)) = $value
end

# load case data - variables now directly available from config unpacking
model_config_dict = YAML.load_file(model_config)
atmos_config = CA.AtmosConfig(model_config_dict)

zc_model = get_z_grid(atmos_config; z_max)

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

# Get simulation data for all ensemble members
all_sim_data = Dict(var_name => [] for var_name in y_var_names)

@info "Loading simulation data for $ensemble_size ensemble members..."

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
                z_max = z_max,
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

@info "Successfully loaded data for $(length(all_sim_data["ta"])) ensemble members"

# Get observation data
forcing_file_path = CA.get_external_monthly_forcing_file_path(
    Dict(
        "start_date" => start_date,
        "site_latitude" => lat,
        "site_longitude" => lon,
    ),
)

# Convert times to DateTime for observations
obs_start = Dates.DateTime(start_date, "yyyymmdd") 
obs_end = Dates.DateTime(start_date, "yyyymmdd") + Dates.Day(1)

# Get observation data for each variable separately
obs_data = Dict()
for var_name in y_var_names
    obs_var = get_obs(
        forcing_file_path,
        [var_name],
        obs_start,
        obs_end;
        normalize = false,  # Don't normalize for plotting physical space
        norm_factors_dict = norm_factors_by_var,
        z_scm = zc_model,
        log_vars = [""],
    )
    obs_data[var_name] = obs_var
end

# Create plots for each variable
plots_list = []

# Variable names and units for plotting
var_info = Dict(
    "ta" => ("Temperature", "K"),
    "hus" => ("Specific Humidity", "kg/kg"),
    "clw" => ("Cloud Liquid Water", "kg/kg")
)

for var_name in y_var_names
    p = Plots.plot(
        xlabel = "$(var_info[var_name][1]) ($(var_info[var_name][2]))",
        ylabel = "Height (m)",
        title = var_info[var_name][1],
        legend = :topright,
        margin = 5Plots.mm  # Add margin to prevent cropping
    )
    
    # Plot all ensemble members in light blue/gray
    for (i, member_data) in enumerate(all_sim_data[var_name])
        Plots.plot!(p, member_data, zc_model,
            label = i == 1 ? "Ensemble Members" : "",
            color = :lightblue,
            alpha = 0.3,
            linewidth = 1
        )
    end
    
    # Plot observations on top in red
    Plots.plot!(p, obs_data[var_name], zc_model,
        label = "Observation",
        color = :red,
        linewidth = 3,
        linestyle = :solid
    )
    
    push!(plots_list, p)
end

# Determine season and create meta title
season = get_season(start_date)
meta_title = "Site: $(lat)°N, $(lon)°E | Season: $season | Config: $config | Iteration: $iteration"

# Create combined plot with proper spacing and meta title
combined_plot = plot(
    plots_list..., 
    layout = (1, 3), 
    size = (1400, 500),  # Increased size to prevent cropping
    plot_title = meta_title,
    plot_titlefontsize = 14,
    left_margin = 10Plots.mm,
    right_margin = 10Plots.mm,
    top_margin = 15Plots.mm,
    bottom_margin = 10Plots.mm
)

# Save the plot
plot_dir = joinpath(output_dir, "plots", "profile_comparison")
if !isdir(plot_dir)
    mkpath(plot_dir)
end

# Create filename with more descriptive information
filename = "ensemble_profiles_$(iteration)_$(config)_lat$(lat)_lon$(lon)_$(season).png"
savefig(combined_plot, joinpath(plot_dir, filename))

# Display the plot
display(combined_plot)

@info "Ensemble profile comparison plot saved to: $(joinpath(plot_dir, filename))"
@info "Plotted $(length(all_sim_data["ta"])) ensemble members vs observations"
