import EnsembleKalmanProcesses: TOMLInterface
import EnsembleKalmanProcesses as EKP
using EnsembleKalmanProcesses.ParameterDistributions
import ClimaCalibrate: observation_map, ExperimentConfig
using ClimaAnalysis
import ClimaAtmos as CA
import ClimaCalibrate
using Plots
using JLD2
using Statistics
import YAML

include("helper_funcs.jl")


model_config_dict = YAML.load_file("model_config.yml")

member_path_rel = "output/gcm_driven_scm"

atmos_config = CA.AtmosConfig(model_config_dict)
# get/store LES obs and norm factors 
zc_model = get_z_grid(atmos_config)

plot_rel_dir = joinpath(member_path_rel, "plots", "timeseries_plots")
if !isdir(plot_rel_dir)
    mkpath(plot_rel_dir)
end

# hus, thetaa, clw, arup, tke, entr, detr
var_name = "hus"
reduction = "inst"


# zc_model = collect(33.333333:66.666666:4000.0)

iteration = 4
iter_formatted = lpad(iteration, 3, '0')

n_members = 9

var_names = ("hus", "arup", "entr", "detr", "wa")



clims_map = Dict(
    "arup" => (0.0, 0.3),
    "detr" => (0.0, 0.1),
    "entr" => (0.0, 1e-2),
    "hus" => (0.0, 2e-2),
)

for var_name in var_names
    plots = []
    for member in 1:n_members
        # println("Member: $member")
        member_formatted = lpad(member, 3, '0')
        member_path = joinpath(
            member_path_rel,
            "iteration_$(iter_formatted)/member_$(member_formatted)",
        )

        simdir = SimDir(joinpath(member_path, "output_active"))

        var_i = get(simdir; short_name = var_name, reduction = reduction)

        t_array = var_i.dims["time"]
        timeseries_matrix = slice(var_i, x = 1, y = 1).data

        # @show mean(timeseries_matrix)
        show_colorbar = (member == n_members)

        if haskey(clims_map, var_name)
            clims = clims_map[var_name]
        else
            clims = nothing
        end

        # Create a heatmap for each member and add it to the plots array
        p = heatmap(
            t_array,
            zc_model,
            timeseries_matrix',
            color = :darkrainbow,
            aspect_ratio = :auto,
            xlabel = "",
            ylabel = "",
            colorbar = show_colorbar,
            clims = clims,
        )
        push!(plots, p)
    end

    # Combine all plots into a 3x3 grid
    combined_plot = plot(plots..., layout = (3, 3), size = (900, 900))

    # Optionally save the combined plot to a file
    savefig(
        combined_plot,
        joinpath(
            plot_rel_dir,
            "timeseries_$(var_name)_iter_$(iter_formatted).png",
        ),
    )
end
