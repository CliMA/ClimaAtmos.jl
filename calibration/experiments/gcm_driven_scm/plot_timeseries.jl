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


# const z_max = experiment_config_dict["z_max"]
# const output_dir = experiment_config_dict["output_dir"]

# output_dir = "output_prior_and_entr_exp/exp_8"
output_dir = "output_prior_and_entr_exp/exp_8"

model_config_dict =
    YAML.load_file(joinpath(output_dir, "configs", "model_config.yml"))
experiment_config_dict =
    YAML.load_file(joinpath(output_dir, "configs", "experiment_config.yml"))

atmos_config = CA.AtmosConfig(model_config_dict)
# get/store LES obs and norm factors 
# zc_model = get_z_grid(atmos_config; z_max = z_max)

# hus, thetaa, clw, arup, tke, entr, detr

reduction = "inst"

iteration = 0
iter_formatted = lpad(iteration, 3, '0')

config_i = 1

n_members = 49
z_max = 3000.0
ylims = (0.0, z_max)

var_names = ("thetaa", "hus", "arup", "entr", "detr", "waup", "tke")


plot_rel_dir = joinpath(output_dir, "plots", "timeseries", "config_$(config_i)")
if !isdir(plot_rel_dir)
    mkpath(plot_rel_dir)
end


clims_map = Dict(
    "thetaa" => (290.0, 320.0),
    "arup" => (-0.1, 0.1),
    "detr" => (0.0, 0.1),
    "entr" => (0.0, 1e-2),
    "hus" => (0.0, 2e-2),
    "waup" => (-3.0, 3.0),
    "tke" => (-1.0, 1.0),
)
colormap_map = Dict(
    "arup" => :balance,
    "waup" => :balance,
    "detr" => :dense,
    "entr" => :dense,
    "tke" => :balance,
)




for var_name in var_names
    plots = []
    for member in 1:n_members
        # println("Member: $member")

        n_rows = ceil(Int, sqrt(n_members))
        n_cols = ceil(Int, n_members / n_rows)

        row = div(member - 1, n_cols) + 1
        col = mod(member - 1, n_cols) + 1

        member_formatted = lpad(member, 3, '0')
        member_path = joinpath(
            output_dir,
            "iteration_$(iter_formatted)/member_$(member_formatted)/config_$(config_i)",
        )
        # try 

        simdir = SimDir(joinpath(member_path, "output_active"))

        var_i = get(simdir; short_name = var_name, reduction = reduction)
        zc_model = var_i.dims["z"]
        # if !isnothing(z_max)
        #     z_window = filter(x -> x <= z_max, var_i.dims["z"])
        #     var_i = window(var_i, "z", right = maximum(z_window))
        # end

        t_array = var_i.dims["time"]
        timeseries_matrix = slice(var_i, x = 1, y = 1).data

        show_colorbar = (member == n_members)

        if haskey(clims_map, var_name)
            clims = clims_map[var_name]
        else
            clims = nothing
        end

        ## Create a heatmap for each member and add it to the plots array
        # p = heatmap(
        #     t_array,
        #     zc_model,
        #     timeseries_matrix',
        #     colormap = get(colormap_map, var_name, :darkrainbow),
        #     aspect_ratio = :auto,
        #     xlabel = "",
        #     ylabel = "",
        #     colorbar = show_colorbar,
        #     clims = clims,
        # )
        # push!(plots, p)


        ############################################################
        # Inside the for loop where each subplot is created

        yticks = nothing
        xticks = nothing
        if col != 1
            yticks = []
            ylabel = ""
        else
            ylabel = "Height (m)"
        end
        if row != n_rows
            xticks = []
            xlabel = ""
        else
            xlabel = "Time (s)"
        end

        p = heatmap(
            t_array,
            zc_model,
            timeseries_matrix',
            colormap = get(colormap_map, var_name, :darkrainbow),
            aspect_ratio = :auto,
            xlabel = xlabel,
            ylabel = ylabel,
            colorbar = show_colorbar,
            clims = clims,
            yticks = yticks,
            xticks = xticks,
            ylims = ylims,
        )




        # Label each subplot with the member number
        annotate!(p, (0.1, 0.9), text("M $member", :left, 5, "black"))

        push!(plots, p)


        # catch 
        #     @warn "Error during observation map for ensemble member $member"
        # end

    end

    # Combine all plots into a 3x3 grid
    # combined_plot = plot(plots..., layout = (5, 5), size = (900, 900), link = :both)

    # Optionally save the combined plot to a file
    # savefig(
    #     combined_plot,
    #     joinpath(
    #         plot_rel_dir,
    #         "timeseries_$(var_name)_iter_$(iter_formatted).png",
    #     ),
    # )


    ############################################################
    # Calculate the number of rows and columns based on the number of members
    n_rows = ceil(Int, sqrt(n_members))
    n_cols = ceil(Int, n_members / n_rows)

    # Combine all plots into a grid
    combined_plot = plot(
        plots...,
        layout = (n_rows, n_cols),
        size = (900, 900),
        link = :both,
        margin = 0.01Plots.mm,
        outer_margin = 5Plots.mm,
    )

    # Optionally save the combined plot to a file
    savefig(
        combined_plot,
        joinpath(
            plot_rel_dir,
            "timeseries_$(var_name)_iter_$(iter_formatted).png",
        ),
    )


end
