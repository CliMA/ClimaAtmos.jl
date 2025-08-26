# Dependencies
import ClimaAtmos as CA
import ClimaAnalysis
import CairoMakie
import ClimaAnalysis.Visualize as viz
import StatsBase

# This is to help keep track of all the paths, since they are referenced in 
# this file as well as the ci_plots_extension.jl file. I've created separate 
# directories for 1M/2M as well as diagnostic/prognostic. You can follow my 
# naming convention or you can choose your own names.
include("all_paths.jl")

# Step 1: Making the configuration YAML files. Edit the chosen intial 
# conditions in make_yaml.jl if more variations are desired.
include(joinpath(pkgdir(CA),"LWP_N_scripts", "make_yaml.jl"))

default_data_path = default_prog_2M
output_dir = prognostic_2M_config
make_yamls(default_data_path, output_dir, is_prog=true)

# Step 2: Run the simulations. There are two options: parallelized and serial. 
# This script loops through all the files, running each simulation one at a 
# time. The parallelized option is called parallel_driver.jl.
prog_dir = prognostic_2M_config
prog_config_files = readdir(prog_dir)

for config_yaml in prog_config_files
    config = CA.AtmosConfig(joinpath(prog_dir, config_yaml))
    include(joinpath(pkgdir(CA),"LWP_N_scripts", "driver.jl"))
end

# Step 3: plot all the runs. We use process_plot_outputs.jl for this one. There 
# are many options for plotting, but we'll use the most basic. We'll sample 
# only 40 simulation outputs.
include(joinpath(pkgdir(CA),"LWP_N_scripts", "process_plot_outputs.jl"))
output_dir = output_2M
all_outputs = make_edmf_vec(output_dir)
filtered = filter_runs(all_outputs)
sampled = StatsBase.sample(filtered, 40; replace=false)

fig = plot_edmf(sampled, is_1M=false, is_time=false, save=false)
