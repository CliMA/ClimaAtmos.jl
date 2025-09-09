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

#include("helper_funcs.jl")

using Distributed
# const experiment_config_dict =
#     YAML.load_file(joinpath(@__DIR__, "experiment_config.yml"))
# const output_dir = experiment_config_dict["output_dir"]
# const model_config = experiment_config_dict["model_config"]
# const batch_size = experiment_config_dict["batch_size"]


# MY STUFF


# FUNCTIONS
default_worker_pool() = WorkerPool(workers())

@everywhere function run_atmos_simulation(atmos_config)
    simulation = CA.get_simulation(atmos_config)
    sol_res = CA.solve_atmos!(simulation)
    if sol_res.ret_code == :simulation_crashed
        if !isnothing(sol_res.sol)
            T = eltype(sol_res.sol)
            if T !== Any && isconcretetype(T)
                sol_res.sol .= T(NaN)
            else
                fill!(sol_res.sol, NaN)
            end
        end
        error(
            "The ClimaAtmos simulation has crashed. See the stack trace for details.",
        )
    end
end

function forward_model(parameter_path, lat, lon, start_date)
    base_config_dict = YAML.load_file(joinpath(@__DIR__, "diagnostic_edmfx_diurnal_scm_imp.yml"))
    config_dict = deepcopy(base_config_dict)

    # update the config_dict with site latitude / longitude
    config_dict["site_latitude"] = lat
    config_dict["site_longitude"] = lon
    config_dict["start_date"] = start_date

    # set the data output directory
    member_path = dirname(parameter_path)
    member_path = joinpath(member_path, "location_$(lat)_$(lon)_$(start_date)")
    config_dict["output_dir"] = member_path

    # add the perturbation toml to the config_dict
    # if haskey(config_dict, "toml")
    #     config_dict["toml"] = abspath.(config_dict["toml"])
    #     push!(config_dict["toml"], parameter_path)
    # else
    #     config_dict["toml"] = [parameter_path]
    # end

    push!(config_dict["toml"], truth_toml)
    
    comms_ctx = ClimaComms.SingletonCommsContext()
    config = CA.AtmosConfig(config_dict; comms_ctx)

    start_time = time()
    try
        run_atmos_simulation(config)
    catch e
        @warn "Simulation crashed for parameter file $(parameter_path): $(e)"
        return
    end
    end_time = time()

    elapsed_time = (end_time - start_time) / 60.0

    @info "Finished simulation. Total time taken: $(elapsed_time) minutes."

end

#################### Site Selection ###########################
# lats = [
#     -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -18.5, -17.0,
#     -15.5, -14.0, -12.5, -11.0, -9.5, -8.0, 35.0, 32.0,
#     29.0, 23.0, 20.0, 17.0
# ]

# lons = [
#     -72.5, -75.0, -77.5, -80.0, -82.5, -85.0, -90.0, -95.0,
#     -100.0, -105.0, -110.0, -115.0, -120.0, -125.1000061,
#     -125.0, -129.0, -133.0, -141.0, -145.0, -149.0
# ]

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

start_dates = ["20071001"] # , 

#################### Site Selection ###########################

# num_procs = 800
# ensemble_size = 100
fast_timescale = false

# experiment_dir = dirname(Base.active_project())
# experiment_dir = "/central/groups/esm/jschmitt/experiments/AtmosLossDesign/ensemble_parameter_perturbations"
# experiment_config = YAML.load_file(joinpath(experiment_dir, "experiment_config.yml"))
# output_dir = experiment_config["output_dir"]
# toml_path_name = experiment_config["toml_path_name"]
# prior = CAL.get_prior(joinpath(experiment_dir, experiment_config["prior_path"]))

output_dir = "/home/oalcabes/SO_singlecolumn"

if fast_timescale
    toml_path_name = "toml/truth_fast.toml"
else
    toml_path_name = "toml/truth.toml"
end

worker_pool = default_worker_pool()

# add workers
@info "Starting $ensemble_size workers."
addprocs(
    CAL.SlurmManager(Int(ensemble_size)),
    t = "6:00:00",
    mem_per_cpu = "25G",
    cpus_per_task = 1,
)

@everywhere begin
    using ClimaCalibrate
    import ClimaCalibrate as CAL
    import ClimaAtmos as CA
    import JLD2
    import YAML
    using LinearAlgebra
    using Distributions
    using Distributed

    include("observation_map.jl")
    include("model_interface.jl")

    ensemble_size = 20
    n_iterations = 10
    output_dir = "/home/oalcabes/EKI_output/test_8"

    experiment_dir = dirname(Base.active_project())
end

@sync begin 
    for start_date in start_dates
        for (site_index, (lat, lon)) in enumerate(zip(lats, lons))
            @async begin
                worker = take!(worker_pool)
                try
                    @show worker site_index
                    # Pass lat and lon into forward_model
                    remotecall_wait(forward_model, output_dir, worker, lat, lon, start_date)
                catch e
                    @warn "Error in worker $(worker) at site $(site_index) for start date $(start_date): $(e)"
                finally
                    put!(worker_pool, worker)
                end
            end
        end
    end
end

# function run_iteration(
#     parameter_paths,
#     ensemble_size,
#     lats, 
#     lons,
#     start_dates,    
#     relaxation_tomls; 
#     worker_pool = default_worker_pool(),
# )
#     @sync begin 
#         for start_date in start_dates
#             for (site_index, (lat, lon, relaxation_toml)) in enumerate(zip(lats, lons, relaxation_tomls))
#                 for m in 1:ensemble_size
#                     @async begin
#                         worker = take!(worker_pool)
#                         try
#                             @show worker site_index m
#                             # Pass lat and lon into forward_model
#                             remotecall_wait(forward_model, worker, parameter_paths[m], lat, lon, start_date, relaxation_toml)
#                         catch e
#                             @warn "Error in worker $(worker) at site $(site_index) for start date $(start_date): $(e)"
#                         finally
#                             put!(worker_pool, worker)
#                         end
#                     end
#                 end
#             end
#         end
#     end
# end

# STUFF FROM JULIAN

# @everywhere function run_atmos_simulation(atmos_config)
#     simulation = CA.get_simulation(atmos_config)
#     sol_res = CA.solve_atmos!(simulation)
#     if sol_res.ret_code == :simulation_crashed
#         if !isnothing(sol_res.sol)
#             T = eltype(sol_res.sol)
#             if T !== Any && isconcretetype(T)
#                 sol_res.sol .= T(NaN)
#             else
#                 fill!(sol_res.sol, NaN)
#             end
#         end
#         error(
#             "The ClimaAtmos simulation has crashed. See the stack trace for details.",
#         )
#     end
# end

# function forward_model(parameter_path, lat, lon, start_date, relaxation_toml)
#     base_config_dict = YAML.load_file(joinpath(@__DIR__, "diagnostic_edmfx_diurnal_scm_imp.yml"))
#     config_dict = deepcopy(base_config_dict)

#     # update the config_dict with site latitude / longitude
#     config_dict["site_latitude"] = lat
#     config_dict["site_longitude"] = lon
#     config_dict["start_date"] = start_date

#     # set the data output directory
#     member_path = dirname(parameter_path)
#     member_path = joinpath(member_path, "location_$(lat)_$(lon)_$(start_date)")
#     config_dict["output_dir"] = member_path

#     # add the perturbation toml to the config_dict
#     if haskey(config_dict, "toml")
#         config_dict["toml"] = abspath.(config_dict["toml"])
#         push!(config_dict["toml"], parameter_path)
#     else
#         config_dict["toml"] = [parameter_path]
#     end

#     # depending on the simulation type, we need to change the relaxation coefficients
#     push!(config_dict["toml"], relaxation_toml)
    
#     comms_ctx = ClimaComms.SingletonCommsContext()
#     config = CA.AtmosConfig(config_dict; comms_ctx)

#     start_time = time()
#     try
#         run_atmos_simulation(config)
#     catch e
#         @warn "Simulation crashed for parameter file $(parameter_path): $(e)"
#         return
#     end
#     end_time = time()

#     elapsed_time = (end_time - start_time) / 60.0

#     @info "Finished simulation. Total time taken: $(elapsed_time) minutes."

# end


# function run_iteration(
#     parameter_paths,
#     ensemble_size,
#     lats, 
#     lons,
#     start_dates,    
#     relaxation_tomls; 
#     worker_pool = default_worker_pool(),
# )
#     @sync begin 
#         for start_date in start_dates
#             for (site_index, (lat, lon, relaxation_toml)) in enumerate(zip(lats, lons, relaxation_tomls))
#                 for m in 1:ensemble_size
#                     @async begin
#                         worker = take!(worker_pool)
#                         try
#                             @show worker site_index m
#                             # Pass lat and lon into forward_model
#                             remotecall_wait(forward_model, worker, parameter_paths[m], lat, lon, start_date, relaxation_toml)
#                         catch e
#                             @warn "Error in worker $(worker) at site $(site_index) for start date $(start_date): $(e)"
#                         finally
#                             put!(worker_pool, worker)
#                         end
#                     end
#                 end
#             end
#         end
#     end
# end

# using TOML 
# using Distributed
# import EnsembleKalmanProcesses as EKP
# import ClimaCalibrate as CAL
# using Glob
# using NCDatasets
# import YAML

# num_procs = 800
# ensemble_size = 100

# # experiment_dir = dirname(Base.active_project())
# experiment_dir = "/central/groups/esm/jschmitt/experiments/AtmosLossDesign/ensemble_parameter_perturbations"
# experiment_config = YAML.load_file(joinpath(experiment_dir, "experiment_config.yml"))
# output_dir = experiment_config["output_dir"]
# toml_path_name = experiment_config["toml_path_name"]
# prior = CAL.get_prior(joinpath(experiment_dir, experiment_config["prior_path"]))

# #################### Site Selection ###########################
# # lats = [
# #     -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -18.5, -17.0,
# #     -15.5, -14.0, -12.5, -11.0, -9.5, -8.0, 35.0, 32.0,
# #     29.0, 23.0, 20.0, 17.0
# # ]

# # lons = [
# #     -72.5, -75.0, -77.5, -80.0, -82.5, -85.0, -90.0, -95.0,
# #     -100.0, -105.0, -110.0, -115.0, -120.0, -125.1000061,
# #     -125.0, -129.0, -133.0, -141.0, -145.0, -149.0
# # ]

# ds = NCDataset("../coszen_data.nc")

# deep_sites = (collect(30:33)..., collect(66:70)..., 82, 92, 94, 96, 99, 100)
# shallow_sites = setdiff(collect(1:119), deep_sites)

# lats, lons, relaxation_tomls = [], [], []
# for site in 1:119
#     push!(lats, ds["lat"][site])
#     push!(lons, (ds["lon"][site] + 180.0) % 360.0 - 180.0)
#     if site in deep_sites
#         push!(relaxation_tomls, experiment_config["forcing_toml_files"]["deep"])
#     else
#         push!(relaxation_tomls, experiment_config["forcing_toml_files"]["shallow"])
#     end
# end

# start_dates = ["20071001"] # , 
#################### Site Selection ###########################


# sample from the parameter distribution and write the elements to the toml files. 
# param_array = EKP.construct_initial_ensemble(prior, ensemble_size)
#EKP.TOMLInterface.save_parameter_ensemble(param_array, prior, CAL.get_param_dict(prior), output_dir, toml_path_name)
# toml_list = glob(joinpath(output_dir, "member_*", toml_path_name * ".toml"))

# add processes 
# addprocs(
#     CAL.SlurmManager(num_procs),
#     t = "12:00:00",
#     mem_per_cpu = "25G",
#     cpus_per_task = "1",
# )


# # Distribute required code and packages
# @everywhere using TOML
# @everywhere include("model_interface.jl")

# # Send toml_list to each worker
# for p in workers()
#     @eval @spawnat $p global toml_list = $(toml_list)
# end

# println("Going to run $(length(toml_list) * length(lats) * length(start_dates)) simulations...")

# default_worker_pool() = WorkerPool(workers())

# run_iteration(toml_list, length(toml_list), lats, lons, start_dates, relaxation_tomls)

# #!/bin/bash
# #SBATCH --ntasks=800
# #SBATCH --time=12:00:00
# #SBATCH --mem-per-cpu=25G
# #SBATCH --cpus-per-task=1  # CPUs per task
# #SBATCH -o workers.txt
# #SBATCH -e workers_error.txt
# #SBATCH --output=scm_ensemble1.out
# #SBATCH --mail-user=jschmitt@caltech.edu
# #SBATCH --mail-type=ALL

# #mv julia*.out juliaout/
# module purge
# module load climacommon

# julia --project=. run_ensemble.jl 2>&1 | tee -a workers.txt

# Generates forcing files in parallel using Distributed.jl

# using Distributed

# Add worker processes (will be managed by SLURM)
# if nprocs() == 1
#     # Add workers based on SLURM_NTASKS or default to 40
#     n_workers = haskey(ENV, "SLURM_NTASKS") ? parse(Int, ENV["SLURM_NTASKS"]) - 1 : 60
#     addprocs(n_workers)
# end

# @everywhere begin
#     import ClimaAtmos as CA
#     using ClimaUtilities.ClimaArtifacts
#     using NCDatasets
# end

# # Read sites from NetCDF file
# ds = NCDataset("../coszen_data.nc")

# lats, lons = [], []
# for site in 1:119
#     push!(lats, ds["lat"][site])
#     push!(lons, (ds["lon"][site] + 180.0) % 360.0 - 180.0)
# end

# start_dates = ["20070401", "20070701", "20071001"]

# Create all combinations of work to be done
# work_items = []
# for start_date in start_dates
#     for i in 1:lastindex(lats)
#         push!(work_items, (start_date, lats[i], lons[i], i))
#     end
# end

# println("Total work items: $(length(work_items))")
# println("Number of workers: $(nworkers())")

# # Function to process a single work item
# @everywhere function process_forcing_file(work_item)
#     start_date, lat, lon, site_index = work_item
    
#     try
#         single_parsed_args = Dict(
#             "start_date" => start_date,
#             "site_latitude" => lat,
#             "site_longitude" => lon,
#         )
        
#         # Get the forcing file path 
#         forcing_file_path = CA.get_external_monthly_forcing_file_path(single_parsed_args)
        
#         # Log progress
#         worker_id = myid()
#         println("Worker $worker_id processing site $site_index (lat=$lat, lon=$lon, date=$start_date)")
        
#         # Check if file exists and passes time check before generating
#         if !isfile(forcing_file_path) || !CA.check_monthly_forcing_times(forcing_file_path, single_parsed_args)
#             CA.generate_external_forcing_file(single_parsed_args, forcing_file_path, Float32; 
#                 input_data_dir = joinpath(@clima_artifact("era5_hourly_atmos_raw"), "monthly"),
#                 data_strs = [
#                     "monthly_diurnal_profiles",
#                     "monthly_diurnal_inst",
#                     "monthly_diurnal_accum",
#                 ])
#             println("Worker $worker_id completed site $site_index")
#         else
#             println("Worker $worker_id: forcing file already exists for site $site_index")
#         end
        
#         return (true, site_index, start_date, nothing)
        
#     catch e
#         error_msg = "Error generating forcing file for site $site_index (lat=$lat, lon=$lon, date=$start_date): $e"
#         println("Worker $(myid()): $error_msg")
#         return (false, site_index, start_date, error_msg)
#     end
# end

# # Process all work items in parallel
# println("Starting parallel processing...")
# results = pmap(process_forcing_file, work_items)

# # Summarize results
# successful = sum(r[1] for r in results)
# failed = length(results) - successful

# println("\n" * "="^50)
# println("SUMMARY:")
# println("Total jobs: $(length(results))")
# println("Successful: $successful")
# println("Failed: $failed")

# if failed > 0
#     println("\nFailed jobs:")
#     for result in results
#         if !result[1]  # if not successful
#             println("  Site $(result[2]), Date $(result[3]): $(result[4])")
#         end
#     end
# end

# println("Done generating forcing files")
