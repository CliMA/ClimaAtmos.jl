using CUDA
using AbbreviatedStackTraces  # shortens type information upon crashing -- only useful in interactive mode
# get access to GPU (on clima cluster) by: srun -G1 --pty bash -l
# check if you're on an exclusive node by: echo $CUDA_VISIBLE_DEVICES
# should return a number (GPU id), or nothing (on a shared node)

# MPI run:
# mpiexec -n 2 julia -E "include("examples/hybrid/simpledriver.jl")"
# or:
# ssh -X clima
# srun -G2 -n2 --pty bash -l
# mpirun -xterm 0,1 -np 2 julia --project=examples
# include("examples/hybrid/simpledriver.jl")  # from ~/ClimaAtmos.jl

# Single Julia process: no MPI
cuda_devices = get(ENV, "CUDA_VISIBLE_DEVICES", "")
if isempty(cuda_devices) || length(split(cuda_devices, ',')) == 1
    ENV["CLIMACOMMS_CONTEXT"] = "SINGLETON"
    @info "Running without MPI, CUDA_VISIBLE_DEVICES: $(cuda_devices)"
else
    @info "Running with MPI, CUDA_VISIBLE_DEVICES: $(cuda_devices)"
    import MPI
    ENV["CLIMACOMMS_CONTEXT"] = "MPI"
end


import ClimaAtmos as CA
import ClimaCore: Fields
import YAML

import Random
# Random.seed!(Random.MersenneTwister())
Random.seed!(1234)

# --> For GCM-driven, edit external_forcing file to one of:
# external_forcing_file: "/groups/esm/zhaoyi/GCMForcedLES/cfsite/07/HadGEM2-A/amip/Output.cfsite23_HadGEM2-A_amip_2004-2008.07.4x/stats/Stats.cfsite23_HadGEM2-A_amip_2004-2008.07.nc"
# external_forcing_file: "/Users/haakonervik/Documents/CliMA/Stats.cfsite23_HadGEM2-A_amip_2004-2008.07.nc"
# external_forcing_file: "/Users/haakon/Documents/CliMA/Stats.cfsite23_HadGEM2-A_amip_2004-2008.07.nc"
# <-- 

# --> get config
configs_path = joinpath(pkgdir(CA), "config/model_configs/")
# pth = joinpath(configs_path, "prognostic_edmfx_gcmdriven_column.yml")  # LES-driven
pth = joinpath(configs_path, "les_isdac_box_noneq.yml"); job_id = "les_isdac_box"  # LES isdac
config_dict = YAML.load_file(pth)
@assert !haskey(config_dict, "restart_file")  # ensure no restart file is specified
# <--

### >> generate `day0.0.hdf5`
ENV["CLIMACOMMS_DEVICE"] = "CPU"
config_dict_init = copy(config_dict)
config_dict_init["output_default_diagnostics"] = false
pop!(config_dict_init, "diagnostics")
config_dict_init["output_dir"] = mktempdir()
config_init = CA.AtmosConfig(config_dict_init; job_id)
simulation_init = CA.get_simulation(config_init)
CA.save_state_to_disk_func(simulation_init.integrator, simulation_init.output_dir)
# Set `restart_file` to `day0.0.hdf5`
config_dict["restart_file"] = joinpath(simulation_init.output_dir, "day0.0.hdf5")
### <<

### >> Restart from actual previous simulation run:
# restart_file = "output/les_isdac_box/output_0023/day0.37306.hdf5"
# config_dict["restart_file"] = restart_file
### <<

### >> Run actual simulation
ENV["CLIMACOMMS_DEVICE"] = "CUDA"
config = CA.AtmosConfig(config_dict; job_id)
simulation = CA.get_simulation(config);

sol_res = CA.solve_atmos!(simulation);  # solve!

(; integrator) = simulation;
(; p) = integrator;
(; atmos, params) = p;

# --> OPTIONAL
sol_res.ret_code == :simulation_crashed && @warn("The ClimaAtmos simulation has crashed. See the stack trace for details.")
# (; sol) = sol_res
# !isempty(integrator.tstops) && (@assert last(sol.t) == simulation.t_end)
# CA.verify_callbacks(sol.t)
# <--

# --> Make ci plots
# ]add ClimaAnalysis, ClimaCoreSpectra
include(joinpath(pkgdir(CA), "post_processing", "ci_plots.jl"))
ref_job_id = config.parsed_args["reference_job_id"]
reference_job_id = isnothing(ref_job_id) ? simulation.job_id : ref_job_id
make_plots(Val(Symbol(reference_job_id)), simulation.output_dir)
# <--

# --> ClimaAnalysis
import ClimaAnalysis
# using ClimaAnalysis.Visualize
import ClimaAnalysis.Visualize as viz
using ClimaAnalysis.Utils: kwargs
using CairoMakie; CairoMakie.activate!()
# using GLMakie; GLMakie.activate!()
simdir = ClimaAnalysis.SimDir(simulation.output_dir);

# entr = get(simdir; short_name = "entr")
# entr.dims  # (time, x, y, z)

# fig = Figure();
# viz.plot!(fig, entr, time=0, x=0, y=0, more_kwargs = Dict(:axis => kwargs(dim_on_y = true)))
# viz.plot!(fig, entr, x=0, y=0);
# fig
# <--
