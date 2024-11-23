import CUDA
using AbbreviatedStackTraces  # shortens type information upon crashing -- only useful in interactive mode
# get access to GPU (on clima cluster) by: srun -G1 --pty bash -l
# check if you're on an exclusive node by: echo $CUDA_VISIBLE_DEVICES
# should return a number (GPU id), or nothing (on a shared node)

# MPI run:
# (allocate CPU): srun -N1 -n20 --mem=100G -t 05:00:00 --pty bash -l
# mpiexec -n $SLURM_NTASKS julia --project=examples examples/hybrid/simpledriver.jl
# or:
# ssh -X clima
# srun -G2 -n2 --pty bash -l
# mpirun -xterm 0,1 -np 2 julia --project=examples
# include("examples/hybrid/simpledriver.jl")  # from ~/ClimaAtmos.jl

import ClimaComms
# Single Julia process: no MPI
cuda_devices = get(ENV, "CUDA_VISIBLE_DEVICES", "")
device = isempty(cuda_devices) ? ClimaComms.CPUSingleThreaded() : ClimaComms.CUDADevice()

auto_ctx = ClimaComms.context_type() == :MPICommsContext
two_or_more_gpus = length(split(cuda_devices, ',')) ≥ 2
two_or_more_slurm_tasks = parse(Int, get(ENV, "SLURM_NTASKS", "0")) ≥ 2
is_mpi_ctx = auto_ctx && (two_or_more_gpus || two_or_more_slurm_tasks)

main_context = if is_mpi_ctx
    import MPI
    ClimaComms.MPICommsContext(device)
else
    ClimaComms.SingletonCommsContext(device)
end

ClimaComms.init(main_context)
if ClimaComms.iamroot(main_context)
    @info "Running with $(ClimaComms.context_type())"
    @info("`CUDA_VISIBLE_DEVICES` is " * (isempty(cuda_devices) ? "not set" : "set to: $cuda_devices"))
end

import ClimaAtmos as CA
import ClimaCore: Fields
import YAML
import ClimaComms
import SciMLBase

import Random
# Random.seed!(Random.MersenneTwister())
Random.seed!(1234)

# --> get config
configs_path = joinpath(pkgdir(CA), "config/model_configs/")
job_id = "les_isdac_box"
# pth = joinpath(configs_path, "prognostic_edmfx_gcmdriven_column.yml")  # LES-driven
# pth = joinpath(configs_path, "$job_id.yml")
pth = joinpath(configs_path, "les_isdac_box_noneq.yml"); job_id = "les_isdac_box"  # LES isdac
config_dict = YAML.load_file(pth)
delete!(config_dict, "restart_file")  # ensure no restart file is specified
# <--

USE_RESTART = false

if USE_RESTART
    @info("restarting")
    ### >> Restart from actual previous simulation run:
    # restart_file = "output/les_isdac_box/output_0023/day0.37306.hdf5"
    restart_file = "/home/haakon/high-res-isdac-day0.0.hdf5"
    config_dict["restart_file"] = restart_file
    ### <<
else
    ### >> generate `day0.0.hdf5`
    init_output_dir = ""
    root_proc = ClimaComms.iamroot(main_context)
    if root_proc
        config_dict_init = copy(config_dict)
        config_dict_init["output_default_diagnostics"] = false
        pop!(config_dict_init, "diagnostics")
        config_dict_init["output_dir"] = mktempdir()
        
        context = ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded())
        config_init = CA.AtmosConfig(config_dict_init; job_id, comms_ctx = context)
        simulation_init = CA.get_simulation(config_init)
        CA.save_state_to_disk_func(simulation_init.integrator, simulation_init.output_dir)
        init_output_dir = simulation_init.output_dir
    end

    init_output_dir = ClimaComms.bcast(main_context, init_output_dir)  # acts as a barrier
    # Set `restart_file` to `day0.0.hdf5`
    config_dict["restart_file"] = joinpath(init_output_dir, "day0.0.hdf5")
    ### <<
    root_proc && println("Finished generating `day0.0.hdf5`")
end

### >> Run actual simulation
config = CA.AtmosConfig(config_dict; job_id, comms_ctx = main_context)
simulation = CA.get_simulation(config);

sol_res = CA.solve_atmos!(simulation);  # solve!
# debug; take one step only:
# SciMLBase.step!(simulation.integrator)

(; integrator) = simulation;
(; p) = integrator;
(; atmos, params) = p;
