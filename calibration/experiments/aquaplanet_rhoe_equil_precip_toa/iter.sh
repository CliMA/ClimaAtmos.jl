
#!/bin/bash
#PBS -j oe
#PBS -A UCIT0011
#PBS -o aquaplanet_rhoe_equil_precip_toa_calibration.txt
#PBS -q develop
#PBS -l walltime=00:05:00
#PBS -l select=1:ncpus=1:ngpus=0:mem=10G
#PBS -v CALIBRATION_EXPERIMENT_DIR,CALIBRATION_ITERATION

export CALIBRATION_EXPERIMENT_DIR="calibration/experiments/aquaplanet_rhoe_equil_precip_toa"
export CALIBRATION_ITERATION=3
echo $CALIBRATION_EXPERIMENT_DIR
julia --project=$CALIBRATION_EXPERIMENT_DIR -e '
import ClimaCalibrate as CAL

experiment_dir = dirname(Base.active_project())
include(joinpath(experiment_dir, "observation_map.jl"))
(; ensemble_size, output_dir, prior) = CAL.ExperimentConfig(experiment_dir)
iteration = CAL.env_iteration()

if iteration > 0
    G_ensemble = CAL.observation_map(iteration - 1)
    CAL.save_G_ensemble(output_dir, iteration - 1, G_ensemble)
    CAL.update_ensemble(output_dir, iteration - 1, prior)
end

b = CAL.get_backend()
model_interface = "calibration/model_interface.jl"
module_load_str = CAL.module_load_string(b)
hpc_kwargs = CAL.kwargs(time = 60, ntasks = 1, gpus_per_task = 1, 
                    cpus_per_task=0, queue = "main")
                    
jobids = map(1:ensemble_size) do member
    @info "Running ensemble member $member"
    CAL.model_run(
    b,
    iteration,
    member,
    output_dir,
    experiment_dir,
    model_interface,
    module_load_str;
    hpc_kwargs,
)
end
'