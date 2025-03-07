#!/bin/bash

#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=6:00:00   # walltime
#SBATCH --mem=32G   # memory
#SBATCH --cpus-per-task=4
#SBATCH -J "aq_run"   # job name
#SBATCH --output=aq_run_%j.out


CA_PATH='/groups/esm/cchristo/clima/copies/ClimaAtmos.jl/'                    
CA_ENV=$CA_PATH'.buildkite/'                    
DRIVER=$CA_PATH'examples/hybrid/driver.jl'                          
CONFIG_FILE=$CA_PATH'config/longrun_configs/amip_target_diagedmf.yml'       
export MODULEPATH="/groups/esm/modules:$MODULEPATH"    

module purge                                   
module load climacommon       


julia --project -e 'using Pkg; Pkg.instantiate(;verbose=true)'                 
julia --project -e 'using Pkg; Pkg.precompile()'          

srun julia --project $DRIVER --config_file $CONFIG_FILE --job_id=amip_prog_edmfx

echo finished