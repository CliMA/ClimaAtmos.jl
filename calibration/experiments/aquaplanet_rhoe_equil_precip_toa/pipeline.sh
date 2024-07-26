#!/bin/bash
set -euo pipefail

n_iterations=3
EXP_DIR="calibration/experiments/aquaplanet_rhoe_equil_precip_toa"
export CALIBRATION_EXPERIMENT_DIR="$EXP_DIR"

julia --project=$EXP_DIR -e "
import Pkg; Pkg.instantiate(;verbose=true)
import ClimaCalibrate as CAL
CAL.initialize(CAL.ExperimentConfig(\"$EXP_DIR\"))
"
dependency=""
for CALIBRATION_ITERATION in $(seq 0 $((n_iterations - 1)))
do
    export CALIBRATION_ITERATION
    iter_job_id=$(
        qsub $dependency \
            -N iter_$CALIBRATION_ITERATION \
            -v CALIBRATION_ITERATION,CALIBRATION_EXPERIMENT_DIR \
            calibration/experiments/aquaplanet_rhoe_equil_precip_toa/iter.sh)
    echo "Iteration $CALIBRATION_ITERATION: $iter_job_id"
    
    # Check if iter_job_id is not empty and set dependency accordingly
    if [[ -n "$iter_job_id" ]]; then
        dependency="-W depend=afterany:$iter_job_id"
    else
        echo "Error: iter_job_id is empty. Exiting loop."
        exit 1
    fi
done
