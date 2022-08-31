#! /bin/bash
set -euxo pipefail
set +x

nargs=$#

if (( nargs != 3 ))
then
    echo "please provide arguments for \"nprocs\", \"resolution\", (high/low) and \"profiling\" (enable/disable)"
fi

nprocs=$1
resolution="$2"
profiling="$3"

job_id="sphere_held_suarez_${resolution}_res_rhoe_$nprocs"

profiling_params="nsys profile --trace=nvtx,mpi --mpi-impl=mpich --output=${job_id}/report.%q{NPROCS}"

if [[ "$resolution" == "low" ]]
then
    sim_params="mpiexec julia --color=yes --project=examples examples/hybrid/driver.jl --job_id $job_id --forcing held_suarez --enable_threading false --tracer_upwinding none --t_end 10days --dt 400secs --z_elem 10 --h_elem 4 --kappa_4 2e17"
else
    sim_params="mpiexec julia --color=yes --project=examples examples/hybrid/driver.jl --job_id $job_id --forcing held_suarez --enable_threading false --tracer_upwinding none --t_end 1days --dt 50secs --z_elem 45 --h_elem 24 --kappa_4 5e14"
fi

if [[ "$profiling" == "enable" ]]
then
    module load cuda/11.3 nsight-systems/2022.2.1
    $profiling_params $sim_params
else
    $sim_params
fi
