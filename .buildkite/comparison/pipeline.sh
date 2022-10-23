#! /bin/bash
set -euo pipefail

slurm_time="48:00:00" # max slurm time
script=".buildkite/comparison/comparison_sphere_held_suarez.sh"
low_res_process_counts=(1 2 3 6 24)
high_res_process_counts=(1 2 3 6 24 54 96 216)
FT="Float64"
resolutions=("low" "high")
max_procs_per_node=32
profiling=disable

# set up environment and agents
cat << EOM
env:
  JULIA_VERSION: "1.8.1"
  MPICH_VERSION: "4.0.0"
  OPENMPI_VERSION: "4.1.1"
  CUDA_VERSION: "11.3"
  OPENBLAS_NUM_THREADS: 1
  CLIMATEMACHINE_SETTINGS_FIX_RNG_SEED: "true"

agents:
  config: cpu
  queue: central
  slurm_time: $slurm_time

steps:
  - label: "init :computer:"
    key: "init_cpu_env"
    command:
      - echo "--- Configure MPI"
      - julia -e 'using Pkg; Pkg.add("MPIPreferences"); using MPIPreferences; use_system_binary()'

      - echo "--- Instantiate"
      - "julia --project=examples -e 'using Pkg; Pkg.instantiate(;verbose=true)'"
      - "julia --project=examples -e 'using Pkg; Pkg.precompile()'"
      - "julia --project=examples -e 'using Pkg; Pkg.status()'"

    agents:
      slurm_cpus_per_task: 8
    env:
      JULIA_NUM_PRECOMPILE_TASKS: 8

  - wait

EOM

for res in ${resolutions[@]}; do

cat << EOM
  - group: "ClimaAtmos $res-resolution tests"
    steps:

EOM


if [[ "$res" == "low" ]]; then
    process_counts=${low_res_process_counts[@]}
else
    process_counts=${high_res_process_counts[@]}
fi

for nprocs in ${process_counts[@]}; do

    nnodes=$(( (nprocs+max_procs_per_node-1) /max_procs_per_node )) # ceiling divide
    procs_per_node=$(( nprocs/nnodes ))

    job_id="comparison_sphere_held_suarez_${res}_res_rhoe_$nprocs"

if [[ "$res" == "low" ]]; then
    command="julia --color=yes --project=examples examples/hybrid/driver.jl --job_id $job_id --forcing held_suarez --enable_threading false --FLOAT_TYPE $FT --tracer_upwinding none --t_end 10days --dt 400secs --z_elem 10 --h_elem 4 --kappa_4 2e17"
else
    command="julia --color=yes --project=examples examples/hybrid/driver.jl --job_id $job_id --forcing held_suarez --enable_threading false --FLOAT_TYPE $FT --tracer_upwinding none --t_end 1days --dt 50secs --z_elem 45 --h_elem 24 --kappa_4 5e14"
fi

if [[ "$profiling" == "enable" ]]; then
    command="nsys profile --trace=nvtx,mpi --mpi-impl=mpich --output=${job_id}/report.%q{NPROCS}.%q{PMI_RANK} $command"
fi

cat << EOM
    - label: ":computer: MPI Held-Suarez $res resolution test(ρe_tot) - ($nprocs) process"
      key: $job_id
      command:
        - module load cuda/11.3 nsight-systems/2022.2.1
        - mpiexec $command
      artifact_paths:
        - "$job_id/scaling_data_${nprocs}_processes.jld2"
        - "$job_id/report.*.nsys-rep"
      env:
        CLIMACORE_DISTRIBUTED: "MPI"
        NPROCS: $nprocs
      agents:
        config: cpu
        queue: central
        slurm_nodes: $nnodes
        slurm_ntasks: $nprocs
        slurm_tasks_per_node: $procs_per_node
        slurm_mem: 0
        slurm_exclusive:

EOM
done

cat << EOM
    - wait

    - label: ":computer: comparison plots ($res-resolution simulations)"
      key: "cpu_comparison_plots_$res-resolution"
      command:
        - "julia --color=yes --project=examples examples/hybrid/comparison_plots.jl --job_id comparison_sphere_held_suarez_${res}_res_rhoe"
      artifact_paths:
        - "$res-resolution_comparison.*"
        - "$res-resolution_sypd.*"
        - "$res-resolution_Scaling.*"
        - "$res-resolution_Scaling_efficiency.*"
      agents:
        config: cpu
        queue: central
        slurm_nodes: 1
        slurm_tasks_per_node: 1

EOM
done
