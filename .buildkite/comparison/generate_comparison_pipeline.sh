#! /bin/bash
set -euxo pipefail
set +x

slurm_time="48:00:00" # max slurm time
script=".buildkite/comparison/comparison_sphere_held_suarez.sh"
low_res_process_counts=(1 2 3 6 24) 
high_res_process_counts=(1 2 3 6 24 54 96 216)
FT="Float64"
resolutions=("low" "high")
max_procs_per_node=32
file="./pipeline.yml"

if ! [[ -f pipeline.yml ]]
then
    echo "creating file $file"
    touch $file
fi
# set up environment and agents
cat << EOM > $file
env:
  JULIA_VERSION: "1.7.2"
  MPICH_VERSION: "4.0.0"
  OPENMPI_VERSION: "4.1.1"
  CUDA_VERSION: "11.3"
  OPENBLAS_NUM_THREADS: 1
  CLIMATEMACHINE_SETTINGS_FIX_RNG_SEED: "true"
  BUILDKITE_COMMIT: "\${BUILDKITE_COMMIT}"
  BUILDKITE_BRANCH: "\${BUILDKITE_BRANCH}"

agents:
  config: cpu
  queue: central
  slurm_time: $slurm_time

steps:
  - label: "init :computer:"
    key: "init_cpu_env"
    command:
      - "julia --project=examples -e 'using Pkg; Pkg.instantiate(;verbose=true)'"
      - "julia --project=examples -e 'using Pkg; Pkg.precompile()'"
      - "julia --project=examples -e 'using Pkg; Pkg.status()'"

    agents:
      slurm_cpus_per_task: 8
    env:
      JULIA_NUM_PRECOMPILE_TASKS: 8

EOM

for res in ${resolutions[@]}; do
    echo "  - wait" >> $file
    echo "" >> $file
    echo "  - group: \"ClimaAtmos $res-resolution tests\"" >> $file
    echo "    steps:" >> $file
    echo "" >> .pipeline.yml

    if [[ "$res" == "low" ]];
    then
        process_counts=${low_res_process_counts[@]}
    else
        process_counts=${high_res_process_counts[@]}
    fi

    for nprocs in ${process_counts[@]}; do
        nnodes=$(( (nprocs+max_procs_per_node-1) /max_procs_per_node )) # ceiling divide
        procs_per_node=$(( nprocs/nnodes ))
        echo "    - label: \":computer: MPI Held-Suarez $res resolution test(ρe_tot) - ($nprocs) process\"" >> $file
        echo "      command:" >> $file
        echo "        - echo \"\$(bash .buildkite/comparison/comparison_sphere_held_suarez.sh $nprocs $res disable $FT)\"" >> $file
        echo "      artifact_paths:" >> $file
        echo "        - \"comparison_sphere_held_suarez_${res}_res_rhoe_$nprocs/scaling_data_${nprocs}_processes.jld2\"" >> $file
        echo "        - \"comparison_sphere_held_suarez_${res}_res_rhoe_$nprocs/report.$nprocs.nsys-rep\"" >> $file
        echo "      env:" >> $file
        echo "        CLIMACORE_DISTRIBUTED: \"MPI\"" >> $file
        echo "        NPROCS: $nprocs" >> $file
        echo "      agents:" >> $file
        echo "        config: cpu" >> $file
        echo "        queue: central" >> $file
        echo "        slurm_nodes: $nnodes" >> $file
        echo "        slurm_ntasks: $nprocs" >> $file
        #echo "        slurm_tasks_per_node: $procs_per_node" >> $file
        echo "        slurm_mem: 64GB" >> $file
        echo "        slurm_exclusive:" >> $file
        echo "" >> $file
    done

    echo "  - wait" >> $file
    echo "" >> $file

    echo "  - label: \":computer: comparison plots ($res-resolution simulations)\"" >> $file
    echo "    key: \"cpu_comparison_plots_$res-resolution\"" >> $file
    echo "    command:" >> $file
    echo "      - \"julia --color=yes --project=examples examples/hybrid/comparison_plots.jl --job_id comparison_sphere_held_suarez_${res}_res_rhoe\"" >> $file
    echo "    artifact_paths:" >> $file
    echo "      - \"$res-resolution_comparison.*\"" >> $file
    echo "      - \"$res-resolution_sypd.*\"" >> $file
    echo "      - \"$res-resolution_Scaling.*\"" >> $file
    echo "      - \"$res-resolution_Scaling_efficiency.*\"" >> $file
    echo "    agents:" >> $file
    echo "      config: cpu" >> $file
    echo "      queue: central" >> $file
    echo "      slurm_nodes: 1" >> $file
    echo "      slurm_tasks_per_node: 1" >> $file

    echo "" >> $file
done
