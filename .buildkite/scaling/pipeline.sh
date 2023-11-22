#! /bin/bash
set -euo pipefail

FT="Float32"
resolutions=("low" "mid" "high")
process_counts=("1 2 4 8 16 32" "1 2 4 8 16 32 64" "1 2 4 8 16 32 64 128")
max_procs_per_node=16 # limit this artificially for profiling
profiling=disable
exclusive=true
mpi_impl="openmpi"

parent_folder=scaling_configs
mkdir -p "$parent_folder"

low_resolution_lines=\
"t_end: 10days
dt: 400secs
z_elem: 10
h_elem: 6
kappa_4: 2e17"

medium_resolution_lines=\
"t_end: 4days
dt: 150secs
z_elem: 45
dz_bottom: 30
h_elem: 16
kappa_4: 1e16"

high_resolution_lines=\
"t_end: 1days
dt: 50secs
z_elem: 45
h_elem: 30
kappa_4: 5e14"

# Create configuration files for each resolution
for i in "${!resolutions[@]}"; do
    resolution="${resolutions[$i]}"
    proc_counts="${process_counts[$i]}"

    for nprocs in $proc_counts; do
        filename="${resolution}_res_${FT}_${nprocs}.yml"
        folder_name="${resolution}_res_Float32"
        mkdir -p "$parent_folder/$folder_name"
        filepath="$parent_folder/$folder_name/$filename"

        echo "job_id: sphere_held_suarez_${resolution}_res_rhoe_${nprocs}" > "$filepath"
        echo "forcing: held_suarez" >> "$filepath"
        echo "FLOAT_TYPE: $FT" >> "$filepath"
        echo "tracer_upwinding: none" >> "$filepath"

        case "$resolution" in
            "low")
                echo -e "$low_resolution_lines" >> "$filepath"
                ;;
            "mid")
                echo -e "$medium_resolution_lines" >> "$filepath"
                ;;
            "high")
                echo -e "$high_resolution_lines" >> "$filepath"
                ;;
        esac
    done
done

# set up environment and agents
cat << 'EOM'
agents:
  queue: central
  modules: julia/1.9.4 cuda/12.2 ucx/1.14.1_cuda-12.2 openmpi/4.1.5_cuda-12.2 nsight-systems/2023.3.1

env:
  JULIA_LOAD_PATH: "${JULIA_LOAD_PATH}:${BUILDKITE_BUILD_CHECKOUT_PATH}/.buildkite"
  OPENBLAS_NUM_THREADS: 1
  JULIA_NVTX_CALLBACKS: gc
  OMPI_MCA_opal_warn_on_missing_libcuda: 0
  JULIA_MAX_NUM_PRECOMPILE_FILES: 100
  JULIA_CPU_TARGET: 'broadwell;skylake'
  SLURM_KILL_BAD_EXIT: 1
  JULIA_NVTX_CALLBACKS: gc
  JULIA_CUDA_MEMORY_POOL: none
  JULIA_MPI_HAS_CUDA: "true"
  MPITRAMPOLINE_LIB: "/groups/esm/software/MPIwrapper/ompi4.1.5_cuda-12.2/lib64/libmpiwrapper.so"
  MPITRAMPOLINE_MPIEXEC: "/groups/esm/software/MPIwrapper/ompi4.1.5_cuda-12.2/bin/mpiwrapperexec"

steps:
  - label: "init :computer:"
    key: "init_cpu_env"
    command:
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

for i in "${!resolutions[@]}"; do

res="${resolutions[$i]}"
proc_counts="${process_counts[$i]}"

cat << EOM
  - group: "ClimaAtmos $res-resolution scaling tests"
    steps:

EOM

for nprocs in ${proc_counts[@]}; do

    job_id="sphere_held_suarez_${res}_res_rhoe_$nprocs"
    folder_name="${res}_res_${FT}"
    config_file="$parent_folder/$folder_name/${res}_res_${FT}_${nprocs}.yml"
    command="julia --color=yes --project=examples examples/hybrid/driver.jl --config_file $config_file"

if [[ "$mpi_impl" == "mpich" ]]; then
    rank_env_var="PMI_RANK"
else
    rank_env_var="OMPI_COMM_WORLD_RANK"
fi
if [[ "$profiling" == "enable" ]]; then
    command="nsys profile --sample=none --trace=nvtx,mpi --mpi-impl=$mpi_impl --output=${job_id}/rank-%q{$rank_env_var} $command"
    cpus_per_proc=2
else
    cpus_per_proc=1
fi
launcher="srun --cpu-bind=cores"

if [[ "$res" == "low" ]]; then
    time="04:00:00"
elif [[ "$res" == "mid" ]]; then
    if [[ $nprocs -gt 2 ]]; then
        time="04:00:00"
    else
        time="16:00:00"
    fi
else
    if [[ $nprocs -gt 8 ]]; then
        time="04:00:00"
    elif [[ $nprocs -ge 4 ]]; then
        time="10:00:00"
    elif [[ $nprocs -ge 2 ]]; then
        time="16:00:00"
    else
        time="24:00:00"
    fi
fi


cat << EOM
    - label: "$nprocs"
      key: "$job_id"
      command:
        - "$launcher $command"
        - "find ${job_id} -iname '*.nsys-rep' -printf '%f\\\\n' | sort -V | jq --raw-input --slurp 'split(\"\n\") | .[0:-1] | {files: .} + {\"extension\": \"nsys-view\", \"version\": \"1.0\"}' > ${job_id}/${job_id}.nsys-view"
        - "find ${job_id} -iname '*.nsys-*' | sort -V | tar cvzf ${job_id}-nsys.tar.gz -T -"
      artifact_paths:
        - "${job_id}/scaling_data_${nprocs}_processes.jld2"
        - "${job_id}-nsys.tar.gz"
      env:
        CLIMACORE_DISTRIBUTED: "MPI"
      agents:
        slurm_time: $time
EOM

if [[ "$exclusive" == "true" ]]; then

    nnodes=$(( (nprocs+max_procs_per_node-1) /max_procs_per_node )) # ceiling divide
    procs_per_node=$(( nprocs/nnodes ))

    cat << EOM
        slurm_nodes: $nnodes
        slurm_tasks_per_node: $procs_per_node
        slurm_cpus_per_task: $cpus_per_proc
        slurm_mem: 0
        slurm_exclusive:
EOM
else
    cat << EOM
        slurm_ntasks: $nprocs
        slurm_cpus_per_task: $cpus_per_proc
        slurm_mem_per_cpu: 8G
EOM
fi

done

cat << EOM
    - wait: ~
      continue_on_failure: true

    - label: ":chart_with_upwards_trend:"
      key: "cpu_scaling_plots_$res-resolution"
      command:
        - "julia --color=yes --project=examples post_processing/plot_scaling_results.jl sphere_held_suarez_${res}_res_rhoe"
      artifact_paths:
        - "${res}-*.png"
        - "${res}-*.pdf"
      agents:
        slurm_nodes: 1
        slurm_tasks_per_node: 1

EOM

done

cat << EOM
  - wait: ~
    continue_on_failure: true

  - label: ":broom: clean up config files"
    command: "rm -rf $parent_folder"

EOM
