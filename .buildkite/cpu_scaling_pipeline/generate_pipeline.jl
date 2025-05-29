# Generate a YAML pipeline to run CPU scaling tests on buildkite
# To run: `julia --project=.buildkite .buildkite/cpu_scaling_pipeline/generate_pipeline.jl`
# nodes = (1, 2, 4, 8, 16, 32)
# helems = (30, 42, 60, 84, 120, 170)

#strong scaling
ss_nodes = (1, 2, 4) # number of nodes for weak scaling runs
ss_helems = (30, 60, 120) # helems for weak scaling runs
ss_procspernode = 16 # number of MPI processes per node

# weak scaling
ws_nodes = (1, 2, 4) # number of nodes for weak scaling runs
ws_helems = (30, 42, 60) # helems for weak scaling runs
ws_procspernode = 16 # number of MPI processes per node

import YAML

agents = Dict("modules" => "climacommon/2024_12_16", "queue" => "new-central")
env = Dict(
    "SLURM_KILL_BAD_EXIT" => 1,
    "OPENBLAS_NUM_THREADS" => 1,
    "JULIA_NVTX_CALLBACKS" => "gc",
    "JULIA_LOAD_PATH" => "\${JULIA_LOAD_PATH}:\${BUILDKITE_BUILD_CHECKOUT_PATH}/.buildkite",
)

CPU_TYPE = "icelake"
MBW_SCALING_CONFIG_PATH = "config/mbw_scaling_configs"

init_step = Dict(
    "label" => "init :computer:",
    "key" => "init_cpu_env",
    "agents" => Dict("slurm_cpus_per_task" => 8),
    "command" => [
        """echo --- Instantiate project""",
        "julia --project -e 'using Pkg; Pkg.instantiate(;verbose=true)'",
        "julia --project -e 'using Pkg; Pkg.precompile(); Pkg.status()'",
        """echo --- Instantiate .buildkite""",
        "julia --project=.buildkite -e 'using Pkg; Pkg.instantiate(;verbose=true)'",
        "julia --project=.buildkite -e 'using Pkg; Pkg.precompile(); Pkg.status()'",
    ],
    "env" => Dict(
        "JULIA_NUM_PRECOMPILE_TASKS" => 8,
        "JULIA_MAX_NUM_PRECOMPILE_FILES" => 50,
    ),
)

function generate_step_ws(nodes::Int, helems::Int, procspernode::Int)
    return Dict(
        "label" => ":computer: MBW weak scaling, $nodes nodes, $procspernode processes per node, helem = $helems",
        "command" => "srun julia --color=yes --project=.buildkite .buildkite/ci_driver.jl --config_file $MBW_SCALING_CONFIG_PATH/moist_baroclinic_wave_helem_$(helems)_0M_ws.yml --job_id moist_baroclinic_wave_helem_$(helems)_0M_ws",
        "artifact_paths" => "moist_baroclinic_wave_helem_$(helems)_0M_ws/output_active/*",
        "key" => "ws_$(nodes)_nodes",
        "agents" => Dict(
            "slurm_constraint" => CPU_TYPE,
            "queue" => "new-central",
            "slurm_nodes" => nodes,
            "slurm_tasks_per_node" => procspernode,
            "slurm_cpus_per_task" => 1,
            "slurm_mem" => 0,
            "slurm_time" => "1:00:00",
            "slurm_reservation" => "false",
            "slurm_exclusive" => true,
        ),
    )
end

function generate_step_ss(nodes::Int, helems::Int, procspernode::Int)
    return Dict(
        "label" => ":computer: MBW strong scaling, $nodes nodes, $procspernode processes per node, helem = $helems",
        "command" => "srun julia --color=yes --project=.buildkite .buildkite/ci_driver.jl --config_file $MBW_SCALING_CONFIG_PATH/moist_baroclinic_wave_helem_$(helems)_0M_ss.yml --job_id moist_baroclinic_wave_helem_$(helems)_0M_ss",
        "artifact_paths" => "moist_baroclinic_wave_helem_$(helems)_0M_ss/output_active/*",
        "key" => "ss_$(nodes)_nodes",
        "agents" => Dict(
            "slurm_constraint" => CPU_TYPE,
            "queue" => "new-central",
            "slurm_nodes" => nodes,
            "slurm_tasks_per_node" => procspernode,
            "slurm_cpus_per_task" => 1,
            "slurm_mem" => 0,
            "slurm_time" => "1:00:00",
            "slurm_reservation" => "false",
            "slurm_exclusive" => true,
        ),
    )
end

pipeline = Dict(
    "agents" => agents,
    "env" => env,
    "steps" => [
        init_step,
        "wait",
        Dict(
            "group" => "Moist Baroclinic Wave, weak scaling",
            "steps" => [
                generate_step_ws.(ws_nodes, ws_helems, ws_procspernode)...,
            ],
        ),
        Dict(
            "group" => "Moist Baroclinic Wave, strong scaling",
            "steps" => [
                generate_step_ss.(ss_nodes, ss_helems, ss_procspernode)...,
            ],
        ),
    ],
)
YAML.write_file(".buildkite/cpu_scaling_pipeline/pipeline.yml", pipeline)
