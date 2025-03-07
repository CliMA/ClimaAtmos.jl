# Generate a YAML pipeline to run CPU scaling tests on buildkite
# To run: `julia --project=.buildkite .buildkite/cpu_scaling_pipeline/generate_pipeline.jl`
# nodes = (1, 2, 4, 8, 16, 32)
# helems = (30, 42, 60, 84, 120, 170)
nodes = (1, 2, 4)
helems = (30, 42, 60)

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

function generate_step(nodes::Int, helems::Int)
    return Dict(
        "label" => ":computer: $nodes node, 16 processes per node, helem = $helems",
        "command" => "srun julia --color=yes --project=.buildkite .buildkite/ci_driver.jl --config_file $MBW_SCALING_CONFIG_PATH/moist_baroclinic_wave_helem_$(helems)_0M_ws.yml --job_id moist_baroclinic_wave_helem_$(helems)_0M_ws",
        "artifact_paths" => "moist_baroclinic_wave_helem_$(helems)_0M_ws/output_active/*",
        "agents" => Dict(
            "slurm_constraint" => CPU_TYPE,
            "queue" => "new-central",
            "slurm_nodes" => nodes,
            "slurm_tasks_per_node" => 16,
            "slurm_cpus_per_task" => 1,
            "slurm_mem" => 0,
            "slurm_time" => "1:00:00",
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
            "steps" => [generate_step.(nodes, helems)...],
        ),
    ],
)
YAML.write_file(".buildkite/cpu_scaling_pipeline/pipeline.yml", pipeline)
