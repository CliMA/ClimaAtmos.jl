# Dependencies
using Distributed

include(joinpath(pkgdir(CA),"LWP_N_scripts","all_paths.jl"))

# Start with 2 workers if not specified.
if nprocs() == 1
    addprocs(2)
end

# Environment Activation.
@everywhere begin
    using Pkg
    Pkg.activate("C:/Users/eric8/Desktop/CliMA/ClimaAtmos.jl/.buildkite")
end

# Worker Dependencies.
@everywhere begin
    import ClimaComms
    import ClimaAtmos as CA
    import Random
end

# Parallelization Block.
@everywhere begin
    ClimaComms.@import_required_backends

    function run_simulation(config_file::String)

        config = CA.AtmosConfig(config_file)
        simulation = CA.AtmosSimulation(config)
        sol_res = CA.solve_atmos!(simulation)

        include(joinpath(pkgdir(CA), "ci_plots_extension.jl"))
        if ClimaComms.iamroot(config.comms_ctx)
            make_plots(Val(Symbol(simulation.job_id)), simulation.output_dir)
        end

        return "finished run $(simulation.job_id) → output in $(simulation.output_dir)"
    end
end

# Parallelizing prognostic EDMF+2M configuration files.
config_dir = prognostic_2M_config
config_files = sort(readdir(config_dir; join = true))
@info "Found $(length(config_files)) config files."

# Run all configs in parallel.
results = pmap(run_simulation, config_files)

println("All runs finished.")
println(results)
