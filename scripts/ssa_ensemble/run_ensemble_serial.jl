# run_ensemble_serial.jl -- run every site of the sea-salt history ensemble in
# ONE Julia process.
#
# The build of a prognostic-EDMF column with prognostic sea salt is dominated
# by compile-time inference of the implicit solver (hours, growing ~quadratically
# with the number of tracer names), while a 5-day column then integrates in
# about a minute. Every site shares the same simulation TYPE, so a single
# process pays the compile once and each further site costs only its runtime;
# a slurm array of 24 processes would pay the compile 24 times.
#
# Usage: julia --project=<env> run_ensemble_serial.jl <site1.yml> [<site2.yml> ...]
# Each config carries its own `job_id`; output lands in the default
# `output/<job_id>/output_active` under the current directory.
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA

configs = ARGS
isempty(configs) && error("no site configs given")
println("ENSEMBLE_SITES=", length(configs)); flush(stdout)
t_start = time()
for (i, config_file) in enumerate(configs)
    t0 = time()
    status = try
        sim = CA.AtmosSimulation(CA.AtmosConfig(config_file))
        println("ENSEMBLE_BUILD_SECONDS[$i]=", round(time() - t0; digits = 1)); flush(stdout)
        res = CA.solve_atmos!(sim)
        string(res.ret_code)
    catch err
        # One site failing must not abort the rest: the compile investment is
        # shared, so the cheapest recovery is to go on and rerun the failures.
        showerror(stderr, err, catch_backtrace()); println(stderr)
        "exception"
    end
    println("ENSEMBLE_SITE[$i]=", basename(config_file), " status=", status,
            " seconds=", round(time() - t0; digits = 1)); flush(stdout)
end
println("ENSEMBLE_TOTAL_SECONDS=", round(time() - t_start; digits = 1))
