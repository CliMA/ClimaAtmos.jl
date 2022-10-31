# Track some important dependencies:
example_dir = joinpath(dirname(@__DIR__), "examples")
include(joinpath(example_dir, "hybrid", "cli_options.jl"));

import ClimaAtmos
import ClimaCore
import SciMLBase
import DiffEqBase
import OrdinaryDiffEq
import ClimaTimeSteppers
import Thermodynamics
import SurfaceFluxes
import CloudMicrophysics

dirs_to_monitor = [
    pkgdir(ClimaAtmos),
    example_dir,
    joinpath(example_dir, "hybrid"),
    joinpath(example_dir, "hybrid", "sphere"),
    pkgdir(ClimaCore),
    pkgdir(SciMLBase),
    pkgdir(DiffEqBase),
    pkgdir(OrdinaryDiffEq),
    pkgdir(ClimaTimeSteppers),
    pkgdir(Thermodynamics),
    pkgdir(SurfaceFluxes),
    pkgdir(CloudMicrophysics),
]
@info "`dirs_to_monitor` (Pre)  = $dirs_to_monitor"
dirs_to_monitor_filtered = filter(x -> x isa String, dirs_to_monitor)
@info "`dirs_to_monitor` (Post) = $dirs_to_monitor_filtered"
dirs_to_monitor_filtered = String.(dirs_to_monitor_filtered)
if length(dirs_to_monitor_filtered) ≠ length(dirs_to_monitor)
    @warn "Some packages' directories not found."
end

#! format: off

dict = parsed_args_per_job_id(; trigger = "benchmark.jl")
for k in keys(dict)
    dict[k]["job_id"] = "allocs_"*dict[k]["job_id"]

    # Lower resolution, since allocation tracking is expensive
    dict[k]["h_elem"] = 6
    dict[k]["z_elem"] = 18
end
cli_options = [
    non_default_command_line_flags_parsed_args(dict["perf_target_unthreaded"]),
]
#! format: on

import ReportMetrics

for clio in cli_options
    job_id = first(split(last(split(clio, "--job_id ")), " "))
    clio_in = split(clio, " ")
    @info "CL options: `$clio_in`"
    ReportMetrics.report_allocs(;
        job_name = string(job_id),
        run_cmd = `$(Base.julia_cmd()) --project=perf/ --track-allocation=all perf/allocs_per_case.jl $clio_in`,
        dirs_to_monitor = dirs_to_monitor_filtered,
        n_unique_allocs = 20,
        process_filename = function process_fn(fn)
            fn = "ClimaAtmos.jl/" * last(split(fn, "climaatmos-ci/"))
            fn = last(split(fn, "depot/cpu/packages/"))
            fn = "ClimaAtmos.jl/" * last(split(fn, "longruns/"))
            return fn
        end,
    )
end
