# Track some important dependencies:
example_dir = joinpath(dirname(@__DIR__), "examples")

import ClimaCore
import SciMLBase
import DiffEqBase
import OrdinaryDiffEq
import DiffEqOperators

dirs_to_monitor = [
    example_dir,
    joinpath(example_dir, "hybrid"),
    joinpath(example_dir, "hybrid", "sphere"),
    pkgdir(ClimaCore),
    pkgdir(SciMLBase),
    pkgdir(DiffEqBase),
    pkgdir(OrdinaryDiffEq),
    pkgdir(DiffEqOperators),
]

#! format: off

cli_options = [
    ("--TEST_NAME baroclinic_wave_rhoe --job_id alloc_sphere_baroclinic_wave_rhoe"),
]
#! format: on

import ReportMetrics

for clio in cli_options
    job_id = first(split(last(split(clio, "--job_id ")), " "))
    clio_in = split(clio, " ")
    ReportMetrics.report_allocs(;
        job_name = string(job_id),
        run_cmd = `$(Base.julia_cmd()) --project=perf/ --track-allocation=all perf/allocs_per_case.jl $clio_in`,
        dirs_to_monitor = dirs_to_monitor,
        n_unique_allocs = 20,
        process_filename = function process_fn(fn)
            fn = "ClimaAtmos.jl/" * last(split(fn, "climaatmos-ci/"))
            fn = last(split(fn, "depot/cpu/packages/"))
            return fn
        end,
    )
end
