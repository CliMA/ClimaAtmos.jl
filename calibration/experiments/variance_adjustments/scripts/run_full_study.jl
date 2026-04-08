#!/usr/bin/env julia
#
# Full study driver: activate this experiment's `Project.toml`, load `lib/run_full_study.jl`, run CLI when executed as main.
#
# From the experiment directory:
#   julia --project=. scripts/run_full_study.jl [flags]
# From any cwd (absolute path):
#   julia /path/to/variance_adjustments/scripts/run_full_study.jl [flags]
#
# REPL (from experiment root):
#   using Pkg; Pkg.activate("."); include("scripts/run_full_study.jl"); run_full_study!()
#
import Pkg

const _VA_ROOT = dirname(@__DIR__) |> abspath
Pkg.activate(_VA_ROOT)
include(joinpath(_VA_ROOT, "lib", "run_full_study.jl"))
if abspath(Base.PROGRAM_FILE) == abspath(@__FILE__)
    run_full_study!(parse_full_study_cli(collect(String, ARGS)))
end
