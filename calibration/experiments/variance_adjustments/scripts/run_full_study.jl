#!/usr/bin/env julia
#
# Entry point you can run from **any working directory** (no `cd` required):
#   julia /path/to/variance_adjustments/scripts/run_full_study.jl [flags]
#
# `--project=...` is optional: the parent driver activates `variance_adjustments/` from its path.
#
include(joinpath(@__DIR__, "..", "run_full_study.jl"))
run_full_study!(parse_full_study_cli(collect(String, ARGS)))
