# Loaded on Distributed workers by `run_end_to_end.jl`.
import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
include(joinpath(@__DIR__, "run_end_to_end.jl"))