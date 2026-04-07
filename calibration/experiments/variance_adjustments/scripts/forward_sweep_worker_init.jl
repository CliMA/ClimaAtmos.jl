# Loaded on Distributed workers by `sweep_forward_core.jl` (`parallel=:distributed`).
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "sweep_forward_core.jl"))
