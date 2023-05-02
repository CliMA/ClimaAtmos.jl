# We only need to include AtmosArtifacts module
include(joinpath(dirname(@__DIR__), "src", "utils", "AtmosArtifacts.jl"))
using .AtmosArtifacts: trigger_download
trigger_download()
