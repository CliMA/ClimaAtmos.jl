#=
julia --project=perf
using Revise; if !("--config_file" in ARGS)
    push!(ARGS, "--config_file")
    push!(ARGS, "config/default_configs/default_perf.yml")
end; include("perf/jet.jl")
=#
redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(true)))
import ClimaCore
import ClimaCore.Fields
ClimaCore.Fields.truncate_printing_field_types() = true
ClimaCore.MatrixFields.truncate_printing_field_types() = true
if !("--config_file" in ARGS)
    push!(ARGS, "--config_file")
    push!(ARGS, "config/default_configs/default_perf.yml")
end
import Random
Random.seed!(1234)
import ClimaAtmos as CA
# config_dict = Dict("z_elem" => 63, "dt" => "10secs", "t_end" => "3600secs")
# config = AtmosCoveragePerfConfig(config_dict)
# config_file = ARGS[1]
# config_dict = YAML.load_file(config_file)
if !(@isdefined(config))
    include("common.jl")
    config = CA.AtmosConfig()

    simulation = CA.get_simulation(config)
    (; integrator) = simulation
end

import JET

import SciMLBase
SciMLBase.step!(integrator) # Make sure no errors
JET.@test_opt ignored_modules = (HDF5,CUDA,NCDatasets) SciMLBase.step!(integrator)

# (; u, p, t) = integrator;
# Yₜ = copy(u);
# Y = u;
# colidx = Fields.ColumnIndex((1,1),1);
# @show p.atmos.precip_model
# CA.precipitation_tendency!(Yₜ, Y, p, t, colidx, p.atmos.precip_model)
# JET.@test_opt CA.precipitation_tendency!(Yₜ, Y, p, t, colidx, p.atmos.precip_model)

