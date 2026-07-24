import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import Random
Random.seed!(1234)

import ClimaTimeSteppers
using BenchmarkTools

config_path = "config"

config_file = [
    joinpath(pkgdir(CA), config_path, "common_configs", "numerics_sphere_he16ze63.yml"),
    joinpath(
        pkgdir(CA),
        config_path,
        "longrun_configs",
        "amip_target.yml",
    ),
]
config = CA.AtmosConfig(config_file; job_id=nothing)
simulation = CA.get_simulation(config)
(; integrator) = simulation;





# step to compile
for i in 1:10
    ClimaTimeSteppers.step!(integrator)
end

function step_three_times(integrator)
    for i in 1:3
        ClimaTimeSteppers.step!(integrator)
    end
    return
end

step_three_times(integrator)
Y = integrator.u;
p = integrator.p;
t = integrator.t;
Yₜ = similar(Y);
Yₜ_lim = similar(Y);

CA.remaining_tendency!(Yₜ, Yₜ_lim, Y, p, t)
BenchmarkTools.@benchmark CUDA.@sync CA.remaining_tendency!(
           $Yₜ,
           $Yₜ_lim,
           $Y,
           $p,
           $t,
       )
