import ClimaComms
import ClimaAtmos as CA
import ClimaCore: Fields
import YAML
import ClimaComms

import Random
# Random.seed!(Random.MersenneTwister())
Random.seed!(1234)

# --> get config
configs_path = joinpath(pkgdir(CA), "config/model_configs/")
pth = joinpath(configs_path, "kinematic_driver.yml");
job_id = "kinematic_driver";
config_dict = YAML.load_file(pth)
# <--


config = CA.AtmosConfig(config_dict; job_id)
simulation = CA.get_simulation(config);

sol_res = CA.solve_atmos!(simulation);  # solve!

(; integrator) = simulation;
(; p) = integrator;
(; atmos, params) = p;

# --> Make ci plots
# ]add ClimaAnalysis, ClimaCoreSpectra
include(joinpath(pkgdir(CA), "post_processing", "ci_plots.jl"))
make_plots(Val(:kinematic_driver), simulation.output_dir)
# <--

# --> ClimaAnalysis
import ClimaAnalysis
# using ClimaAnalysis.Visualize
import ClimaAnalysis.Visualize as viz
using ClimaAnalysis.Utils: kwargs
using CairoMakie;
CairoMakie.activate!();
# using GLMakie; GLMakie.activate!()
simdir = ClimaAnalysis.SimDir(simulation.output_dir);

# entr = get(simdir; short_name = "entr")
# entr.dims  # (time, x, y, z)

# fig = Figure();
# viz.plot!(fig, entr, time=0, x=0, y=0, more_kwargs = Dict(:axis => kwargs(dim_on_y = true)))
# viz.plot!(fig, entr, x=0, y=0);
# fig
# <--



#=
%use upwinding for the rain -- yes!

qr, qs, all specific humidities,

take 30% acoustic Courant number c=300m/s, divided by vertical res
- for ~200

ill make some plots

is smag applied to qr??? check this!

most likely precip would cause blow-ups.
=#
