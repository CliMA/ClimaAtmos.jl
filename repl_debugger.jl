#ENV["CLIMACOMMS_DEVICE"]="CUDA"

include("driver.jl")
include(joinpath(@__DIR__, "post_processing", "jacobian_summary.jl"))
import ClimaCore.Fields
using PrettyTables
import Statistics: mean

diffusion = "DecayWithHeightDiffusion"
timestepper = "ARS343" 
dt = "400secs"

config_file = "config/model_configs/baroclinic_wave_equil.yml"

# Config 1
# config_1 = CA.AtmosConfig(config_file);
# config_1.parsed_args["vert_diff"]=diffusion;
# config_1.parsed_args["dt"]=dt;
# config_1.parsed_args["implicit_diffusion"]=true
# config_1.parsed_args["approximate_linear_solve_iters"]=2
# config_1.parsed_args["ode_algo"]=timestepper
# simulation_1 = CA.AtmosSimulation(config_1);
# step!(simulation_1.integrator);
# S1 = simulation_1.integrator;
# ᶜp1 = S1.p.precomputed.ᶜp
# Δᶜp1 = Fields.field_values(Fields.level(ᶜp1,2)) .- Fields.field_values(Fields.level(ᶜp1,1))
# open("config_results.txt", "a") do file
#     println(file, "Config 1: ", (config_1.parsed_args["vert_diff"], 
#                                  config_1.parsed_args["dt"],
#                                  config_1.parsed_args["implicit_diffusion"], 
#                                  config_1.parsed_args["approximate_linear_solve_iters"], 
#                                  config_1.parsed_args["deep_atmosphere"],
#                                  config_1.parsed_args["ode_algo"]))
#     println(file, "Extrema: ", extrema(parent(Δᶜp1)))
# end

# Config 2
config_2 = CA.AtmosConfig(config_file);
config_2.parsed_args["vert_diff"]=diffusion;
config_2.parsed_args["dt"]=dt;
config_2.parsed_args["implicit_diffusion"]=true
config_2.parsed_args["ode_algo"]=timestepper
# config_2.parsed_args["ode_algo"]="SSPKnoth"
config_2.parsed_args["deep_atmosphere"]=false
config_2.parsed_args["use_auto_jacobian"]=false
config_2.parsed_args["approximate_linear_solve_iters"]=5
simulation_2 = CA.AtmosSimulation(config_2);
ᶜp2 = simulation_2.integrator.p.precomputed.ᶜp
Δᶜp2 = Fields.field_values(Fields.level(ᶜp2,2)) .- Fields.field_values(Fields.level(ᶜp2,1))
open("config_results.txt", "a") do file
    println(file, "Config 2: ", (config_2.parsed_args["vert_diff"], 
                                 config_2.parsed_args["dt"],
                                 config_2.parsed_args["implicit_diffusion"], 
                                 config_2.parsed_args["approximate_linear_solve_iters"], 
                                 config_2.parsed_args["deep_atmosphere"],
                                 config_2.parsed_args["ode_algo"]))
    println(file, "Extrema: ", extrema(parent(Δᶜp2)))
end
tmp_integrator_bef = deepcopy(simulation_2.integrator)
hasproperty(safehouse, :tmp_integrator_bef) ? print_jacobian_summary(safehouse.tmp_integrator_bef, simulation_2.integrator) : nothing
step!(simulation_2.integrator);
tmp_integrator_aft = deepcopy(simulation_2.integrator)
# @Main.infiltrate
hasproperty(safehouse, :tmp_integrator_aft) ? print_jacobian_summary(safehouse.tmp_integrator_aft, simulation_2.integrator) : nothing
@Main.infiltrate
S2 = simulation_2.integrator;
ᶜp2 = S2.p.precomputed.ᶜp
Δᶜp2 = Fields.field_values(Fields.level(ᶜp2,2)) .- Fields.field_values(Fields.level(ᶜp2,1))
open("config_results.txt", "a") do file
    println(file, "Config 2: ", (config_2.parsed_args["vert_diff"], 
                                 config_2.parsed_args["dt"],
                                 config_2.parsed_args["implicit_diffusion"], 
                                 config_2.parsed_args["approximate_linear_solve_iters"], 
                                 config_2.parsed_args["deep_atmosphere"],
                                 config_2.parsed_args["ode_algo"]))
    println(file, "Extrema: ", extrema(parent(Δᶜp2)))
    println(file, "\n-----------------------------------\n")
end

# Config 3
# config_3 = CA.AtmosConfig(config_file);
# config_3.parsed_args["vert_diff"]=diffusion;
# config_3.parsed_args["implicit_diffusion"]=true
# config_3.parsed_args["dt"]=dt;
# config_3.parsed_args["approximate_linear_solve_iters"]=10
# config_3.parsed_args["ode_algo"]=timestepper
# simulation_3 = CA.AtmosSimulation(config_3);
# step!(simulation_3.integrator);
# S3 = simulation_3.integrator;
# ᶜp3 = S3.p.precomputed.ᶜp
# Δᶜp3 = Fields.field_values(Fields.level(ᶜp3,2)) .- Fields.field_values(Fields.level(ᶜp3,1))
# open("config_results.txt", "a") do file
#     println(file, "Config 3: ", (config_3.parsed_args["vert_diff"], 
#                                  config_3.parsed_args["dt"],
#                                  config_3.parsed_args["implicit_diffusion"], 
#                                  config_3.parsed_args["approximate_linear_solve_iters"], 
#                                  config_3.parsed_args["deep_atmosphere"],
#                                  config_3.parsed_args["ode_algo"]))
#     println(file, "Extrema: ", extrema(parent(Δᶜp3)))
# end

# Config 4
# config_4 = CA.AtmosConfig(config_file);
# config_4.parsed_args["vert_diff"]=diffusion;
# config_4.parsed_args["implicit_diffusion"]=true
# config_4.parsed_args["dt"]=dt;
# config_4.parsed_args["ode_algo"]="SSPKnoth"
# config_4.parsed_args["deep_atmosphere"]=true
# simulation_4 = CA.AtmosSimulation(config_4);
# step!(simulation_4.integrator);
# S4 = simulation_4.integrator;
# ᶜp4 = S4.p.precomputed.ᶜp
# Δᶜp4 = Fields.field_values(Fields.level(ᶜp4,2)) .- Fields.field_values(Fields.level(ᶜp4,1))
# open("config_results.txt", "a") do file
#     println(file, "Config 4: ", (config_4.parsed_args["vert_diff"], 
#                                  config_4.parsed_args["dt"],
#                                  config_4.parsed_args["implicit_diffusion"], 
#                                  config_4.parsed_args["approximate_linear_solve_iters"], 
#                                  config_4.parsed_args["deep_atmosphere"],
#                                  config_4.parsed_args["ode_algo"]))
#     println(file, "Extrema: ", extrema(parent(Δᶜp4)))
# end

# Config 5
# config_5 = CA.AtmosConfig(config_file);
# config_5.parsed_args["vert_diff"]=diffusion;
# config_5.parsed_args["implicit_diffusion"]=false
# config_5.parsed_args["dt"]=dt;
# config_5.parsed_args["ode_algo"]="SSPKnoth"
# config_5.parsed_args["approximate_linear_solve_iters"]=2
# config_5.parsed_args["deep_atmosphere"]=false
# simulation_5 = CA.AtmosSimulation(config_5);
# step!(simulation_5.integrator);
# S5 = simulation_5.integrator;
# ᶜp5 = S5.p.precomputed.ᶜp
# Δᶜp5 = Fields.field_values(Fields.level(ᶜp5,2)) .- Fields.field_values(Fields.level(ᶜp5,1))
# open("config_results.txt", "a") do file
#     println(file, "Config 5: ", (config_5.parsed_args["vert_diff"], 
#                                  config_5.parsed_args["dt"],
#                                  config_5.parsed_args["implicit_diffusion"], 
#                                  config_5.parsed_args["approximate_linear_solve_iters"], 
#                                  config_5.parsed_args["deep_atmosphere"],
#                                  config_5.parsed_args["ode_algo"]))
#     println(file, "Extrema: ", extrema(parent(Δᶜp5)))
# end