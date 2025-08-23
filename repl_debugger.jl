#ENV["CLIMACOMMS_DEVICE"]="CUDA"

include("driver.jl")
import ClimaCore.Fields

diffusion = "DecayWithHeightDiffusion"
timestepper = "ARS343" 
dt = "400secs"

config_file = "config/model_configs/baroclinic_wave_equil.yml"
config_1 = CA.AtmosConfig(config_file);
config_2 = CA.AtmosConfig(config_file);
config_3 = CA.AtmosConfig(config_file);
config_4 = CA.AtmosConfig(config_file);
config_5 = CA.AtmosConfig(config_file);

# Update config 1
config_1.parsed_args["vert_diff"]=diffusion;
config_1.parsed_args["dt"]=dt;
config_1.parsed_args["implicit_diffusion"]=true
config_1.parsed_args["approximate_linear_solve_iters"]=2
config_1.parsed_args["ode_algo"]=timestepper

# Update config 2
config_2.parsed_args["vert_diff"]=diffusion;
config_2.parsed_args["dt"]=dt;
config_2.parsed_args["implicit_diffusion"]=true
config_2.parsed_args["ode_algo"]=timestepper
config_2.parsed_args["deep_atmosphere"]=false

# Update config 3
config_3.parsed_args["vert_diff"]=diffusion;
config_3.parsed_args["implicit_diffusion"]=true
config_3.parsed_args["dt"]=dt;
config_3.parsed_args["approximate_linear_solve_iters"]=10
config_3.parsed_args["ode_algo"]=timestepper

# Update config 4
config_4.parsed_args["vert_diff"]=diffusion;
config_4.parsed_args["implicit_diffusion"]=true
config_4.parsed_args["dt"]=dt;
config_4.parsed_args["ode_algo"]="SSPKnoth"
config_5.parsed_args["deep_atmosphere"]=true

# Update config 5
config_5.parsed_args["vert_diff"]=diffusion;
config_5.parsed_args["implicit_diffusion"]=false
config_5.parsed_args["dt"]=dt;
config_5.parsed_args["ode_algo"]="SSPKnoth"
config_5.parsed_args["approximate_linear_solve_iters"]=2
config_5.parsed_args["deep_atmosphere"]=false

# Generate simulations
simulation_1 = CA.AtmosSimulation(config_1);
simulation_2 = CA.AtmosSimulation(config_2);
simulation_3 = CA.AtmosSimulation(config_3);
simulation_4 = CA.AtmosSimulation(config_4);
simulation_5 = CA.AtmosSimulation(config_5);

# Run ? 
step!(simulation_1.integrator);
S1 = simulation_1.integrator;

step!(simulation_2.integrator);
S2 = simulation_2.integrator;

step!(simulation_3.integrator);
S3 = simulation_3.integrator;

step!(simulation_4.integrator);
S4 = simulation_4.integrator;

step!(simulation_5.integrator);
S5 = simulation_5.integrator;

ᶜp1 = S1.p.precomputed.ᶜp
ᶜp2 = S2.p.precomputed.ᶜp
ᶜp3 = S3.p.precomputed.ᶜp
ᶜp4 = S4.p.precomputed.ᶜp
ᶜp5 = S5.p.precomputed.ᶜp

Δᶜp1 = Fields.field_values(Fields.level(ᶜp1,2)) .- Fields.field_values(Fields.level(ᶜp1,1))
Δᶜp2 = Fields.field_values(Fields.level(ᶜp2,2)) .- Fields.field_values(Fields.level(ᶜp2,1))
Δᶜp3 = Fields.field_values(Fields.level(ᶜp3,2)) .- Fields.field_values(Fields.level(ᶜp3,1))
Δᶜp4 = Fields.field_values(Fields.level(ᶜp4,2)) .- Fields.field_values(Fields.level(ᶜp4,1))
Δᶜp5 = Fields.field_values(Fields.level(ᶜp5,2)) .- Fields.field_values(Fields.level(ᶜp5,1))

@info (config_1.parsed_args["vert_diff"], 
       config_1.parsed_args["dt"],
       config_1.parsed_args["implicit_diffusion"], 
       config_1.parsed_args["approximate_linear_solve_iters"], 
       config_1.parsed_args["deep_atmosphere"],
       config_1.parsed_args["ode_algo"]);
extrema(parent(Δᶜp1))

@info (config_2.parsed_args["vert_diff"], 
       config_2.parsed_args["dt"],
       config_2.parsed_args["implicit_diffusion"], 
       config_2.parsed_args["approximate_linear_solve_iters"], 
       config_2.parsed_args["deep_atmosphere"],
       config_2.parsed_args["ode_algo"]);
extrema(parent(Δᶜp2))

@info (config_3.parsed_args["vert_diff"], 
       config_3.parsed_args["dt"],
       config_3.parsed_args["implicit_diffusion"], 
       config_3.parsed_args["approximate_linear_solve_iters"], 
       config_3.parsed_args["deep_atmosphere"],
       config_3.parsed_args["ode_algo"]);
extrema(parent(Δᶜp3))

@info (config_4.parsed_args["vert_diff"], 
       config_4.parsed_args["dt"],
       config_4.parsed_args["implicit_diffusion"], 
       config_4.parsed_args["approximate_linear_solve_iters"], 
       config_4.parsed_args["deep_atmosphere"],
       config_4.parsed_args["ode_algo"]);
extrema(parent(Δᶜp4))

@info (config_5.parsed_args["vert_diff"], 
       config_5.parsed_args["dt"],
       config_5.parsed_args["implicit_diffusion"], 
       config_5.parsed_args["approximate_linear_solve_iters"], 
       config_5.parsed_args["deep_atmosphere"],
       config_5.parsed_args["ode_algo"]);
extrema(parent(Δᶜp5))