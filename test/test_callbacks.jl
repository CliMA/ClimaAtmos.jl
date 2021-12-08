# Generate simple model using shallow water equations for bickley jet problem
include("test_cases/initial_conditions/bickley_jet_2d_plane.jl");
FT = Float64;
npolynomial = 4;
nelements = (2, 2);
params = map(FT, (
    g = 9.8,  # gravitational constant
    D₄ = 1e-4,  # hyperdiffusion constant
    ϵ = 0.1,  # perturbation size for initial condition
    l = 0.5,  # Gaussian width
    k = 0.5,  # sinusoidal wavenumber
    h₀ = 1.0,  # reference density
));

@unpack h, u, c = init_bickley_jet_2d_plane(params);
domain = Plane(
    FT,
    xlim = (-2π, 2π),
    ylim = (-2π, 2π),
    nelements = nelements,
    npolynomial = npolynomial,
);
model = ShallowWaterModel(domain = domain, parameters = params);

# Populate Callback Containers
temp_filepath = joinpath(@__DIR__, "callback_tests")
mkpath(temp_filepath)
cb_1 = JLD2Output(model, temp_filepath, "TestFilename1", 0.01);
cb_2 = JLD2Output(model, temp_filepath, "TestFilename2", 0.02);

# Generate CallbackSet 
cb_set = CallbackSet(generate_callback(cb_1), generate_callback(cb_2))

# Type Checks
@test generate_callback(cb_1) isa DiscreteCallback
@test generate_callback(cb_2) isa DiscreteCallback

# Generate simple simulation data for test
simulation = Simulation(
    model,
    SSPRK33(),
    dt = 0.01,
    tspan = (0.0, 0.03),
    callbacks = cb_set,
)
run!(simulation)

# Test simulation restart
simulation = Simulation(
    model,
    SSPRK33(),
    dt = 0.01,
    tspan = (0.0, 0.03),
    callbacks = cb_set,
    restart = Restart(
        restartfile = joinpath(cb_1.filedir, cb_1.filename * "_0.02.jld2"),
        end_time = 0.05,
    ),
)
set!(simulation, :swm, h = h, u = u, c = c);
run!(simulation)
@test simulation.integrator.t == 0.05


# Delete test output files
@test isfile(joinpath(cb_1.filedir, cb_1.filename * "_0.01" * ".jld2")) == true
@test isfile(joinpath(cb_2.filedir, cb_2.filename * "_0.02" * ".jld2")) == true

rm(temp_filepath, recursive = true)
